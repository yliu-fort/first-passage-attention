"""First Passage Attention (FPA).

For query t and key s <= t, FPA adds to the logit q_t . k_s / sqrt(d) the first-passage bias

    b_ts = beta * min(0, min_{s < u <= t} phi_u - psi_s),    b_tt = 0,

of a field phi, levels psi and a strength beta > 0 per head. The forward pass uses the hard range minimum; the backward
pass uses in its place the gradient of the soft minimum -tau * log sum_{s < u <= t} exp(-phi_u / tau).

Paper: [arXiv:2610.22714](https://arxiv.org/abs/2610.22714)
"""
import math

import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F

L2E = 1.4426950408889634


def _dtype(*xs):
    return torch.float64 if any(x.dtype == torch.float64 for x in xs) else torch.float32


def _after(T, device):
    return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)


def _range_min(phi):
    T = phi.shape[-1]
    y = phi.unsqueeze(-2).expand(*phi.shape[:-1], T, T).masked_fill(~_after(T, phi.device), float("inf"))
    return torch.cummin(y, dim=-1).values.transpose(-1, -2)


def _range_softmin(phi, tau):
    T = phi.shape[-1]
    big = 1e30
    y = (-phi / tau).unsqueeze(-2).expand(*phi.shape[:-1], T, T).masked_fill(~_after(T, phi.device), -big)
    return (-tau * torch.logcumsumexp(y, dim=-1)).transpose(-1, -2)


def first_passage_bias(phi, psi, beta, tau=1.0):
    """The bias (B, H, T, T), indexed [t, s], of phi and psi (B, H, T) and beta (H,); tau enters only the gradient."""
    if not tau > 0:
        raise ValueError("tau must be positive")
    dt = _dtype(phi, psi)
    phi, psi = phi.to(dt), psi.to(dt)
    if torch.is_tensor(beta):
        beta = beta.reshape(-1, 1, 1)
    soft = _range_softmin(phi, tau)
    x = _range_min(phi).detach() + (soft - soft.detach()) - psi.unsqueeze(-2)
    return beta * torch.clamp(x, max=0.0)


def _causal_attention(q, k, v, bias):
    dt = _dtype(q, bias)
    T = q.shape[-2]
    with torch.autocast(device_type=q.device.type, enabled=False):
        s = (q.to(dt) @ k.to(dt).transpose(-1, -2)) * q.shape[-1] ** -0.5
        s = s + bias.to(dt)
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=q.device).tril(), float("-inf"))
        return (s.softmax(-1) @ v.to(dt)).to(q.dtype)


@triton.jit
def _min_combine(a, b):
    return tl.minimum(a, b)


@triton.jit
def _lse2_combine(a, b):
    m = tl.maximum(a, b)
    return tl.where(m == float("-inf"), m, m + tl.log2(1.0 + tl.exp2(tl.minimum(a, b) - m)))


@triton.jit
def _affine_rev(r1, g1, r2, g2):
    return r1 * r2, g2 + r2 * g1


@triton.jit
def _stats(PHI, PRE, PREL, SUF, SUFL, BMIN, BL, TP, NB, inv_tau, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    zh = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    base = zh * TP + b * BLOCK
    phi = tl.load(PHI + base + offs)
    a2 = -phi * (inv_tau * 1.4426950408889634)
    nxt = offs < BLOCK - 1
    phin = tl.load(PHI + base + offs + 1, mask=nxt, other=float("inf"))
    a2n = tl.where(nxt, -phin * (inv_tau * 1.4426950408889634), float("-inf"))
    tl.store(PRE + base + offs, tl.associative_scan(phi, 0, _min_combine))
    tl.store(PREL + base + offs, tl.associative_scan(a2, 0, _lse2_combine))
    tl.store(SUF + base + offs, tl.associative_scan(phin, 0, _min_combine, reverse=True))
    tl.store(SUFL + base + offs, tl.associative_scan(a2n, 0, _lse2_combine, reverse=True))
    tl.store(BMIN + zh * NB + b, tl.min(phi, 0))
    m = tl.max(a2, 0)
    tl.store(BL + zh * NB + b, m + tl.log2(tl.sum(tl.exp2(a2 - m), 0)))


@triton.jit
def _fwd(Q, K, V, PHI, PSI, PRE, SUF, BMIN, BETA, O, LSE,
         sqz, sqh, sqt, sqd, skz, skh, skt, skd, svz, svh, svt, svd, soz, soh, sot, sod,
         H, T, TP, NB, scale2,
         BLOCK: tl.constexpr, SUB: tl.constexpr, D: tl.constexpr):
    i = NB - 1 - tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    offs = tl.arange(0, BLOCK)
    offk = tl.arange(0, SUB)
    offd = tl.arange(0, D)
    row = (z * H + h) * TP
    brow = (z * H + h) * NB
    betal = tl.load(BETA + h) * 1.4426950408889634
    ti = i * BLOCK + offs
    q = tl.load(Q + z * sqz + h * sqh + ti[:, None] * sqt + offd[None, :] * sqd)
    pre_t = tl.load(PRE + row + ti)
    phi_t = tl.load(PHI + row + ti)
    m_i = tl.full([BLOCK], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK], tl.float32)
    acc = tl.zeros([BLOCK, D], tl.float32)
    for hf in tl.static_range(BLOCK // SUB):
        cols = hf * SUB + offk
        sk = i * BLOCK + cols
        k = tl.load(K + z * skz + h * skh + sk[None, :] * skt + offd[:, None] * skd)
        v = tl.load(V + z * svz + h * svh + sk[:, None] * svt + offd[None, :] * svd)
        psi_s = tl.load(PSI + row + sk)
        later = offs[:, None] > cols[None, :]
        y = tl.where(later, tl.broadcast_to(phi_t[:, None], (BLOCK, SUB)), float("inf"))
        x = tl.associative_scan(y, 0, _min_combine) - psi_s[None, :]
        s = tl.dot(q, k, input_precision="ieee") * scale2 + betal * tl.minimum(x, 0.0)
        ok = (offs[:, None] >= cols[None, :]) & (sk[None, :] < T)
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    mid = tl.full([1], float("inf"), tl.float32)
    for jj in range(0, i):
        j = i - 1 - jj
        for hf in tl.static_range(BLOCK // SUB):
            sk = j * BLOCK + hf * SUB + offk
            k = tl.load(K + z * skz + h * skh + sk[None, :] * skt + offd[:, None] * skd)
            v = tl.load(V + z * svz + h * svh + sk[:, None] * svt + offd[None, :] * svd)
            psi_s = tl.load(PSI + row + sk)
            suf_s = tl.load(SUF + row + sk)
            x = tl.minimum(tl.minimum(suf_s, mid)[None, :], pre_t[:, None]) - psi_s[None, :]
            s = tl.dot(q, k, input_precision="ieee") * scale2 + betal * tl.minimum(x, 0.0)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp2(s - m_new[:, None])
            alpha = tl.exp2(m_i - m_new)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
        mid = tl.minimum(mid, tl.load(BMIN + brow + j))
    acc = acc / l_i[:, None]
    tl.store(O + z * soz + h * soh + ti[:, None] * sot + offd[None, :] * sod, acc.to(O.dtype.element_ty))
    tl.store(LSE + row + ti, m_i + tl.log2(l_i))


@triton.jit
def _bwd_delta(O, DO, DELTA, soz, soh, sot, sod, sdz, sdh, sdt, sdd, H, TP, BLOCK: tl.constexpr, D: tl.constexpr):
    i = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    ti = i * BLOCK + tl.arange(0, BLOCK)
    offd = tl.arange(0, D)
    o = tl.load(O + z * soz + h * soh + ti[:, None] * sot + offd[None, :] * sod)
    do = tl.load(DO + z * sdz + h * sdh + ti[:, None] * sdt + offd[None, :] * sdd)
    tl.store(DELTA + (z * H + h) * TP + ti, tl.sum(do.to(tl.float32) * o.to(tl.float32), 1))


@triton.jit
def _bwd_kv(Q, K, V, DO, LSE, DELTA, PHI, PSI, PRE, PREL, SUF, SUFL, BMIN, BL, BETA,
            DK, DV, DPSI, W, TBUF, DBETA, DQD, GD,
            sqz, sqh, sqt, sqd, skz, skh, skt, skd, svz, svh, svt, svd, sdz, sdh, sdt, sdd, sgz, sgh, sgt, sgd,
            H, T, TP, NB, scale, inv_tau,
            BLOCK: tl.constexpr, D: tl.constexpr):
    j = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    offs = tl.arange(0, BLOCK)
    offd = tl.arange(0, D)
    row = (z * H + h) * TP
    brow = (z * H + h) * NB
    beta = tl.load(BETA + h)
    betal = beta * 1.4426950408889634
    scale2 = scale * 1.4426950408889634
    il2 = inv_tau * 1.4426950408889634
    sj = j * BLOCK + offs
    k = tl.load(K + z * skz + h * skh + sj[:, None] * skt + offd[None, :] * skd)
    v = tl.load(V + z * svz + h * svh + sj[:, None] * svt + offd[None, :] * svd)
    psi_s = tl.load(PSI + row + sj)
    suf_s = tl.load(SUF + row + sj)
    sufl_s = tl.load(SUFL + row + sj)
    has_c = suf_s < float("inf")
    c_s = tl.where(has_c, -suf_s * il2, float("-inf"))
    dk = tl.zeros([BLOCK, D], tl.float32)
    dv = tl.zeros([BLOCK, D], tl.float32)
    gsum = tl.zeros([BLOCK], tl.float32)
    w = tl.zeros([BLOCK], tl.float32)
    dbeta = tl.zeros([BLOCK], tl.float32)
    q = tl.load(Q + z * sqz + h * sqh + sj[:, None] * sqt + offd[None, :] * sqd)
    do = tl.load(DO + z * sdz + h * sdh + sj[:, None] * sdt + offd[None, :] * sdd)
    lse = tl.load(LSE + row + sj)
    delta = tl.load(DELTA + row + sj)
    phi_t = tl.load(PHI + row + sj)
    later = offs[None, :] > offs[:, None]
    y = tl.where(later, tl.broadcast_to(phi_t[None, :], (BLOCK, BLOCK)), float("inf"))
    x = tl.associative_scan(y, 1, _min_combine) - psi_s[:, None]
    xm = tl.minimum(x, 0.0)
    st = tl.dot(k, tl.trans(q), input_precision="ieee") * scale2 + betal * xm
    ok = (offs[None, :] >= offs[:, None]) & (sj[:, None] < T)
    pt = tl.where(ok, tl.exp2(st - lse[None, :]), 0.0)
    dv += tl.dot(pt.to(do.dtype), do, input_precision="ieee")
    dpt = tl.dot(v, tl.trans(do), input_precision="ieee")
    dst = pt * (dpt - delta[None, :])
    dk += tl.dot(dst.to(q.dtype), q, input_precision="ieee")
    tl.store(DQD + (row + sj)[:, None] * D + offd[None, :], tl.dot(tl.trans(dst.to(k.dtype)), k, input_precision="ieee"))
    strict = later & (sj[:, None] < T)
    g = tl.where(strict & (x <= 0.0), dst, 0.0)
    gsum += tl.sum(g, 1)
    dbeta += tl.sum(tl.where(strict, dst * xm, 0.0), 1)
    tl.store(GD + ((brow + j) * BLOCK + offs[:, None]) * BLOCK + offs[None, :], g)
    mid = tl.full([1], float("inf"), tl.float32)
    midl = tl.full([1], float("-inf"), tl.float32)
    for i in range(j + 1, NB):
        ti = i * BLOCK + offs
        q = tl.load(Q + z * sqz + h * sqh + ti[:, None] * sqt + offd[None, :] * sqd)
        do = tl.load(DO + z * sdz + h * sdh + ti[:, None] * sdt + offd[None, :] * sdd)
        lse = tl.load(LSE + row + ti)
        delta = tl.load(DELTA + row + ti)
        pre_t = tl.load(PRE + row + ti)
        prel_t = tl.load(PREL + row + ti)
        x = tl.minimum(tl.minimum(suf_s, mid)[:, None], pre_t[None, :]) - psi_s[:, None]
        xm = tl.minimum(x, 0.0)
        st = tl.dot(k, tl.trans(q), input_precision="ieee") * scale2 + betal * xm
        pt = tl.exp2(st - lse[None, :])
        dv += tl.dot(pt.to(do.dtype), do, input_precision="ieee")
        dpt = tl.dot(v, tl.trans(do), input_precision="ieee")
        dst = pt * (dpt - delta[None, :])
        dk += tl.dot(dst.to(q.dtype), q, input_precision="ieee")
        g = tl.where(x <= 0.0, dst, 0.0)
        gsum += tl.sum(g, 1)
        dbeta += tl.sum(dst * xm, 1)
        cmid = -mid * il2
        rs = tl.maximum(c_s, cmid)
        valid = rs > float("-inf")
        rs = tl.where(valid, rs, 0.0)
        sv = tl.exp2(sufl_s - rs) + tl.exp2(midl - rs)
        fsum = tl.where(valid, tl.sum(g / (sv[:, None] + tl.exp2(prel_t[None, :] - rs[:, None])), 1), 0.0)
        w += tl.exp2(c_s - rs) * fsum
        if i > j + 1:
            tl.store(TBUF + (brow + i) * NB + j, tl.sum(tl.exp2(cmid - rs) * fsum, 0))
        mid = tl.minimum(mid, tl.load(BMIN + brow + i))
        midl = _lse2_combine(midl, tl.load(BL + brow + i))
    tl.store(DK + z * sgz + h * sgh + sj[:, None] * sgt + offd[None, :] * sgd, (dk * scale).to(DK.dtype.element_ty))
    tl.store(DV + z * sgz + h * sgh + sj[:, None] * sgt + offd[None, :] * sgd, dv.to(DV.dtype.element_ty))
    tl.store(DPSI + row + sj, -beta * gsum)
    tl.store(W + row + sj, w)
    tl.store(DBETA + brow + j, tl.sum(dbeta, 0))


@triton.jit
def _bwd_dphi_diag(PHI, GD, DPHI4, TP, NB, inv_tau, BLOCK: tl.constexpr):
    j = tl.program_id(0)
    zh = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    il2 = inv_tau * 1.4426950408889634
    tj = zh * TP + j * BLOCK + offs
    a2_t = -tl.load(PHI + tj) * il2
    a2_next = -tl.load(PHI + tj + 1, mask=offs < BLOCK - 1, other=0.0) * il2
    g = tl.load(GD + ((zh * NB + j) * BLOCK + offs[:, None]) * BLOCK + offs[None, :])
    later = offs[None, :] > offs[:, None]
    ya = tl.where(later, tl.broadcast_to(a2_t[None, :], (BLOCK, BLOCK)), float("-inf"))
    logz = tl.associative_scan(ya, 1, _lse2_combine)
    rr = tl.where(later & (offs[None, :] < BLOCK - 1), 1.0 / (1.0 + tl.exp2(a2_next[None, :] - logz)), 0.0)
    _, lam = tl.associative_scan((rr, g), 1, _affine_rev, reverse=True)
    tl.store(DPHI4 + tj, tl.sum(tl.where(later, tl.exp2(a2_t[None, :] - logz), 0.0) * lam, 0))


@triton.jit
def _bwd_q(Q, K, V, DO, LSE, DELTA, PHI, PSI, PRE, PREL, SUF, SUFL, BMIN, BL, BETA, DQD, DPHI4,
           DQ, DPHI2,
           sqz, sqh, sqt, sqd, skz, skh, skt, skd, svz, svh, svt, svd, sdz, sdh, sdt, sdd, sgz, sgh, sgt, sgd,
           H, T, TP, NB, scale, inv_tau,
           BLOCK: tl.constexpr, SUB: tl.constexpr, D: tl.constexpr):
    i = NB - 1 - tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    offs = tl.arange(0, BLOCK)
    offk = tl.arange(0, SUB)
    offd = tl.arange(0, D)
    row = (z * H + h) * TP
    brow = (z * H + h) * NB
    betal = tl.load(BETA + h) * 1.4426950408889634
    scale2 = scale * 1.4426950408889634
    il2 = inv_tau * 1.4426950408889634
    ti = i * BLOCK + offs
    q = tl.load(Q + z * sqz + h * sqh + ti[:, None] * sqt + offd[None, :] * sqd)
    do = tl.load(DO + z * sdz + h * sdh + ti[:, None] * sdt + offd[None, :] * sdd)
    delta = tl.load(DELTA + row + ti)
    lse = tl.load(LSE + row + ti)
    pre_t = tl.load(PRE + row + ti)
    prel_t = tl.load(PREL + row + ti)
    c_t = -pre_t * il2
    bpre_t = tl.exp2(prel_t - c_t)
    dq = tl.load(DQD + (row + ti)[:, None] * D + offd[None, :])
    r = tl.zeros([BLOCK], tl.float32)
    mid = tl.full([1], float("inf"), tl.float32)
    midl = tl.full([1], float("-inf"), tl.float32)
    for jj in range(0, i):
        j = i - 1 - jj
        mterm = tl.exp2(midl - c_t) + bpre_t
        for hf in tl.static_range(BLOCK // SUB):
            sk = j * BLOCK + hf * SUB + offk
            k = tl.load(K + z * skz + h * skh + sk[:, None] * skt + offd[None, :] * skd)
            v = tl.load(V + z * svz + h * svh + sk[:, None] * svt + offd[None, :] * svd)
            psi_s = tl.load(PSI + row + sk)
            suf_s = tl.load(SUF + row + sk)
            sufl_s = tl.load(SUFL + row + sk)
            x = tl.minimum(tl.minimum(suf_s, mid)[None, :], pre_t[:, None]) - psi_s[None, :]
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale2 + betal * tl.minimum(x, 0.0)
            p = tl.exp2(s - lse[:, None])
            dp = tl.dot(do, tl.trans(v), input_precision="ieee")
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k.dtype), k, input_precision="ieee")
            g = tl.where(x <= 0.0, ds, 0.0)
            r += tl.sum(g / (tl.exp2(sufl_s[None, :] - c_t[:, None]) + mterm[:, None]), 1)
        mid = tl.minimum(mid, tl.load(BMIN + brow + j))
        midl = _lse2_combine(midl, tl.load(BL + brow + j))
    a2_t = -tl.load(PHI + row + ti) * il2
    e = tl.where(offs[:, None] <= offs[None, :], tl.exp2(a2_t[:, None] - c_t[None, :]), 0.0)
    part2 = tl.sum(e * r[None, :], 1)
    tl.store(DQ + z * sgz + h * sgh + ti[:, None] * sgt + offd[None, :] * sgd, (dq * scale).to(DQ.dtype.element_ty))
    tl.store(DPHI2 + row + ti, part2 + tl.load(DPHI4 + row + ti))


@triton.jit
def _bwd_dphi(PHI, SUF, BMIN, W, TBUF, DPHI2, BETA, DPHI, H, TP, NB, inv_tau,
              BLOCK: tl.constexpr, NBP: tl.constexpr):
    U = tl.program_id(0)
    h = tl.program_id(1)
    z = tl.program_id(2)
    zh = z * H + h
    il2 = inv_tau * 1.4426950408889634
    nb = tl.arange(0, NBP)
    cu = tl.where(nb < NB, -tl.load(BMIN + zh * NB + nb, mask=nb < NB, other=0.0) * il2, float("-inf"))
    cU = tl.sum(tl.where(nb == U, cu, 0.0), 0)
    cub = tl.broadcast_to(cu[None, :], (NBP, NBP))
    A = tl.max(tl.where((nb[None, :] > nb[:, None]) & (nb[None, :] <= U), cub, float("-inf")), 1)
    Bv = tl.max(tl.where((nb[None, :] >= U) & (nb[None, :] < nb[:, None]), cub, float("-inf")), 1)
    inside = (nb[:, None] > U) & (nb[:, None] < NB) & (nb[None, :] < U)
    tij = tl.load(TBUF + (zh * NB + nb[:, None]) * NB + nb[None, :], mask=inside, other=0.0)
    cmid = tl.maximum(Bv[:, None], A[None, :])
    S = tl.sum(tl.sum(tl.where(inside, tl.exp2(cU - cmid) * tij, 0.0), 1), 0)
    offs = tl.arange(0, BLOCK)
    pos = zh * TP + U * BLOCK + offs
    a2 = -tl.load(PHI + pos) * il2
    part3 = tl.where((U > 0) & (U < NB - 1), tl.exp2(a2 - cU) * S, 0.0)
    suf_s = tl.load(SUF + pos)
    has_c = suf_s < float("inf")
    c_s = tl.where(has_c, -suf_s * il2, 0.0)
    e = tl.where((offs[:, None] > offs[None, :]) & has_c[None, :], tl.exp2(a2[:, None] - c_s[None, :]), 0.0)
    part1 = tl.sum(e * tl.load(W + pos)[None, :], 1)
    tl.store(DPHI + pos, tl.load(BETA + h) * (part1 + tl.load(DPHI2 + pos) + part3))


SUB = {"fwd": 32, "q": 64}


def _launch(x):
    return {"num_stages": 1} if x.dtype == torch.float32 else {}


def _pad_t(x, TP):
    T = x.shape[2]
    if T == TP:
        return x.contiguous()
    return F.pad(x, (0, TP - T) if x.dim() == 3 else (0, 0, 0, TP - T)).contiguous()


def _block_stats(phip, tau, block):
    B, H, TP = phip.shape
    NB = TP // block
    st = {n: torch.empty_like(phip) for n in ("pre", "preL", "suf", "sufL")}
    st["bmin"] = torch.empty(B, H, NB, dtype=torch.float32, device=phip.device)
    st["bL"] = torch.empty_like(st["bmin"])
    _stats[(NB, B * H)](phip, st["pre"], st["preL"], st["suf"], st["sufL"], st["bmin"], st["bL"], TP, NB, 1.0 / tau,
                        BLOCK=block, num_warps=1)
    return st


def _forward(q, k, v, phi, psi, beta, tau, block, sub):
    B, H, T, D = q.shape
    TP = -(-T // block) * block
    NB = TP // block
    qp, kp, vp = (_pad_t(x, TP) for x in (q, k, v))
    phip, psip = _pad_t(phi.float(), TP), _pad_t(psi.float(), TP)
    st = _block_stats(phip, tau, block)
    betaf = beta.float().contiguous()
    o = torch.empty_like(qp)
    lse = torch.empty(B, H, TP, dtype=torch.float32, device=q.device)
    _fwd[(NB, H, B)](qp, kp, vp, phip, psip, st["pre"], st["suf"], st["bmin"], betaf, o, lse,
                     *qp.stride(), *kp.stride(), *vp.stride(), *o.stride(),
                     H, T, TP, NB, D ** -0.5 * L2E,
                     BLOCK=block, SUB=sub["fwd"], D=D, **_launch(qp))
    return o, lse, (qp, kp, vp, phip, psip, betaf, st, T, TP, NB)


class _FirstPassageAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, phi, psi, beta, tau, block, sub):
        o, lse, (qp, kp, vp, phip, psip, betaf, st, T, TP, NB) = _forward(q, k, v, phi, psi, beta, tau, block, sub)
        ctx.save_for_backward(qp, kp, vp, o, lse, phip, psip, betaf)
        ctx.st, ctx.meta = st, (T, TP, NB, tau, block, sub)
        return o[:, :, :T]

    @staticmethod
    def backward(ctx, do):
        qp, kp, vp, o, lse, phip, psip, betaf = ctx.saved_tensors
        T, TP, NB, tau, block, sub = ctx.meta
        st = ctx.st
        B, H, _, D = qp.shape
        dop = _pad_t(do.to(qp.dtype), TP)
        dq, dk, dv = torch.empty_like(qp), torch.empty_like(kp), torch.empty_like(vp)
        delta = torch.empty(B, H, TP, dtype=torch.float32, device=qp.device)
        dpsi, w, dphi2, dphi4, dphi = (torch.empty_like(delta) for _ in range(5))
        dqd = torch.empty(B, H, TP, D, dtype=torch.float32, device=qp.device)
        gd = torch.empty(B, H, NB, block, block, dtype=torch.float32, device=qp.device)
        tbuf = torch.empty(B, H, NB, NB, dtype=torch.float32, device=qp.device)
        dbeta = torch.empty(B, H, NB, dtype=torch.float32, device=qp.device)
        args = (H, T, TP, NB, D ** -0.5, 1.0 / tau)
        _bwd_delta[(NB, H, B)](o, dop, delta, *o.stride(), *dop.stride(), H, TP, BLOCK=block, D=D)
        _bwd_kv[(NB, H, B)](qp, kp, vp, dop, lse, delta, phip, psip, st["pre"], st["preL"], st["suf"], st["sufL"],
                            st["bmin"], st["bL"], betaf, dk, dv, dpsi, w, tbuf, dbeta, dqd, gd,
                            *qp.stride(), *kp.stride(), *vp.stride(), *dop.stride(), *dk.stride(),
                            *args, BLOCK=block, D=D, **_launch(qp))
        _bwd_dphi_diag[(NB, B * H)](phip, gd, dphi4, TP, NB, 1.0 / tau, BLOCK=block, num_warps=8)
        _bwd_q[(NB, H, B)](qp, kp, vp, dop, lse, delta, phip, psip, st["pre"], st["preL"], st["suf"], st["sufL"],
                           st["bmin"], st["bL"], betaf, dqd, dphi4, dq, dphi2,
                           *qp.stride(), *kp.stride(), *vp.stride(), *dop.stride(), *dq.stride(),
                           *args, BLOCK=block, SUB=sub["q"], D=D, **_launch(qp))
        _bwd_dphi[(NB, H, B)](phip, st["suf"], st["bmin"], w, tbuf, dphi2, betaf, dphi, H, TP, NB, 1.0 / tau,
                              BLOCK=block, NBP=max(16, triton.next_power_of_2(NB)), num_warps=4)
        return (dq[:, :, :T], dk[:, :, :T], dv[:, :, :T], dphi[..., :T], dpsi[..., :T], dbeta.sum((0, 2)),
                None, None, None)


def _pow2(n):
    return n >= 16 and n & (n - 1) == 0


def first_passage_attention(q, k, v, phi, psi, beta, tau=1.0, impl=None, block=64):
    """o (B, H, T, D) of q, k, v (B, H, T, D), phi, psi (B, H, T) and beta (H,).

    impl: "triton" (q, k, v all fp16 or all fp32; D a power of two >= 16), "reference" (any device), or None for
    "triton" on CUDA tensors and "reference" otherwise. block: the kernels' block of positions (at most 32 in fp32)."""
    if impl is None:
        impl = "triton" if q.is_cuda else "reference"
    if impl == "reference":
        return _causal_attention(q, k, v, first_passage_bias(phi, psi, beta, tau))
    if impl != "triton":
        raise ValueError(f"unknown implementation {impl!r}")
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k and v must have the same shape (B, H, T, D)")
    if q.dtype not in (torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k and v must all be fp16 or all fp32")
    B, H, T, D = q.shape
    if phi.shape != (B, H, T) or psi.shape != (B, H, T):
        raise ValueError("phi and psi must have shape (B, H, T)")
    if not _pow2(D) or not _pow2(block):
        raise ValueError("the head dimension and the block must be powers of two >= 16")
    if not torch.is_tensor(beta):
        beta = torch.tensor(float(beta), device=q.device)
    beta = beta.reshape(-1)
    if beta.numel() == 1:
        beta = beta.expand(H)
    if beta.shape != (H,):
        raise ValueError("beta must have shape (H,) or a single value")
    if not tau > 0:
        raise ValueError("tau must be positive")
    if q.dtype == torch.float32:
        block = min(block, 32)
    return _FirstPassageAttention.apply(q, k, v, phi, psi, beta, tau, block, {n: min(s, block) for n, s in SUB.items()})


class Field(nn.Module):
    """phi = W + g, psi = W + h with W = cumsum(delta), and beta = softplus(beta_raw), per head, from x (B, T, d_model).
    delta, g and h are linear maps of x, computed in fp32."""

    def __init__(self, d_model, n_heads, beta_init=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.proj = nn.Linear(d_model, 3 * n_heads)
        self.beta_raw = nn.Parameter(torch.full((n_heads,), beta_init + math.log(-math.expm1(-beta_init))))

    def forward(self, x):
        B, T, _ = x.shape
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = F.linear(x.to(dt), self.proj.weight.to(dt), self.proj.bias.to(dt))
        delta, g, h = out.view(B, T, 3, self.n_heads).permute(2, 0, 3, 1)
        w = torch.cumsum(delta, dim=-1)
        return w + g, w + h, F.softplus(self.beta_raw)


class FirstPassageAttention(nn.Module):
    """Multi-head causal FPA without positional encoding: x (B, T, d_model) -> (B, T, d_model)."""

    def __init__(self, d_model, n_heads, tau=1.0, beta_init=1.0, impl=None, block=64):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be a multiple of n_heads")
        self.h, self.d = n_heads, d_model // n_heads
        self.tau, self.impl, self.block = tau, impl, block
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.field = Field(d_model, n_heads, beta_init)

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        phi, psi, beta = self.field(x)
        o = first_passage_attention(q, k, v, phi, psi, beta, tau=self.tau, impl=self.impl, block=self.block)
        return self.out(o.transpose(1, 2).reshape(B, T, -1))
