import math
import os

import pytest
import torch

from kernel import FirstPassageAttention, first_passage_attention, first_passage_bias

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
KERNEL_DEVICE = "cpu" if INTERPRET else ("cuda" if torch.cuda.is_available() else None)
needs_kernel = pytest.mark.skipif(KERNEL_DEVICE is None, reason="needs a CUDA GPU or TRITON_INTERPRET=1")
gpu_only = pytest.mark.skipif(INTERPRET, reason="too slow in Triton's interpreter")
if INTERPRET:
    pytestmark = [pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning"),
                  pytest.mark.filterwarnings("ignore:overflow encountered in exp2:RuntimeWarning"),
                  pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")]

NAMES = ("o", "dq", "dk", "dv", "dphi", "dpsi", "dbeta")
TOL = {torch.float16: 2e-3, torch.float32: 2e-4}
TOL_DBETA_FP16 = 1e-2


def random_field(B, H, T, seed, ties=False):
    g = torch.Generator().manual_seed(seed)
    phi = torch.randn(B, H, T, generator=g, dtype=torch.float64).cumsum(-1)
    if ties:
        phi = torch.round(phi)
    psi = phi + 0.5 * torch.randn(B, H, T, generator=g, dtype=torch.float64)
    beta = torch.rand(H, generator=g, dtype=torch.float64) + 0.5
    return phi, psi, beta


def bias_by_loops(phi, psi, beta):
    B, H, T = phi.shape
    out = torch.zeros(B, H, T, T, dtype=torch.float64)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                for s in range(t):
                    m = min(phi[b, h, u].item() for u in range(s + 1, t + 1))
                    out[b, h, t, s] = beta[h].item() * min(0.0, m - psi[b, h, s].item())
    return out


def test_bias_is_the_definition():
    for ties in (False, True):
        phi, psi, beta = random_field(2, 3, 12, seed=1 + ties, ties=ties)
        assert torch.equal(first_passage_bias(phi, psi, beta), bias_by_loops(phi, psi, beta))


def test_attention_is_softmax_with_the_bias():
    B, H, T, D = 2, 3, 10, 8
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(B, H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
    phi, psi, beta = random_field(B, H, T, seed=4)
    bias = bias_by_loops(phi, psi, beta)
    o = torch.zeros(B, H, T, D, dtype=torch.float64)
    for t in range(T):
        logits = (q[:, :, t:t + 1] @ k[:, :, :t + 1].transpose(-1, -2)).squeeze(-2) / math.sqrt(D)
        logits = logits + bias[:, :, t, :t + 1]
        o[:, :, t] = (logits.softmax(-1).unsqueeze(-2) @ v[:, :, :t + 1]).squeeze(-2)
    ref = first_passage_attention(q, k, v, phi, psi, beta, impl="reference")
    assert torch.allclose(ref, o, rtol=1e-12, atol=1e-12)


def test_gradient_of_the_bias():
    B, H, T, tau = 1, 2, 9, 0.7
    phi, psi, beta = random_field(B, H, T, seed=5)
    G = torch.randn(B, H, T, T, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    args = [t.clone().requires_grad_() for t in (phi, psi, beta)]
    dphi, dpsi, dbeta = torch.autograd.grad((first_passage_bias(*args, tau=tau) * G).sum(), args)
    e_phi, e_psi, e_beta = torch.zeros_like(phi), torch.zeros_like(psi), torch.zeros_like(beta)
    for h in range(H):
        for t in range(T):
            for s in range(t):
                rng = phi[0, h, s + 1:t + 1]
                x = rng.min().item() - psi[0, h, s].item()
                gate = 1.0 if x <= 0 else 0.0
                g = G[0, h, t, s].item()
                e_phi[0, h, s + 1:t + 1] += beta[h] * gate * g * torch.softmax(-rng / tau, 0)
                e_psi[0, h, s] -= beta[h] * gate * g
                e_beta[h] += min(0.0, x) * g
    for got, want in ((dphi, e_phi), (dpsi, e_psi), (dbeta, e_beta)):
        assert torch.allclose(got, want, rtol=1e-10, atol=1e-12)


def test_causal():
    B, H, T, D, t0 = 1, 2, 16, 8, 9
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(B, H, T, D, generator=g, dtype=torch.float64) for _ in range(3))
    phi, psi, beta = random_field(B, H, T, seed=8)
    o = first_passage_attention(q, k, v, phi, psi, beta, impl="reference")
    later = [t.clone() for t in (q, k, v, phi, psi)]
    for t in later:
        t[:, :, t0 + 1:] = torch.randn(t[:, :, t0 + 1:].shape, generator=g, dtype=torch.float64) * 10
    o2 = first_passage_attention(*later, beta, impl="reference")
    assert torch.equal(o[:, :, :t0 + 1], o2[:, :, :t0 + 1])


def test_bias_never_rises():
    phi, psi, beta = random_field(2, 3, 24, seed=10)
    b = first_passage_bias(phi, psi, beta)
    T = phi.shape[-1]
    keys = torch.ones(T, T, dtype=torch.bool).tril()[:-1]
    assert (b[:, :, 1:][..., keys] <= b[:, :, :-1][..., keys]).all()


def make_inputs(B, H, T, D, phi_scale=1.0, ties=False, seed=0, dtype=torch.float16, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(B, H, T, D, generator=g).to(dtype) for _ in range(3))
    phi = torch.randn(B, H, T, generator=g).cumsum(-1) * phi_scale
    if ties:
        phi = torch.round(phi)
    psi = phi + 0.5 * torch.randn(B, H, T, generator=g)
    beta = torch.rand(H, generator=g) + 0.5
    do = torch.randn(B, H, T, D, generator=g).to(dtype)
    return [t.to(device) for t in (q, k, v, phi, psi, beta, do)]


def rel_err(x, ref):
    return ((x.double() - ref).abs().max() / ref.abs().max().clamp(min=1e-30)).item()


def kernel_and_reference(q, k, v, phi, psi, beta, do, block, tau=1.0):
    a = [t.clone().requires_grad_() for t in (q, k, v, phi, psi, beta)]
    o = first_passage_attention(*a, tau=tau, impl="triton", block=block)
    kern = [o.detach().cpu()] + [x.cpu() for x in torch.autograd.grad(o, a, do)]
    r = [t.detach().cpu().double().requires_grad_() for t in (q, k, v, phi, psi, beta)]
    o = first_passage_attention(*r, tau=tau, impl="reference")
    ref = [o.detach()] + list(torch.autograd.grad(o, r, do.cpu().double()))
    return kern, ref


CASES = [
    (1, 2, 64, 64, 32, 1.0, False, torch.float16, 100),
    (2, 3, 100, 64, 32, 1.0, False, torch.float16, 101),
    (1, 2, 128, 64, 64, 1.0, False, torch.float16, 102),
    (1, 2, 96, 64, 32, 40.0, False, torch.float16, 103),
    (1, 2, 96, 64, 32, 1.0, True, torch.float16, 104),
    (1, 2, 80, 32, 16, 1.0, False, torch.float16, 110),
    (1, 2, 100, 64, 32, 1.0, False, torch.float32, 111),
    (1, 2, 96, 16, 16, 40.0, False, torch.float32, 112),
    pytest.param(2, 4, 512, 64, 64, 1.0, False, torch.float16, 105, marks=gpu_only),
    pytest.param(1, 2, 2048, 64, 64, 1.0, False, torch.float16, 106, marks=gpu_only),
    pytest.param(1, 2, 2048, 64, 32, 1.0, False, torch.float32, 200, marks=gpu_only),
    pytest.param(1, 2, 1000, 64, 32, 40.0, False, torch.float32, 201, marks=gpu_only),
]


@needs_kernel
@pytest.mark.parametrize("B,H,T,D,block,scale,ties,dtype,seed", CASES)
def test_kernel_matches_the_reference(B, H, T, D, block, scale, ties, dtype, seed):
    inp = make_inputs(B, H, T, D, scale, ties, seed=seed, dtype=dtype, device=KERNEL_DEVICE)
    kern, ref = kernel_and_reference(*inp, block)
    for name, x, r in zip(NAMES, kern, ref):
        assert x.dtype == (dtype if name in ("o", "dq", "dk", "dv") else torch.float32), name
        assert torch.isfinite(x).all(), name
        tol = TOL_DBETA_FP16 if (name == "dbeta" and dtype == torch.float16) else TOL[dtype]
        assert rel_err(x, r) <= tol, (name, rel_err(x, r))


@needs_kernel
def test_kernel_temperature():
    inp = make_inputs(1, 2, 72, 32, seed=11, dtype=torch.float32, device=KERNEL_DEVICE)
    kern, ref = kernel_and_reference(*inp, block=16, tau=0.25)
    for name, x, r in zip(NAMES, kern, ref):
        assert rel_err(x, r) <= TOL[torch.float32], (name, rel_err(x, r))


@needs_kernel
def test_kernel_deterministic():
    inp = make_inputs(2, 2, 96, 32, seed=12, device=KERNEL_DEVICE)
    runs = [kernel_and_reference(*inp, block=32)[0] for _ in range(2)]
    assert all(torch.equal(x, y) for x, y in zip(*runs))


def layer_and_input(impl, d_model=64, n_heads=4, T=50, seed=0):
    torch.manual_seed(seed)
    layer = FirstPassageAttention(d_model, n_heads, impl=impl)
    x = torch.randn(2, T, d_model, generator=torch.Generator().manual_seed(seed + 1))
    return layer, x


def test_layer_causal():
    layer, x = layer_and_input("reference")
    t0 = 30
    x2 = x.clone()
    x2[:, t0 + 1:] = torch.randn(x2[:, t0 + 1:].shape) * 5
    with torch.no_grad():
        assert torch.allclose(layer(x)[:, :t0 + 1], layer(x2)[:, :t0 + 1], rtol=0, atol=1e-6)


def test_layer_gradients():
    layer, x = layer_and_input("reference")
    layer(x).square().mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0, name


@needs_kernel
@pytest.mark.parametrize("amp", [False, pytest.param(True, marks=pytest.mark.skipif(
    KERNEL_DEVICE != "cuda", reason="fp16 autocast on the GPU"))])
def test_layer_kernel_matches_the_reference(amp):
    out = []
    for impl in ("triton", "reference"):
        layer, x = layer_and_input(impl)
        layer, x = layer.to(KERNEL_DEVICE), x.to(KERNEL_DEVICE)
        with torch.autocast(KERNEL_DEVICE, dtype=torch.float16, enabled=amp):
            y = layer(x)
        loss = y.float().square().mean()
        out.append([y.detach().float()] + list(torch.autograd.grad(loss, list(layer.parameters()))))
    tol = 5e-3 if amp else 1e-4
    for a, b in zip(*out):
        assert ((a - b).norm() / b.norm()).item() <= tol
