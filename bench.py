import argparse
import json
import statistics

import torch
import triton

from kernel import first_passage_attention


def timeit(fn, warm=3, n=15):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description="Time the Triton kernels: forward and forward+backward of one call.")
    ap.add_argument("--shape", default="4,12,2048,64", help="B,H,T,D")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--json", help="write the results to this file")
    a = ap.parse_args()
    B, H, T, D = (int(x) for x in a.shape.split(","))
    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    g = torch.Generator().manual_seed(0)
    q, k, v, do = (torch.randn(B, H, T, D, generator=g).to("cuda", dtype) for _ in range(4))
    phi = torch.randn(B, H, T, generator=g).cumsum(-1).cuda()
    psi = phi + 0.5 * torch.randn(B, H, T, generator=g).cuda()
    beta = (torch.rand(H, generator=g) + 0.5).cuda()
    inputs = (q, k, v, phi, psi, beta)

    def fwd():
        with torch.no_grad():
            first_passage_attention(*inputs, impl="triton")

    def fwd_bwd():
        x = [t.clone().requires_grad_() for t in inputs]
        first_passage_attention(*x, impl="triton").backward(do)

    rounds = {"fwd": [], "fwd_bwd": []}
    for _ in range(a.rounds):
        rounds["fwd"].append(timeit(fwd))
        rounds["fwd_bwd"].append(timeit(fwd_bwd))
    med = {key: statistics.median(ts) for key, ts in rounds.items()}
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, triton {triton.__version__}")
    print(f"(B, H, T, D) = ({B}, {H}, {T}, {D}), {a.dtype}, median of {a.rounds} rounds")
    print(f"forward {med['fwd']:.3f} ms, forward+backward {med['fwd_bwd']:.3f} ms")
    if a.json:
        res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
               "shape": [B, H, T, D], "dtype": a.dtype, "median_ms": {key: round(t, 4) for key, t in med.items()},
               "rounds_ms": {key: [round(t, 4) for t in ts] for key, ts in rounds.items()}}
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
