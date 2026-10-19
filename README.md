# First Passage Attention

Triton kernels (forward and backward) and a PyTorch reference for First Passage Attention (FPA), as used to train the 110M-parameter FPA language model.

## Definition

For query $t$ and key $s\le t$ of a head:

$$o_t=\sum_{s\le t}\operatorname{softmax}_s\Bigl(\frac{q_t^\top k_s}{\sqrt d}+b_{ts}\Bigr)v_s,\qquad b_{ts}=\beta\min\Bigl(0,\ \min_{s<u\le t}\phi_u-\psi_s\Bigr),\qquad b_{tt}=0.$$

Backward: the gradient for $\phi$ uses the soft minimum $-\tau\log\sum_{s<u\le t}e^{-\phi_u/\tau}$ ($\tau=1$) in place of $\min_{s<u\le t}\phi_u$; the other gradients are exact.

Layer (`FirstPassageAttention`): per head, $\delta_t$, $g_t$, $h_t$ are linear maps of $x_t$, $W_t=\sum_{u\le t}\delta_u$, $\phi=W+g$, $\psi=W+h$, $\beta=\mathrm{softplus}(\cdot)$.

## Requirements

PyTorch and Triton (included in PyTorch's CUDA wheels on Linux); pytest for the tests. Tested with Python 3.12, PyTorch 2.5.1+cu121, Triton 3.1.0.

## Usage

```python
import torch
from kernel import FirstPassageAttention, first_passage_attention

B, H, T, D = 4, 12, 2048, 64
q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
phi = torch.randn(B, H, T, device="cuda").cumsum(-1).requires_grad_()
psi = (phi.detach() + torch.randn(B, H, T, device="cuda")).requires_grad_()
beta = torch.ones(H, device="cuda", requires_grad=True)
o = first_passage_attention(q, k, v, phi, psi, beta)  # impl="reference" for the PyTorch reference
o.float().sum().backward()

layer = FirstPassageAttention(d_model=768, n_heads=12).cuda()
with torch.autocast("cuda", dtype=torch.float16):
    y = layer(torch.randn(4, 2048, 768, device="cuda"))
```

Kernel inputs: `q`, `k`, `v` all fp16 or all fp32; head dimension a power of two, at least 16; any length.

## Tests

```bash
pytest test.py                      # with a GPU: all tests; without: the reference tests
TRITON_INTERPRET=1 pytest test.py   # without a GPU: the kernels in Triton's CPU interpreter (Triton 3.1: NumPy 1.26)
```

Checked: the reference against the definition; the kernels' output and gradients against the fp64 reference (max relative error 2e-3 in fp16, 1e-2 for the gradient of $\beta$; 2e-4 in fp32); determinism; the layer.

## Benchmark

```bash
python bench.py --shape 4,12,2048,64 --dtype fp16
```

Reference (Quadro RTX 6000 vGPU, sm_75, PyTorch 2.5.1, Triton 3.1.0): forward 0.80 ms, forward+backward 4.38 ms.
