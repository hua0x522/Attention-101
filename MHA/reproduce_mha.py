#!/usr/bin/env python3
"""Reproduce multi-head attention with FlashInfer and plain PyTorch.

This script focuses on the attention core:

    output = softmax(Q K^T / sqrt(head_dim)) V

Q, K, and V are random tensors with shape [seq_len, num_heads, head_dim].
FlashInfer is used as the ground truth, and the PyTorch implementation is
written in a deliberately direct style for learning.
"""

import argparse
import math

import flashinfer
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FlashInfer MHA attention with a teaching PyTorch implementation."
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--head-dim", type=int, default=64, help="Per-head hidden dimension.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Tensor dtype used for Q/K/V.",
    )
    parser.add_argument("--causal", action="store_true", help="Use causal attention masking.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_fields = {
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
    }
    for name, value in positive_fields.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}.")


def dtype_from_name(dtype_name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def tolerances_for_dtype(dtype_name: str) -> tuple[float, float]:
    if dtype_name == "float32":
        return 1e-4, 1e-4
    if dtype_name == "bfloat16":
        return 2e-2, 2e-2
    return 1e-2, 1e-2


def make_random_qkv(args: argparse.Namespace, dtype, device: str):
    torch.manual_seed(args.seed)
    shape = (args.seq_len, args.num_heads, args.head_dim)
    q = torch.randn(shape, dtype=dtype, device=device)
    k = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device)
    return q, k, v


def flashinfer_attention(q, k, v, causal: bool):
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    with torch.inference_mode():
        return flashinfer.prefill.single_prefill_with_kv_cache(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
        )


def pytorch_attention(q, k, v, causal: bool):
    seq_len, _num_heads, head_dim = q.shape

    q_by_head = q.transpose(0, 1)
    k_by_head = k.transpose(0, 1)
    v_by_head = v.transpose(0, 1)

    # scores[h, query_pos, key_pos] stores one attention matrix per head.
    scores = torch.matmul(q_by_head, k_by_head.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)

    if causal:
        causal_mask = torch.ones(
            (seq_len, seq_len), dtype=torch.bool, device=q.device
        ).tril()
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v_by_head)
    return output.transpose(0, 1)


def main() -> int:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required because FlashInfer attention kernels run on GPU, "
            "but torch.cuda.is_available() is False."
        )

    device = "cuda"
    dtype = dtype_from_name(args.dtype)
    q, k, v = make_random_qkv(args, dtype, device)

    flashinfer_out = flashinfer_attention(q, k, v, args.causal)
    pytorch_out = pytorch_attention(q, k, v, args.causal)

    diff = (flashinfer_out.float() - pytorch_out.float()).abs()
    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()

    print("MHA attention core reproduction")
    print(f"  q/k/v shape:       {tuple(q.shape)}")
    print(f"  dtype:             {args.dtype}")
    print(f"  causal:            {args.causal}")
    print(f"  flashinfer output: {tuple(flashinfer_out.shape)}")
    print(f"  pytorch output:    {tuple(pytorch_out.shape)}")
    print(f"  max abs error:     {max_abs_error:.6e}")
    print(f"  mean abs error:    {mean_abs_error:.6e}")

    rtol, atol = tolerances_for_dtype(args.dtype)
    torch.testing.assert_close(
        pytorch_out,
        flashinfer_out,
        rtol=rtol,
        atol=atol,
    )

    print("Correctness check passed: PyTorch teaching version matches FlashInfer.")
    return 0


if __name__ == "__main__":
    main()
