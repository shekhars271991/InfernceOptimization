#!/usr/bin/env python3
"""
Quantize Gemma (or other causal LM) with AutoAWQ (W4A16) for vLLM `--quantization awq`.

Requires HF token for gated models. Calibration uses token windows (not one huge string):
AutoAWQ skips any text line whose encode length exceeds 512 tokens, which used to leave
zero samples for Gemma. Override with --calib-path for better quality.

VRAM: AutoAWQ builds a batch of size ``max_calib_samples`` (see their ``get_calib_dataset``).
The upstream default (128) is often too large for a single ~24GB GPU on 7B; this script defaults
to smaller values and serializes layer forwards with ``n_parallel_calib_samples=1``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _pad_token_id(tokenizer):
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    return 0


def _chunks_token_ids(
    tokenizer, text: str, *, max_seq_len: int, n_chunks: int
) -> list[list[int]]:
    """
    AutoAWQ's get_calib_dataset() skips any string whose encode length is > max_seq_len,
    and returns [] if no valid chunks exist — so a single long default paragraph fails Gemma.
    Passing list[list[int]] bypasses that filter and supplies fixed-length windows.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < max_seq_len:
        pad = _pad_token_id(tokenizer)
        ids = (ids + [pad] * max_seq_len)[:max_seq_len]
    out: list[list[int]] = []
    for i in range(0, min(len(ids), max_seq_len * n_chunks), max_seq_len):
        chunk = ids[i : i + max_seq_len]
        if len(chunk) < max_seq_len:
            pad = _pad_token_id(tokenizer)
            chunk = chunk + [pad] * (max_seq_len - len(chunk))
        out.append(chunk)
        if len(out) >= n_chunks:
            break
    if not out:
        raise SystemExit(
            "Calibration chunking produced no samples; try --calib-path with more UTF-8 text."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default=os.environ.get("HF_MODEL_NAME", "google/gemma-7b-it"),
        help="Hugging Face model id or local directory",
    )
    ap.add_argument(
        "--quant-path",
        default=str(Path(__file__).resolve().parent.parent / "gemma-7b-awq"),
        help="Output directory for quantized weights",
    )
    ap.add_argument("--w-bit", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=128)
    ap.add_argument("--zero-point", action="store_true", default=True)
    ap.add_argument(
        "--calib-path",
        default="",
        help="Optional path to a UTF-8 text file used as calibration data",
    )
    ap.add_argument(
        "--calib-chunks",
        type=int,
        default=128,
        help="Number of max-seq-len token windows for calibration (default 128)",
    )
    ap.add_argument(
        "--calib-seq-len",
        type=int,
        default=512,
        help="Token length per calibration window (passed as max_calib_seq_len; default 512)",
    )
    ap.add_argument(
        "--max-calib-samples",
        type=int,
        default=16,
        help="Max calibration sequences after AutoAWQ split (batch size during init); lower for less VRAM (default 16)",
    )
    ap.add_argument(
        "--n-parallel-calib-samples",
        type=int,
        default=1,
        help="How many calib sequences run on GPU per layer forward; 1 minimizes VRAM (default 1). None in AutoAWQ means all at once.",
    )
    ap.add_argument(
        "--max-chunk-memory-mb",
        type=int,
        default=0,
        help="Cap for AutoAWQ max_chunk_memory in MiB (0 = use library default ~1024 MiB)",
    )
    args = ap.parse_args()
    if args.max_calib_samples < 1:
        raise SystemExit("--max-calib-samples must be >= 1")
    if args.n_parallel_calib_samples < 1:
        raise SystemExit("--n-parallel-calib-samples must be >= 1")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "autoawq / transformers not installed. Run: pip install -r requirements.txt"
        ) from e

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }

    out_dir = Path(args.quant_path)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        safetensors=True,
        device_map="auto",
    )

    max_len = args.calib_seq_len
    n_chunks = args.calib_chunks

    if args.calib_path:
        data = Path(args.calib_path).read_text(encoding="utf-8", errors="ignore")
        if not data.strip():
            raise SystemExit(f"Calibration file is empty: {args.calib_path}")
        ids = tokenizer.encode(data, add_special_tokens=False)
        samples: list[list[int]] = []
        for i in range(0, len(ids), max_len):
            chunk = ids[i : i + max_len]
            if len(chunk) < max_len:
                pad = _pad_token_id(tokenizer)
                chunk = chunk + [pad] * (max_len - len(chunk))
            samples.append(chunk)
            if len(samples) >= n_chunks:
                break
        if not samples:
            raise SystemExit("Calibration text tokenized to nothing; add more text.")
    else:
        seed = (
            "Large language model inference uses KV cache and continuous batching. "
            "Quantization maps high-precision weights to fewer bits while preserving behavior. "
            * 400
        )
        samples = _chunks_token_ids(
            tokenizer, seed, max_seq_len=max_len, n_chunks=n_chunks
        )

    quant_kwargs = dict(
        quant_config=quant_config,
        calib_data=samples,
        max_calib_samples=args.max_calib_samples,
        max_calib_seq_len=args.calib_seq_len,
        n_parallel_calib_samples=args.n_parallel_calib_samples,
    )
    if args.max_chunk_memory_mb > 0:
        quant_kwargs["max_chunk_memory"] = args.max_chunk_memory_mb * 1024 * 1024

    model.quantize(tokenizer, **quant_kwargs)
    model.save_quantized(str(out_dir), safetensors=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved AWQ model to {out_dir}")
    print("Serve with vLLM, e.g.:")
    print(
        f'  vllm serve "{out_dir}" --quantization awq --trust-remote-code --max-model-len 4096'
    )


if __name__ == "__main__":
    main()
