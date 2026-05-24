#!/usr/bin/env python3
"""
Quantize Gemma (or other causal LM) with AutoAWQ (W4A16) for vLLM `--quantization awq`.

Requires HF token for gated models. Calibration uses a small default text sample;
override with --calib-path for better quality.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    args = ap.parse_args()

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

    if args.calib_path:
        data = Path(args.calib_path).read_text(encoding="utf-8", errors="ignore")
        samples = [data[i : i + 512] for i in range(0, min(len(data), 512 * 128), 512)]
        if not samples:
            samples = ["Calibration fallback text for AWQ. " * 32]
    else:
        samples = [
            "Large language model inference uses KV cache and continuous batching. " * 24
        ]

    model.quantize(tokenizer, quant_config=quant_config, calib_data=samples)
    model.save_quantized(str(out_dir), safetensors=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved AWQ model to {out_dir}")
    print("Serve with vLLM, e.g.:")
    print(
        f'  vllm serve "{out_dir}" --quantization awq --trust-remote-code --max-model-len 8192'
    )


if __name__ == "__main__":
    main()
