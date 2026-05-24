# InferenceOpt

Scripts and benchmarks for **Gemma 7B** (default: `google/gemma-7b-it`) on **vLLM**: single-node baseline, optional **disaggregated prefill/decode**, **prefix caching** / chunked prefill experiments, **AWQ** quantization, and **lm-evaluation-harness** runs against an OpenAI-compatible server.

## Prerequisites

- Linux GPU host with NVIDIA drivers (and CUDA compatible with your vLLM wheel).
- Python 3.10+ recommended.
- Hugging Face: accept the Gemma license and set `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN`.

## Setup

```bash
./scripts/setup_instance.sh
```

Or manually: `pip install -r requirements.txt` (vLLM may need the [official install](https://docs.vllm.ai/en/stable/getting_started/installation.html) for your CUDA version).

## Single-node baseline (vLLM)

```bash
export HF_MODEL_NAME=google/gemma-7b-it   # optional; this is the default
./scripts/launch_baseline.sh
```

Optional prefix caching (for fair comparison with disagg scripts, which enable it):

```bash
ENABLE_PREFIX_CACHE=1 ./scripts/launch_baseline.sh
```

Useful environment variables: `VLLM_PORT`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`.

## Disaggregated prefill + decode (experimental)

This matches the upstream vLLM pattern: **kv_producer** on port **8100** (KV **14579**), **kv_consumer** on **8200** (KV **14580**), and a small **Quart** proxy that does prefill with `max_tokens=1` then streams decode.

1. Set `VLLM_HOST_IP` to a reachable IP for multi-host or keep `127.0.0.1` on one machine with two GPUs.
2. Start prefill and decode (defaults: GPU 0 and GPU 1 on the same host):

   ```bash
   ./scripts/launch_prefill.sh
   ./scripts/launch_decode.sh
   ```

3. Start the proxy (client hits this URL; default port **8000**):

   ```bash
   python3 scripts/launch_proxy.py --prefill-url http://127.0.0.1:8100 --decode-url http://127.0.0.1:8200
   ```

   Override with `DISAGG_PROXY_PORT`, `PREFILL_URL`, `DECODE_URL`, `KV_HOST`, `PREFILL_KV_PORT`, `DECODE_KV_PORT` as needed.

The proxy exposes `/v1/completions` and `/v1/chat/completions`.

## Benchmarks

Point clients at the server (baseline or proxy). Outputs go under `results/` (created automatically).

| Script | Purpose |
|--------|---------|
| `benchmarks/bench_latency.py` | TTFT and ITL percentiles (streaming chat) |
| `benchmarks/bench_throughput.py` | Concurrency sweep → tokens/s |
| `benchmarks/bench_prefix_cache.py` | Long shared system prefix → TTFT speedup heuristic |

Common environment variables: `BENCH_BASE_URL` (default `http://127.0.0.1:8000`), `BENCH_MODEL`, `OPENAI_API_KEY` (if your gateway requires it).

Example:

```bash
export BENCH_BASE_URL=http://127.0.0.1:8000
python3 benchmarks/bench_latency.py
python3 benchmarks/bench_throughput.py
```

## Quantization (AWQ)

```bash
python3 quantization/quantize_awq.py --model-path google/gemma-7b-it --quant-path ./gemma-7b-awq
```

Serve the output directory with vLLM using `--quantization awq`.

Compare FP16 vs AWQ servers (two URLs):

```bash
export BENCH_FP16_URL=http://127.0.0.1:8000
export BENCH_AWQ_URL=http://127.0.0.1:8001
python3 quantization/bench_quant_compare.py
```

## Evaluation (lm-eval)

With vLLM serving chat completions:

```bash
export BENCH_BASE_URL=http://127.0.0.1:8000
export BENCH_MODEL=google/gemma-7b-it
./eval/run_lm_eval.sh
```

Optional: `LM_EVAL_TASKS`, `LM_EVAL_LIMIT` (smoke tests), `LM_EVAL_NUM_CONCURRENT`, `LM_EVAL_BATCH_SIZE`.

Some tasks (notably certain **MMLU** setups) expect loglikelihood via a **completions** API; if a task fails on chat, trim `LM_EVAL_TASKS` or use a completions-based workflow (see [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) docs).

## Notebook

Open `notebooks/results_analysis.ipynb` to load JSON files from `results/` and plot throughput / FP16 vs AWQ summaries.

## Repository layout

```
InferenceOpt/
├── README.md
├── requirements.txt
├── scripts/
├── benchmarks/
├── quantization/
├── eval/
├── notebooks/
└── results/          # created when you run benchmarks
```

## License

Inference scripts in this repo are for local experimentation. **vLLM** disaggregated proxy logic in `scripts/launch_proxy.py` is derived from the vLLM project (Apache-2.0); see file header. Respect **Gemma** and **vLLM** upstream licenses and terms of use.
