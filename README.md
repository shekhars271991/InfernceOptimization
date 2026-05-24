# InferenceOpt

Scripts and benchmarks for **Gemma 7B** (default: `google/gemma-7b-it`) on **vLLM**: single-node baseline, optional **disaggregated prefill/decode**, **prefix caching** / chunked prefill experiments, **AWQ** quantization, and **lm-evaluation-harness** runs against an OpenAI-compatible server.

## Prerequisites

- Linux GPU host with NVIDIA drivers (and CUDA compatible with your vLLM wheel).
- Python 3.10+ recommended.
- Hugging Face: accept the Gemma license and set `HUGGING_FACE_HUB_TOKEN` or `HF_TOKEN`.

## Where to run (instances)

**Gemma 7B in FP16/BF16** needs on the order of **~14–16 GB** of GPU memory per running server (more with long `MAX_MODEL_LEN` and concurrent KV cache). Pick a GPU with enough headroom.

| Workload | Sensible choices |
|----------|------------------|
| **Single-node baseline + benchmarks** | One **24 GB** class GPU is comfortable, e.g. **AWS `g5.xlarge`** (1× NVIDIA **A10G** 24 GB), **GCP** with **L4** / **A10**, or a **Vast.ai / RunPod** 3090/4090 24 GB box. **16 GB** (e.g. **T4**) can work but is tighter—lower concurrency and `max-model-len` if you OOM. |
| **Disaggregated prefill + decode** | Each side runs a **full** vLLM replica, so aim for **two similar GPUs**, either **two VMs** (e.g. **2× `g5.xlarge`**, one prefill and one decode, set `VLLM_HOST_IP` / URLs) or **one host with two GPUs** (e.g. **AWS `g5.12xlarge`** with 4× A10G—use two of them for prefill/decode). **NVLink is not required** for the default P2P NCCL connector: cross-node traffic usually goes over the VPC (TCP) unless you attach **EFA** and tune NCCL for it (see below). |
| **AWQ 4-bit** | Often fits comfortably on **one 16–24 GB** GPU; good for a second pass on a single cheap instance after disagg experiments. |

**AWS G6 and “RDMA / InfiniBand”:** Public EC2 GPU instances do **not** expose customer classical **InfiniBand** the way many on-prem HPC clusters do. What people usually mean on AWS is **EFA** (Elastic Fabric Adapter), which can give **RDMA read/write** semantics for NCCL/MPI over the **AWS network fabric**—not the same as NVLink or IB on bare metal. Per [AWS EFA supported instance types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html), **EFA with RDMA read/write** is listed for **G6 starting at `g6.8xlarge`** (and similarly large **g6e** sizes)—**not** for the smallest 1×GPU sizes like **`g6.xlarge`**, which typically rely on ordinary **TCP** over the VPC for inter-node NCCL. Even on EFA-enabled G6/G6e, **GPUDirect RDMA** (GPU memory straight to the NIC) is **not** in the same class as **P4d / P5**; NCCL may still **bounce through host memory**, which is usually fine for experiments but not the same as full GPU RDMA.

Use a recent **Linux** image with NVIDIA drivers (or a CUDA-capable deep-learning AMI). Match your **vLLM / PyTorch** install to the driver/CUDA stack on that image.

## Setup

The repo uses a **project virtualenv** at **`.venv/`** so installs work on **Ubuntu 24.04+** (PEP 668 blocks `pip install` into the system Python).

```bash
./scripts/setup_instance.sh
```

This script:

1. Checks GPU / driver (`nvidia-smi`).
2. On **Ubuntu/Debian**, installs **`python3-venv`** and **`python3-full`** via `apt` when needed (sudo).
3. Creates **`.venv`** (if missing) and runs **`pip install -r requirements.txt`** inside it.

Then either:

```bash
source .venv/bin/activate
```

or rely on **`scripts/launch_*.sh`** and **`eval/run_lm_eval.sh`**, which prepend **`.venv/bin`** to `PATH` when `.venv` exists.

**Manual venv (if you skip the script):**

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-full
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

vLLM wheels may still depend on your driver/CUDA stack—see the [vLLM install docs](https://docs.vllm.ai/en/stable/getting_started/installation.html).


**Benchmarks and `launch_proxy.py`:** use the venv interpreter if you are not activating the env, e.g. `.venv/bin/python benchmarks/bench_latency.py` or `source .venv/bin/activate` first.

## Single-node baseline (vLLM)

### Option B — Docker (recommended on minimal Ubuntu)

Uses the official **`vllm/vllm-openai`** image so **FlashInfer / `nvcc`** live inside the container (no host CUDA toolkit fight).

**One-time host setup (Ubuntu):**

```bash
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker "$USER"   # then log out/in, or: newgrp docker
```

Install the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)** so `docker run --gpus all` works, then:

```bash
export HF_MODEL_NAME=google/gemma-7b-it   # optional; default
# Optional: put HF_TOKEN or HUGGING_FACE_HUB_TOKEN in scripts/.env (see scripts/docker-env.template)
./scripts/launch_baseline_docker.sh
```

Foreground run (`-it`); **Ctrl+C** stops the container. For **background**:

```bash
VLLM_DOCKER_DETACH=1 ./scripts/launch_baseline_docker.sh
docker logs -f vllm-baseline
```

Useful env vars: `VLLM_DOCKER_IMAGE` (default `vllm/vllm-openai:latest` — **pin a tag** for reproducibility), `VLLM_DOCKER_SKIP_PULL=1`, `HF_CACHE`, `VLLM_PORT`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `ENABLE_PREFIX_CACHE=1`.

**24 GB GPU + “KV cache … larger than available”:** The Docker script defaults to **`MAX_MODEL_LEN=4096`** and **`GPU_MEMORY_UTILIZATION=0.93`** for a safer fit on **L4 / A10G** class cards. vLLM **0.21+** may also reserve VRAM for CUDA-graph estimates; if you still hit KV limits after raising `MAX_MODEL_LEN`, you can pass e.g. **`-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`** on `docker run` (see vLLM startup logs). For **8192** on one 24 GB GPU you may still need a **smaller model**, **AWQ**, or **more VRAM** — see [vLLM memory tuning](https://docs.vllm.ai/en/latest/configuration/conserving_memory/).

Benchmarks from the host still use `BENCH_BASE_URL=http://127.0.0.1:8000` (or your `VLLM_PORT`) against the container thanks to **`--network host`**.

### Option A — venv on the host

```bash
export HF_MODEL_NAME=google/gemma-7b-it   # optional; this is the default
./scripts/launch_baseline.sh
```

Optional prefix caching (for fair comparison with disagg scripts, which enable it):

```bash
ENABLE_PREFIX_CACHE=1 ./scripts/launch_baseline.sh
```

Useful environment variables: `VLLM_PORT`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`.

On **24 GB** GPUs with **vLLM 0.21+**, if startup fails with **KV cache larger than available**, lower **`MAX_MODEL_LEN`** (for example `4096`) or raise **`GPU_MEMORY_UTILIZATION`**; see the Docker baseline section for optional **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`** tuning.

On **driver-only** Ubuntu images, host vLLM may require **`nvcc`** (see **`./scripts/setup_instance.sh`** and **`scripts/_inferopt_cuda_env.sh`**) unless you use **Docker** above.

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

## Quantization (AWQ) — first optimization after baseline

Use this after you have a **baseline** FP16/BF16 server and optional **benchmarks + lm-eval** numbers to compare against.

### 1. Free GPU memory and quantize (host venv; not Docker)

**vLLM cannot share the GPU** with a long AWQ calibration—**stop** the baseline container or server first.

```bash
source .venv/bin/activate
export HF_TOKEN=...   # or HUGGING_FACE_HUB_TOKEN; Gemma is gated on Hugging Face
python3 quantization/quantize_awq.py \
  --model-path google/gemma-7b-it \
  --quant-path ./gemma-7b-awq
```

Optional: **`--calib-path /path/to/calibration.txt`** (UTF-8) for better W4 quality; **`--calib-chunks`** / **`--calib-seq-len`** tune how many 512-token windows are built (defaults match AutoAWQ). The script feeds **pre-tokenized** windows so Gemma is not skipped by AutoAWQ’s “line longer than 512 tokens” rule. Output defaults to **`./gemma-7b-awq/`** (gitignored).

### 2. Serve AWQ with vLLM

**Docker (same pattern as baseline):** mounts `./gemma-7b-awq` and uses **`--quantization awq`**. Default port **`8001`** so you can run FP16 on **8000** when you have **two GPUs** (or two hosts).

```bash
./scripts/launch_awq_docker.sh
# Foreground; Ctrl+C stops. Background:
VLLM_DOCKER_DETACH=1 ./scripts/launch_awq_docker.sh
docker logs -f vllm-awq
```

Useful env vars: **`AWQ_QUANT_PATH`** (default repo `./gemma-7b-awq`), **`VLLM_PORT`** (default **8001**), **`MAX_MODEL_LEN`**, **`GPU_MEMORY_UTILIZATION`**, **`AWQ_SERVED_MODEL_NAME`** (default **`google/gemma-7b-it`** so clients keep the same **`BENCH_MODEL`**), same Docker image / detach / HF cache vars as **`launch_baseline_docker.sh`**.

**Host venv** (if you prefer not to use Docker):

```bash
vllm serve ./gemma-7b-awq --quantization awq --trust-remote-code \
  --served-model-name google/gemma-7b-it --host 0.0.0.0 --port 8001 \
  --max-model-len 4096 --enable-chunked-prefill
```

### 3. Re-run benchmarks and/or lm-eval

Point tools at the AWQ port (**`8001`** if you used the AWQ Docker defaults):

```bash
export BENCH_BASE_URL=http://127.0.0.1:8001
export BENCH_MODEL=google/gemma-7b-it
python3 benchmarks/bench_latency.py
python3 benchmarks/bench_throughput.py
export BENCH_COMPLETIONS_URL=http://127.0.0.1:8001/v1/completions
./eval/run_lm_eval.sh
```

Append a short summary to **`results/lm_eval/recorded_runs.txt`** if you want a permanent log next to your FP16 run.

### 4. Side-by-side FP16 vs AWQ (optional)

**`quantization/bench_quant_compare.py`** calls latency + throughput against **both** URLs in one go, so **both servers must be up at once** (e.g. **two GPUs**: FP16 on **8000**, AWQ on **8001**, with `CUDA_VISIBLE_DEVICES` set per process/container as you usually do for multi-GPU).

```bash
export BENCH_FP16_URL=http://127.0.0.1:8000
export BENCH_AWQ_URL=http://127.0.0.1:8001
python3 quantization/bench_quant_compare.py
```

On a **single GPU**, run FP16 benches first and save JSON, then **stop FP16**, serve AWQ on **8000**, run the same bench scripts again, and compare in **`notebooks/results_analysis.ipynb`** (or diff the JSON files under **`results/`**).

## Evaluation (lm-eval)

`requirements.txt` installs **`lm-eval[api]`** (e.g. **tenacity**) for API-backed eval. If you installed lm-eval without extras, run **`pip install 'lm-eval[api]'`**.

`eval/run_lm_eval.sh` uses **`local-completions`** against **`/v1/completions`** so tasks that need **loglikelihood** (MMLU, HellaSwag, TruthfulQA MC, …) work. The chat-only API does not support loglikelihood; see [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) docs.

With vLLM serving (OpenAI-compatible **completions** on the same host/port as your server):

```bash
export BENCH_BASE_URL=http://127.0.0.1:8000
export BENCH_MODEL=google/gemma-7b-it
./eval/run_lm_eval.sh
```

Optional env: `LM_EVAL_TASKS`, **`LM_EVAL_LIMIT`** (defaults to **100** per task; **`export LM_EVAL_LIMIT=`** empty for full runs), **`LM_EVAL_MAX_LENGTH`** (default **4096**, align with vLLM `--max-model-len`), **`BENCH_COMPLETIONS_URL`** if the completions path is not `${BENCH_BASE_URL}/v1/completions`, **`LM_EVAL_NUM_CONCURRENT`**, **`LM_EVAL_BATCH_SIZE`** (default **1** for API loglikelihood), **`LM_EVAL_TOKENIZER_BACKEND`** (default **huggingface**), **`LM_EVAL_TOKENIZED_REQUESTS`**. The script passes **`--apply_chat_template`** for instruct models. **`HF_TOKEN`** / **`HUGGING_FACE_HUB_TOKEN`** are forwarded for tokenizer hub access when set. Append **`--trust_remote_code`** after the script if the tokenizer needs it. Each run writes aggregated JSON to **`results/lm_eval/run-<timestamp>.json`** (under **`results/`**, gitignored) unless **`LM_EVAL_NO_SAVE=1`** or you set **`LM_EVAL_OUTPUT_PATH`**. Optional **`LM_EVAL_OUT_DIR`** changes the default directory. Committed score snapshots: append in **`results/lm_eval/recorded_runs.txt`**.

If vLLM returns errors about **logprobs** limits on multiple-choice tasks, increase the server’s **`--max-logprobs`** (see [vLLM OpenAI server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) options).

## Notebook

Open `notebooks/results_analysis.ipynb` to load JSON files from `results/` and plot throughput / FP16 vs AWQ summaries.

## Repository layout

```
InferenceOpt/
├── README.md
├── requirements.txt
├── scripts/            # launch_baseline*.sh, launch_awq_docker.sh, …
├── benchmarks/
├── quantization/
├── eval/               # run_lm_eval.sh
├── notebooks/
└── results/            # benchmarks + lm_eval JSON (gitignored); lm_eval/recorded_runs.txt tracked
```

## License

Inference scripts in this repo are for local experimentation. **vLLM** disaggregated proxy logic in `scripts/launch_proxy.py` is derived from the vLLM project (Apache-2.0); see file header. Respect **Gemma** and **vLLM** upstream licenses and terms of use.
