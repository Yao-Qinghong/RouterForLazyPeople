#!/bin/bash
# One-shot test: start Nemotron Nano NVFP4 via TRT-LLM Docker and wait for it to become healthy.
# Run: bash test-nano.sh
# Ctrl-C to stop once you see "Application startup complete" or an error.
docker rm -f test-nemotron-nano 2>/dev/null || true
docker run --rm --name test-nemotron-nano \
  --ipc host --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8109:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/tensorrt_llm:/root/.cache/tensorrt_llm \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc7 \
  trtllm-serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --host 0.0.0.0 --port 8000 \
  --max_seq_len 65536 --max_num_tokens 16384 --max_batch_size 4
