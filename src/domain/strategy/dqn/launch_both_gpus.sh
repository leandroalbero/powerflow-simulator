#!/bin/bash
cd ~/powerflow-simulator

# Kill any existing training
pkill -f train_gpu 2>/dev/null
pkill -f train_vectorized 2>/dev/null
sleep 1

# GPU 0 (RTX 3090, 24GB) — vectorized with 256-512 parallel envs
nohup python3 -u -m src.domain.strategy.dqn.train_vectorized --device cuda:0 > /tmp/dqn_gpu0.log 2>&1 &
echo "GPU0 PID=$!"

# GPU 1 (RTX 2060, 6GB) — vectorized with 128-256 parallel envs
nohup python3 -u -m src.domain.strategy.dqn.train_vectorized --device cuda:1 > /tmp/dqn_gpu1.log 2>&1 &
echo "GPU1 PID=$!"

echo "Both GPUs launched with vectorized training."
echo "Monitor: tail -f /tmp/dqn_gpu0.log /tmp/dqn_gpu1.log"
