set -euo pipefail
umask 022
cd /mnt/workspace/umi/tactile_encoder
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/cache/matplotlib
export HF_HOME=/mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/cache/hf
export XDG_CACHE_HOME=/mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/cache
export TMPDIR=/tmp/wfb_tactile_final_mae_s0
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/root/miniconda3/envs/lerobot051/bin/python /mnt/workspace/umi/tactile_encoder/releases/v1/train.py --data-root /mnt/workspace/umi/tactile_encoder/data/prepared --out /mnt/workspace/umi/tactile_encoder/runs/final_mae_s0 --pretrained /mnt/workspace/umi/tactile_encoder/weights/open_clip_pytorch_model.bin --lambda-force 0.0 --epochs 100 --seed 0 --batch-size 64 --accum-iter 4 --num-workers 4 --cpu-threads 4 --device cuda --resume auto --save-freq 5 --eval-freq 5
/root/miniconda3/envs/lerobot051/bin/python /mnt/workspace/umi/tactile_encoder/releases/v1/evaluate.py --checkpoint /mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/encoder.pt --data-root /mnt/workspace/umi/tactile_encoder/data/prepared --out /mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/evaluation --device cuda --batch-size 128 --workers 4 --bootstrap 300 --final
/root/miniconda3/envs/lerobot051/bin/python /mnt/workspace/umi/tactile_encoder/releases/visualization_v1/test_vis.py --checkpoint /mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/encoder.pt --data-root /mnt/workspace/umi/tactile_encoder/data/prepared --out /mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/visualization --split val --samples 4 --device cuda
/root/miniconda3/envs/lerobot051/bin/python -c 'import json,pathlib; p=pathlib.Path('"'"'/mnt/workspace/umi/tactile_encoder/runs/final_mae_s0/JOB_COMPLETE.json'"'"'); p.write_text('"'"'{"logical": "final_mae_s0", "phase": "final", "lambda_force": 0.0, "seed": 0, "epochs": 100, "max_steps": 0}'"'"'); print('"'"'TACTILE_JOB_COMPLETE '"'"'+p.read_text(),flush=True)'
