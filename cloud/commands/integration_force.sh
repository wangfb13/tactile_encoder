set -euo pipefail
umask 022
cd /mnt/workspace/umi/tactile_encoder
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/mnt/workspace/umi/tactile_encoder/runs/integration_force/cache/matplotlib
export HF_HOME=/mnt/workspace/umi/tactile_encoder/runs/integration_force/cache/hf
export XDG_CACHE_HOME=/mnt/workspace/umi/tactile_encoder/runs/integration_force/cache
export TMPDIR=/tmp/wfb_tactile_integration_force
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/root/miniconda3/envs/lerobot051/bin/python /mnt/workspace/umi/tactile_encoder/releases/v1/train.py --data-root /mnt/workspace/umi/tactile_encoder/data/gpu_smoke --out /mnt/workspace/umi/tactile_encoder/runs/integration_force --pretrained /mnt/workspace/umi/tactile_encoder/weights/open_clip_pytorch_model.bin --lambda-force 0.3 --epochs 5 --seed 0 --batch-size 256 --accum-iter 1 --num-workers 4 --cpu-threads 4 --device cuda --resume auto --save-freq 5 --eval-freq 5
/root/miniconda3/envs/lerobot051/bin/python /mnt/workspace/umi/tactile_encoder/releases/v1/evaluate.py --checkpoint /mnt/workspace/umi/tactile_encoder/runs/integration_force/encoder.pt --data-root /mnt/workspace/umi/tactile_encoder/data/gpu_smoke --out /mnt/workspace/umi/tactile_encoder/runs/integration_force/evaluation --device cuda --batch-size 128 --workers 4 --bootstrap 100 --selection-only
/root/miniconda3/envs/lerobot051/bin/python -c 'import json,pathlib; p=pathlib.Path('"'"'/mnt/workspace/umi/tactile_encoder/runs/integration_force/JOB_COMPLETE.json'"'"'); p.write_text('"'"'{"logical": "integration_force", "phase": "screen", "lambda_force": 0.3, "seed": 0, "epochs": 5, "max_steps": 0}'"'"'); print('"'"'TACTILE_JOB_COMPLETE '"'"'+p.read_text(),flush=True)'
