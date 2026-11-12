set -euo pipefail
umask 022
cd /mnt/workspace/umi/tactile_encoder
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/mnt/workspace/umi/tactile_encoder/runs/smoke_mae/cache/matplotlib
export HF_HOME=/mnt/workspace/umi/tactile_encoder/runs/smoke_mae/cache/hf
export XDG_CACHE_HOME=/mnt/workspace/umi/tactile_encoder/runs/smoke_mae/cache
export TMPDIR=/mnt/workspace/umi/tactile_encoder/runs/smoke_mae/tmp
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/root/miniconda3/envs/lerobot051/bin/python /mnt/workspace/umi/tactile_encoder/releases/smoke_v1/train.py --data-root /mnt/workspace/umi/tactile_encoder/data/gpu_smoke --out /mnt/workspace/umi/tactile_encoder/runs/smoke_mae --pretrained /mnt/workspace/umi/tactile_encoder/weights/open_clip_pytorch_model.bin --lambda-force 0.0 --epochs 100 --seed 0 --batch-size 64 --accum-iter 4 --num-workers 4 --cpu-threads 4 --device cuda --resume auto --save-freq 5 --eval-freq 5 --max-steps 20 --log-freq 5
/root/miniconda3/envs/lerobot051/bin/python -c 'import json,pathlib; p=pathlib.Path('"'"'/mnt/workspace/umi/tactile_encoder/runs/smoke_mae/JOB_COMPLETE.json'"'"'); p.write_text('"'"'{"logical": "smoke_mae", "phase": "smoke", "lambda_force": 0.0, "seed": 0, "epochs": 100, "max_steps": 20}'"'"'); print('"'"'TACTILE_JOB_COMPLETE '"'"'+p.read_text(),flush=True)'
