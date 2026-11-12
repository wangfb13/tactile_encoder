set -euo pipefail
umask 022
cd /mnt/workspace/umi/tactile_encoder
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR=/mnt/workspace/umi/tactile_encoder/runs/mountcheck/cache/matplotlib
export HF_HOME=/mnt/workspace/umi/tactile_encoder/runs/mountcheck/cache/hf
export XDG_CACHE_HOME=/mnt/workspace/umi/tactile_encoder/runs/mountcheck/cache
export TMPDIR=/mnt/workspace/umi/tactile_encoder/runs/mountcheck/tmp
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
/root/miniconda3/envs/lerobot051/bin/python -c 'import pathlib,json,importlib,sys
root=pathlib.Path('"'"'/mnt/workspace/umi/tactile_encoder'"'"')
assert (root/'"'"'provenance/source.json'"'"').is_file()
out={'"'"'python'"'"':sys.version,'"'"'mounted_project'"'"':str(root),'"'"'modules'"'"':{}}
for name in ['"'"'torch'"'"','"'"'numpy'"'"','"'"'PIL'"'"','"'"'matplotlib'"'"','"'"'av'"'"','"'"'cv2'"'"']:
 try:
  m=importlib.import_module(name);out['"'"'modules'"'"'][name]=getattr(m,'"'"'__version__'"'"','"'"'present'"'"')
 except ImportError:out['"'"'modules'"'"'][name]=None
(root/'"'"'cloud/mount_probe_result.json'"'"').write_text(json.dumps(out))
print('"'"'TACTILE_MOUNT_PROBE '"'"'+json.dumps(out),flush=True)
'
