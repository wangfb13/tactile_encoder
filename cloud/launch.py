"""Resume only the task-owned two-job controller; no blind duplicate launcher."""
from pathlib import Path
import json,subprocess,sys,datetime,fcntl
ROOT=Path('/mnt/workspace/umi/tactile_encoder');C=ROOT/'cloud'
with (C/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 state=json.loads((C/'pipeline_state.json').read_text()) if (C/'pipeline_state.json').exists() else {}
 if state.get('phase')=='complete':
  print(json.dumps({'already_complete':True}));raise SystemExit(0)
 if (C/'daemon.json').exists():
  d=json.loads((C/'daemon.json').read_text());proc=Path('/proc')/str(d['pid'])/'cmdline'
  if proc.exists() and str(C/'pipeline.py').encode() in proc.read_bytes():
   print(json.dumps({'already_running':True,'pid':d['pid']}));raise SystemExit(0)
 with (C/'pipeline.log').open('a') as log:
  child=subprocess.Popen([sys.executable,str(C/'pipeline.py'),'run','--interval','60'],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 receipt={'pid':child.pid,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'executable':sys.executable}
 (C/'daemon.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
