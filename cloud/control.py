"""Task-owned DLC controller. Never writes credentials or touches other jobs."""
from pathlib import Path
import argparse, contextlib, datetime, fcntl, hashlib, json, os, shlex, sys, time, uuid
ROOT=Path('/mnt/workspace/umi/tactile_encoder')
CLOUD=ROOT/'cloud'
sys.path.insert(0,str(CLOUD/'deps'))
from alibabacloud_credentials.client import Client as Credentials
from alibabacloud_credentials.models import Config as CredentialConfig
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_pai_dlc20201203 import models
TERMINAL={'Succeeded','Failed','Stopped'}
PREFIX='wfb_tac_20261112_'
IMAGE='daimon-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/daimon/bin_image:bin'
PYTHON='/root/miniconda3/envs/lerobot051/bin/python'


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        t.write_text(json.dumps(data,ensure_ascii=False,indent=2));os.replace(t,path)
    finally:
        if t.exists():t.unlink()


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def client():
    raw=json.loads(Path('/root/.aliyun/config.json').read_text())
    p=next(x for x in raw['profiles'] if x['name']==raw['current'])
    if p['mode']!='CredentialsURI':raise RuntimeError('Unexpected credential provider')
    cred=Credentials(CredentialConfig(type='credentials_uri',credentials_uri=p['credentials_uri']))
    return Client(Config(credential=cred,region_id='cn-wulanchabu',endpoint='pai-dlc.cn-wulanchabu.aliyuncs.com',read_timeout=30000,connect_timeout=10000))


def owned():
    p=CLOUD/'owned_jobs.json';return json.loads(p.read_text()) if p.exists() else []


def get_status(c,entry,logs=False):
    d=c.get_job(entry['JobId'],models.GetJobRequest(need_detail=True)).body.to_map()
    row={k:d.get(k) for k in ['JobId','DisplayName','Status','ReasonCode','ReasonMessage','GmtCreateTime','GmtStartTime','GmtFinishTime']}
    row.update(logical=entry['logical'],gpu=entry['gpu'])
    row['pods']=[{k:p.get(k) for k in ['PodId','Status','SubStatus','GmtStartTime','GmtFinishTime']} for p in d.get('Pods',[])]
    if logs:
        out=[]
        for p in d.get('Pods',[]):
            try: out.extend(c.get_pod_logs(entry['JobId'],p['PodId'],models.GetPodLogsRequest(max_lines=100)).body.to_map().get('Logs',[]))
            except Exception as exc:out.append('log_fetch_error='+type(exc).__name__)
        row['logs']=out
    save(CLOUD/'jobs'/entry.get('folder',entry['logical'])/'status.json',row)
    return row


def _reconcile_unlocked(c):
    entries=owned()
    for path in sorted((CLOUD/'jobs').glob('*/pending.json')):
        pending=json.loads(path.read_text());matches=c.list_jobs(models.ListJobsRequest(workspace_id='218916',display_name=pending['DisplayName'],page_size=100,page_number=1)).body.to_map().get('Jobs',[])
        matches=[x for x in matches if x.get('DisplayName')==pending['DisplayName']]
        if len(matches)!=1:raise RuntimeError('Unresolved submission: '+pending['logical']+'; refuse blind retry')
        jid=matches[0]['JobId']
        if not any(x['JobId']==jid for x in entries):
            entry={**pending,'JobId':jid};entries.append(entry);save(CLOUD/'owned_jobs.json',entries)
        save(path.parent/'receipt.json',{'JobId':jid,'reconciled_at':now()})
        path.rename(path.parent/'resolved.json')
    return entries


def reconcile(c):
    with (CLOUD/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _reconcile_unlocked(c)


def statuses(c,logs=False):
    rows=[get_status(c,e,logs) for e in reconcile(c)]
    save(CLOUD/'status.json',{'updated_at':now(),'jobs':rows,'active_jobs':sum(x['Status'] not in TERMINAL for x in rows),'active_gpus':sum(x['gpu'] for x in rows if x['Status'] not in TERMINAL)})
    return rows


def request(logical,command,gpu=1,minutes=720):
    if not logical.replace('_','').isalnum():raise ValueError('Bad logical name')
    assert gpu in [0,1]
    assert 1<=minutes<=1440
    return {'WorkspaceId':'218916','ResourceId':'quota1boxu6idx3j','DisplayName':PREFIX+logical,'JobType':'PyTorchJob','Priority':1,'Accessibility':'PRIVATE','DryRun':False,'JobMaxRunningTimeMinutes':minutes,
      'JobSpecs':[{'Type':'Worker','PodCount':1,'Image':IMAGE,'ResourceConfig':{'CPU':'8' if gpu else '2','GPU':str(gpu),'Memory':'64Gi' if gpu else '8Gi','SharedMemory':'8Gi' if gpu else '1Gi'},'RestartPolicy':'Never'}],
      'DataSources':[{'DataSourceId':'d-iu3dunhkuejrz0w4jo','DataSourceVersion':'v1','MountPath':'/mnt/workspace','MountAccess':'RW'}],
      'UserVpc':{'DefaultRoute':'eth0','ExtendedCIDRs':['10.11.65.0/24','10.11.64.0/24'],'SecurityGroupId':'sg-0jl5nh21jy9r5uts3pk6','SwitchId':'vsw-0jlkzeskp0wfxqbgged89','VpcId':'vpc-0jlibv5lchq2ljpnl6p4i'},
      'Settings':{'Driver':'580.95.05-open','EnableRDMA':False,'AllocateAllRDMADevices':False,'Shell':'/bin/bash','TerminationGracePeriodSeconds':30},
      'UserCommand':command}


def submit(c,logical,command,gpu=1,minutes=720):
    with (CLOUD/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        entries=_reconcile_unlocked(c)
        previous=next((e for e in reversed(entries) if e['logical']==logical),None)
        if previous:return {'already_submitted':True,'JobId':previous['JobId']}
        rows=[get_status(c,e) for e in entries]
        active=[r for r in rows if r['Status'] not in TERMINAL]
        if len(active)>=2 or sum(r['gpu'] for r in active)+gpu>2:raise RuntimeError('Task concurrency limit: max two jobs/two GPUs')
        name=PREFIX+logical
        existing=c.list_jobs(models.ListJobsRequest(workspace_id='218916',display_name=name,page_size=100,page_number=1)).body.to_map().get('Jobs',[])
        if any(x.get('DisplayName')==name for x in existing):raise RuntimeError('Cloud job already exists without ledger: reconcile explicitly')
        req=request(logical,command,gpu,minutes)
        folder=CLOUD/'jobs'/logical;folder.mkdir(parents=True,exist_ok=True)
        save(folder/'request.json',req)
        pending={'logical':logical,'DisplayName':name,'submitted_at':now(),'gpu':gpu,'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()}
        save(folder/'pending.json',pending)
        result=c.create_job_with_options(models.CreateJobRequest().from_map(req),{},RuntimeOptions(autoretry=False,max_attempts=1)).body.to_map()
        entry={**pending,'JobId':result['JobId']};entries.append(entry);save(CLOUD/'owned_jobs.json',entries)
        save(folder/'receipt.json',{'JobId':entry['JobId'],'RequestId':result.get('RequestId')});(folder/'pending.json').rename(folder/'resolved.json')
        return entry


def stop(c,logical):
    e=next(e for e in reversed(owned()) if e['logical']==logical)
    row=get_status(c,e)
    if row['Status'] not in TERMINAL:c.stop_job(e['JobId'])
    save(CLOUD/'jobs'/e.get('folder',logical)/'stop_requested.json',{'JobId':e['JobId'],'time':now()})
    return {'JobId':e['JobId'],'stop_requested':True}


def environment(logical):
    p=ROOT/'runs'/logical
    return '\n'.join(['set -euo pipefail','umask 022','cd '+shlex.quote(str(ROOT)),
      'export PYTHONDONTWRITEBYTECODE=1','export OMP_NUM_THREADS=4','export OPENBLAS_NUM_THREADS=4','export MKL_NUM_THREADS=4',
      'export MPLCONFIGDIR='+shlex.quote(str(p/'cache/matplotlib')),
      'export HF_HOME='+shlex.quote(str(p/'cache/hf')),
      'export XDG_CACHE_HOME='+shlex.quote(str(p/'cache')),
      'export TMPDIR='+shlex.quote('/tmp/wfb_tactile_'+logical),
      'mkdir -p "$TMPDIR" "$MPLCONFIGDIR"'])+'\n'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['status','submit','stop']);ap.add_argument('--logical');ap.add_argument('--command-file');ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--minutes',type=int,default=720);ap.add_argument('--logs',action='store_true');a=ap.parse_args()
    try:
        c=client()
        if a.action=='status':out=statuses(c,a.logs)
        elif a.action=='stop':out=stop(c,a.logical)
        else:
            path=Path(a.command_file).resolve();path.relative_to(ROOT)
            out=submit(c,a.logical,path.read_text(),a.gpu,a.minutes)
        print(json.dumps(out,ensure_ascii=False),flush=True)
    except Exception as exc:
        print(json.dumps({'error_type':type(exc).__name__,'code':getattr(exc,'code',None),'status_code':getattr(exc,'status_code',None)}),flush=True)
        raise SystemExit(1)
