"""Bounded two-job experiment pipeline; selection precedes untouched final tests."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shlex,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import control as cloud
ROOT=cloud.ROOT
CLOUD=cloud.CLOUD
RELEASE=ROOT/'releases/v1'
DATA=ROOT/'data/prepared'
PYTHON=cloud.PYTHON
TERMINAL=cloud.TERMINAL


def load(path):return json.loads(Path(path).read_text())


def verify_release():
    stamp=load(RELEASE/'VERIFIED.json')
    for name,value in stamp['files'].items():
        if hashlib.sha256((RELEASE/name).read_bytes()).hexdigest()!=value:
            raise RuntimeError('Frozen release changed: '+name)
    return stamp


def training_command(logical,weight,epochs,seed,phase,data_root=None,max_steps=0):
    out=ROOT/'runs'/logical
    source=RELEASE
    data_root=Path(data_root or DATA)
    args=[PYTHON,str(source/'train.py'),'--data-root',str(data_root),'--out',str(out),
          '--pretrained',str(ROOT/'weights/open_clip_pytorch_model.bin'),
          '--lambda-force',str(weight),'--epochs',str(epochs),'--seed',str(seed),
          '--batch-size','64','--accum-iter','4','--num-workers','4','--cpu-threads','4',
          '--device','cuda','--resume','auto','--save-freq','5','--eval-freq','5']
    if max_steps:args+=['--max-steps',str(max_steps),'--log-freq','5']
    cmd=cloud.environment(logical)
    cmd+='nvidia-smi --query-gpu=name,memory.total --format=csv,noheader\n'
    cmd+=shlex.join(args)+'\n'
    if not max_steps:
        evaluation=[PYTHON,str(source/'evaluate.py'),'--checkpoint',str(out/'encoder.pt'),
                    '--data-root',str(data_root),'--out',str(out/'evaluation'),
                    '--device','cuda','--batch-size','128','--workers','4',
                    '--bootstrap','100' if phase=='screen' else '300',
                    '--selection-only' if phase=='screen' else '--final']
        cmd+=shlex.join(evaluation)+'\n'
    if phase=='final' and not max_steps:
        vis=load(ROOT/'configs/visualization.json');vroot=(ROOT/vis['path']).resolve();vroot.relative_to(ROOT)
        for name,digest in vis['files'].items():
            if hashlib.sha256((vroot/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen visualization code changed')
        cmd+=shlex.join([PYTHON,str(vroot/'test_vis.py'),'--checkpoint',str(out/'encoder.pt'),
                        '--data-root',str(data_root),'--out',str(out/'visualization'),
                        '--split','val','--samples','4','--device','cuda'])+'\n'
    completion={'logical':logical,'phase':phase,'lambda_force':weight,'seed':seed,'epochs':epochs,'max_steps':max_steps}
    footer="import json,pathlib; p=pathlib.Path("+repr(str(out/'JOB_COMPLETE.json'))+"); p.write_text("+repr(json.dumps(completion))+"); print('TACTILE_JOB_COMPLETE '+p.read_text(),flush=True)"
    cmd+=PYTHON+' -c '+shlex.quote(footer)+'\n'
    path=CLOUD/'commands'/(logical+'.sh');path.parent.mkdir(exist_ok=True);path.write_text(cmd)
    return cmd


def call_local(args):
    result=subprocess.run([sys.executable]+[str(x) for x in args],cwd=str(ROOT),capture_output=True,text=True)
    if result.returncode:
        (CLOUD/'pipeline_subprocess_error.log').write_text(result.stdout+'\n'+result.stderr)
        raise RuntimeError('Local pipeline stage failed; see pipeline_subprocess_error.log')
    return result.stdout


def screened():
    return [('screen_l000_s0',0.),('screen_l010_s0',.1),('screen_l030_s0',.3),('screen_l050_s0',.5)]


def data_fingerprint():
    value=hashlib.sha256()
    for name in ['normalization.json','train_manifest.jsonl']:
        value.update(name.encode());value.update((DATA/name).read_bytes())
    return value.hexdigest()


def require_artifacts(logical,phase,weight,epochs,seed):
    out=ROOT/'runs'/logical
    paths=[out/n for n in ['JOB_COMPLETE.json','summary.json','COMPLETED.json','run_config.json','encoder.pt','evaluation/evaluation.json','evaluation/EVALUATION_COMPLETED.json']]
    if phase=='final':paths.append(out/'visualization/visualization.json')
    if not all(p.is_file() for p in paths):raise RuntimeError('Missing experiment artifacts: '+logical)
    expected={'logical':logical,'phase':phase,'lambda_force':weight,'epochs':epochs,'seed':seed,'data_fingerprint':data_fingerprint()}
    signature={str(p.relative_to(out)):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths}
    cached=out/'ARTIFACT_VERIFIED.json'
    if cached.exists():
        prior=load(cached)
        if prior.get('expected')==expected and prior.get('files')==signature:return out/'evaluation/evaluation.json'
    footer=load(out/'JOB_COMPLETE.json');summary=load(out/'summary.json');completion=load(out/'COMPLETED.json');config=load(out/'run_config.json')
    for key in ['logical','phase','lambda_force','epochs','seed']:
        if footer.get(key)!=expected[key]:raise RuntimeError('Job artifact configuration mismatch: '+logical+'/'+key)
    for row in [summary,completion]:
        if row.get('status')!='completed':raise RuntimeError('Partial training cannot satisfy experiment: '+logical)
        for key in ['lambda_force','epochs','seed','data_fingerprint']:
            if row.get(key)!=expected[key]:raise RuntimeError('Training artifact configuration mismatch: '+logical+'/'+key)
        if Path(row['checkpoint']).resolve()!=(out/'encoder.pt').resolve():raise RuntimeError('Unexpected checkpoint path')
    if config['args']['max_steps']!=0 or config['args']['model_size']!='base' or config['effective_batch_size']!=256:
        raise RuntimeError('Formal run configuration mismatch')
    result=load(out/'evaluation/evaluation.json');mode='selection_only' if phase=='screen' else 'final'
    if result.get('mode')!=mode:raise RuntimeError('Unexpected evaluation mode')
    for key in ['lambda_force','seed']:
        if result.get(key)!=expected[key]:raise RuntimeError('Evaluation identity mismatch: '+logical+'/'+key)
    expected_splits={'train','val'} if phase=='screen' else {'train','val','test','ood_funnel','ood_cups'}
    if set(result['split_pair_counts'])!=expected_splits:raise RuntimeError('Unexpected evaluation split set')
    if Path(result['checkpoint']).resolve()!=(out/'encoder.pt').resolve():raise RuntimeError('Evaluation checkpoint path mismatch')
    digest=hashlib.sha256()
    with (out/'encoder.pt').open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=result['checkpoint_sha256']:raise RuntimeError('Evaluation checkpoint digest mismatch')
    eval_complete=load(out/'evaluation/EVALUATION_COMPLETED.json')
    if eval_complete.get('checkpoint_sha256')!=digest.hexdigest() or eval_complete.get('mode')!=mode:raise RuntimeError('Evaluation completion identity mismatch')
    if phase=='final' and set(result.get('mlp_difference_ablation',{}))!={'concat','concat_difference'}:raise RuntimeError('Required paired MLP comparison missing')
    if phase=='final':
        visual=load(out/'visualization/visualization.json')
        if visual.get('checkpoint_sha256')!=digest.hexdigest() or visual.get('split')!='val' or visual.get('training_performed') is not False:raise RuntimeError('Reconstruction visualization identity mismatch')
    cloud.save(cached,{'verified_at':cloud.now(),'expected':expected,'files':signature,'checkpoint_sha256':digest.hexdigest()})
    return out/'evaluation/evaluation.json'


def advance():
    with (CLOUD/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_release()
        if not (CLOUD/'TRAINING_ENABLED.json').is_file():raise RuntimeError('Training not enabled: GPU smoke and data QA are required')
        if not (DATA/'READY.json').is_file() or not (ROOT/'data/DATA_QA_APPROVED.json').is_file():
            raise RuntimeError('Prepared data not yet verified')
        approval=load(ROOT/'data/DATA_QA_APPROVED.json')
        if hashlib.sha256((DATA/'integrity.json').read_bytes()).hexdigest()!=approval['integrity_sha256']:
            raise RuntimeError('Prepared data integrity manifest changed after QA')
        c=cloud.client();rows=cloud.statuses(c)
        byname={r['logical']:r for r in rows}
        active=[r for r in rows if r['Status'] not in TERMINAL]
        experiment_rows=[r for r in byname.values() if r['logical'].startswith(('screen_','final_'))]
        failed=[r for r in experiment_rows if r['Status'] in {'Failed','Stopped'}]
        if failed:
            state={'phase':'needs_attention','updated_at':cloud.now(),'failed':failed,'active_jobs':len(active)}
            cloud.save(CLOUD/'pipeline_state.json',state)
            return state
        screen=screened()
        for name,weight in screen:
            if name in byname and byname[name]['Status']=='Succeeded':require_artifacts(name,'screen',weight,20,0)
        pending_screen=[(n,w) for n,w in screen if n not in byname]
        incomplete_screen=[(n,w) for n,w in screen if n not in byname or byname[n]['Status']!='Succeeded']
        if incomplete_screen:
            submitted=[]
            for name,weight in pending_screen[:max(0,2-len(active))]:
                command=training_command(name,weight,20,0,'screen')
                submitted.append(cloud.submit(c,name,command,gpu=1,minutes=360))
            state={'phase':'screening','updated_at':cloud.now(),'submitted':submitted,'screening_complete':len(screen)-len(incomplete_screen),'screening_total':4,'active_jobs':len(active)+len(submitted)}
            cloud.save(CLOUD/'pipeline_state.json',state);return state
        selection_path=ROOT/'reports/selection.json'
        if not selection_path.exists():
            results=[require_artifacts(n,'screen',w,20,0) for n,w in screen]
            call_local([RELEASE/'scripts/select_lambda.py','--results',*results,'--out',selection_path])
        selection=load(selection_path);weight=float(selection['selected_lambda'])
        if weight not in [.1,.3,.5]:raise RuntimeError('Selected force comparison must use one of the predeclared weights')
        final=[(f'final_{variant}_s{seed}',lam,seed) for seed in [0,1,2] for variant,lam in [('mae',0.),('force',weight)]]
        for name,w,seed in final:
            if name in byname and byname[name]['Status']=='Succeeded':require_artifacts(name,'final',w,100,seed)
        incomplete=[x for x in final if x[0] not in byname or byname[x[0]]['Status']!='Succeeded']
        if incomplete:
            submitted=[]
            for name,w,seed in [x for x in final if x[0] not in byname][:max(0,2-len(active))]:
                command=training_command(name,w,100,seed,'final')
                submitted.append(cloud.submit(c,name,command,gpu=1,minutes=1440))
            state={'phase':'final_training','updated_at':cloud.now(),'selected_lambda':weight,'force_accepted_as_improvement':selection.get('accepted_as_improvement'),'submitted':submitted,'final_complete':len(final)-len(incomplete),'final_total':6,'active_jobs':len(active)+len(submitted)}
            cloud.save(CLOUD/'pipeline_state.json',state);return state
        results=[require_artifacts(n,'final',w,100,seed) for n,w,seed in final]
        call_local([RELEASE/'scripts/build_report.py','--results',*results,'--selection',selection_path,'--out',ROOT/'reports/final'])
        renderer=load(ROOT/'configs/report_renderer.json')
        render_path=(ROOT/renderer['path']).resolve();render_path.relative_to(ROOT)
        if hashlib.sha256(render_path.read_bytes()).hexdigest()!=renderer['sha256']:raise RuntimeError('Frozen report renderer changed')
        call_local([render_path,'--report-dir',ROOT/'reports/final','--project-root',ROOT])
        if not load(ROOT/'reports/final/reviewed_report_receipt.json').get('complete_evidence'):raise RuntimeError('Report completion evidence did not pass')
        state={'phase':'complete','updated_at':cloud.now(),'selected_lambda':weight,'final_complete':6,'active_jobs':len(active),'report_directory':str(ROOT/'reports/final')}
        cloud.save(CLOUD/'pipeline_state.json',state);return state


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['once','run']);p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    daemon_lock=None
    if a.command=='run':
        daemon_lock=(CLOUD/'daemon.lock').open('a')
        try:fcntl.flock(daemon_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('TACTILE_PIPELINE_DAEMON_ALREADY_RUNNING',flush=True);return 0
    while True:
        try:state=advance();print('TACTILE_PIPELINE '+json.dumps(state,ensure_ascii=False),flush=True)
        except Exception as exc:
            state={'phase':'controller_error','updated_at':cloud.now(),'error_type':type(exc).__name__,'code':getattr(exc,'code',None)}
            # Application-originated invariants are safe; provider request internals are not.
            if isinstance(exc,(RuntimeError,ValueError,FileNotFoundError)):state['reason']=str(exc)
            cloud.save(CLOUD/'pipeline_state.json',state);print('TACTILE_PIPELINE '+json.dumps(state),flush=True)
            if a.command=='once':return 1
        if a.command=='once' or state.get('phase') in ['complete','needs_attention']:return 0
        time.sleep(max(30,a.interval))

if __name__=='__main__':raise SystemExit(main())
