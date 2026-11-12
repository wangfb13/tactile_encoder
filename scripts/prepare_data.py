#!/usr/bin/env python
"""Build reproducible paired DD data; every write is under tactile_encoder/data.

Only the train partition sets normalization and contact thresholds. Force-based
sampling is a proxy for temporal strata, not a ground-truth contact annotation.
Recorded DD/state alignment is checked; original device clock freshness cannot
be recovered from these trainready exports and is explicitly not asserted.
"""
import argparse, collections, concurrent.futures, hashlib, json, math, os, sys, time
from pathlib import Path
import av
import cv2
import numpy as np
import pyarrow.parquet as pq

PROJECT=Path(__file__).resolve().parents[1]
DATA=PROJECT/'data'
SOURCES=[
 ('green_normal','wfb_data/261022_greenbottle_normal_30_trainready_224',2000,'right'),
 ('vase','wfb_data/261105_vase_128_trainready_224_small_tag',6000,'both'),
 ('wire','wfb_data/261020_straighten_wire_trainready_224',4000,'right'),
 ('pinggai','wfb_data/261021_pinggai_trainready_224',2000,'right'),
 ('green_guo','wfb_data/261022_greenbottle_guo_30_trainready_224',2000,'right'),
 ('cable','wfb_data/261020_pick_cable_1_trainready_224',3000,'right'),
 ('wipe','train_ready/261012_wipe_ss_plate_trainready_224_small_tag',6000,'left'),
 ('funnel','wfb_data/261009_funnel_4stage_test_trainready_224_small_tag',0,'both'),
 ('cups','train_ready/260917_three_cup_DAP9122676008694_trainready_224_small_tag',0,'both'),
]
SEED=20261112
AXES=['Fx','Fy','Fz','Mx','My','Mz']
GEOMETRY=['contact','cx','cy','area','mxx','myy','mxy']

def atomic_json(path,obj):
    path=Path(path); p=path.with_suffix(path.suffix+'.tmp'); p.write_text(json.dumps(obj,indent=2,ensure_ascii=False));os.replace(p,path)

def log(event,**fields): print(json.dumps(dict(event=event,**fields),ensure_ascii=False),flush=True)

def sha(data): return hashlib.sha256(data).hexdigest()

# Every accepted source declares unified_format.missing_value=9930 in meta/info.json.
def force_valid(a): return np.isfinite(a) & (a!=9930.)

def get_episode_metadata(root):
    rows=[]
    for path in sorted((root/'meta/episodes').rglob('*.parquet')):
        table=pq.read_table(path)
        keep=[n for n in table.column_names if not n.startswith('stats/') or n in ['stats/observation.state/mean','stats/observation.state/max']]
        rows.extend(table.select(keep).to_pylist())
    return sorted(rows,key=lambda x:x['episode_index'])

def build_index():
    index_path=DATA/'source_index.json'
    if index_path.exists():
        cached=json.loads(index_path.read_text())
        for item in cached:
            info_bytes=(Path(item['root'])/'meta/info.json').read_bytes()
            assert sha(info_bytes)==item['info_sha256'], 'Source metadata changed; inspect before resuming'
            assert json.loads(info_bytes)['unified_format']['missing_value']==9930
        return cached
    result=[]; fingerprint_groups={}
    for name,rel,quota,hand in SOURCES:
        root=Path('/mnt/oss_umi_validation')/rel
        info=json.loads((root/'meta/info.json').read_text()); names=info['features']['observation.state']['names']
        assert info['unified_format']['missing_value']==9930, 'Unexpected source missing-value convention'
        ids=np.array([[[names.index(f'tactile_{h}_{f}_{a}') for a in AXES] for f in ['left','right']] for h in ['left','right']])
        episodes=get_episode_metadata(root)
        tables={};ep_summary=[]
        cache_dir=DATA/'index'/name;cache_dir.mkdir(parents=True,exist_ok=True)
        for ep in episodes:
            key=(ep['data/chunk_index'],ep['data/file_index'])
            if key not in tables:
                p=root/info['data_path'].format(chunk_index=key[0],file_index=key[1])
                table=pq.read_table(p,columns=['observation.state','episode_index','frame_index','timestamp','index'])
                tables[key]={n:table[n].to_pylist() for n in table.column_names}
            table=tables[key];select=np.flatnonzero(np.asarray(table['episode_index'])==ep['episode_index'])
            state=np.asarray([table['observation.state'][i] for i in select],dtype=np.float32)
            frame=np.asarray([table['frame_index'][i] for i in select],dtype=np.int64)
            timestamp=np.asarray([table['timestamp'][i] for i in select],dtype=np.float64)
            globalindex=np.asarray([table['index'][i] for i in select],dtype=np.int64)
            assert len(state)==ep['length'],(name,ep['episode_index'],'length mismatch')
            assert len(np.unique(frame))==len(frame) and np.all(np.diff(timestamp)>0),(name,ep['episode_index'],'nonunique/nonmonotonic')
            assert np.max(np.abs((timestamp-timestamp[0])-(frame-frame[0])/info['fps'])) < .5/info['fps'], 'state timeline mismatch'
            fingerprint=sha(np.nan_to_num(state,nan=9930).tobytes()+np.asarray([len(state)],dtype=np.int64).tobytes())
            group=fingerprint_groups.setdefault(fingerprint,[]);group.append([name,int(ep['episode_index'])])
            f=state[:,ids]
            destination=cache_dir/f"episode_{ep['episode_index']:05d}.npz"
            np.savez(destination,force=f,frame=frame,timestamp=timestamp,index=globalindex)
            ep_summary.append(dict(episode=int(ep['episode_index']),length=len(state),fingerprint=fingerprint,cache=str(destination.relative_to(PROJECT)),meta=ep))
        # Group exact duplicate episode exports together before partitioning.
        groups=sorted(set(e['fingerprint'] for e in ep_summary));rng=np.random.default_rng(SEED+sum(name.encode()));rng.shuffle(groups)
        if quota:
            nv=max(1,round(len(groups)*.1)); nt=max(1,round(len(groups)*.1))
            assignment={g:('val' if i<nv else 'test' if i<nv+nt else 'train') for i,g in enumerate(groups)}
        else: assignment={g:'ood_'+name for g in groups}
        for e in ep_summary: e['split']=assignment[e['fingerprint']]
        result.append(dict(name=name,root=str(root),quota=quota,hand=hand,fps=info['fps'],info_sha256=sha((root/'meta/info.json').read_bytes()),episodes=ep_summary))
        log('indexed',dataset=name,episodes=len(ep_summary))
    # Exact duplicates across datasets must not cross splits; fail rather than hide leakage.
    assignments={}
    for ds in result:
        for e in ds['episodes']:
            old=assignments.setdefault(e['fingerprint'],e['split'])
            if old!=e['split']: raise RuntimeError('Cross-dataset duplicate episode crosses partitions: '+str(fingerprint_groups[e['fingerprint']]))
    atomic_json(DATA/'exact_duplicate_groups.json',[x for x in fingerprint_groups.values() if len(x)>1])
    atomic_json(index_path,result);return result

def choose_stratified(rows,n,rng):
    if len(rows)<n: raise RuntimeError(f'Only {len(rows)} unique candidates for {n} samples')
    # Keep real zero-force frames, invalid force remains an eligible unknown stratum.
    active=[x for x in rows if x['stratum']=='active']; empty=[x for x in rows if x['stratum']=='low_force']; unknown=[x for x in rows if x['stratum']=='unknown']
    rng.shuffle(active);rng.shuffle(empty);rng.shuffle(unknown)
    nempty=min(round(n*.15),len(empty));nunknown=min(round(n*.05),len(unknown))
    chosen=empty[:nempty]+unknown[:nunknown]+active[:n-nempty-nunknown]
    selected={(x['episode'],x['hand'],x['row']) for x in chosen}
    leftovers=[x for x in rows if (x['episode'],x['hand'],x['row']) not in selected]
    rng.shuffle(leftovers); chosen+=leftovers[:n-len(chosen)];rng.shuffle(chosen)
    return chosen

def make_plan(index):
    dest=DATA/'candidate_plan.json'
    if dest.exists(): return json.loads(dest.read_text())
    rng=np.random.default_rng(SEED);plan=[]
    for ds in index:
        split_names=['train','val','test'] if ds['quota'] else ['ood_'+ds['name']]
        for split in split_names:
            rows=[]
            for ep in ds['episodes']:
                if ep['split']!=split:continue
                cached=np.load(PROJECT/ep['cache']);f=cached['force'];hands=['left','right'] if ds['hand']=='both' else [ds['hand']]
                # 3-frame spacing (10 Hz max); final selections are much sparser.
                for h in hands:
                    hi=['left','right'].index(h)
                    for row in range(int(ep['episode'])%3,len(f),3):
                        value=f[row,hi];valid=force_valid(value)
                        active=np.linalg.norm(np.where(valid[:,:3],value[:,:3],0),axis=1).max()>.5
                        rows.append(dict(dataset=ds['name'],episode=ep['episode'],hand=h,row=row,stratum='active' if active else ('low_force' if valid.all() else 'unknown')))
            target=ds['quota'] if split=='train' else (max(1,ds['quota']//8) if ds['quota'] else min(2000,len(rows)))
            oversample=min(len(rows),int(math.ceil(target*1.25)))
            selected=choose_stratified(rows,oversample,rng)
            plan.append(dict(dataset=ds['name'],split=split,target=target,candidates=selected))
            log('planned',dataset=ds['name'],split=split,target=target,candidates=len(selected))
    atomic_json(dest,plan);return plan

def video_key(hand,finger):return f'observation.depth_deformation.tactile_{hand}_{finger}'

def decode_times(path,times,fps):
    """Read true 16-bit RGB, matching declared video times to <= half a frame."""
    order=np.argsort(times); wanted=np.asarray(times)[order];results={}; skews={}; cursor=0
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='AUTO'
        if stream.codec_context.format.name != 'gbrp16le':raise ValueError('Unexpected DD pixel format')
        container.seek(max(0,int((wanted[0]-.1)/float(stream.time_base))),stream=stream,backward=True,any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:raise ValueError('DD frame without PTS')
            t=float(frame.pts*stream.time_base)
            while cursor<len(wanted) and wanted[cursor]<t-.51/fps:
                cursor+=1
            if cursor>=len(wanted):break
            while cursor<len(wanted) and abs(t-wanted[cursor])<=.51/fps:
                a=frame.to_ndarray(format='rgb48le')
                if a.dtype!=np.uint16 or a.shape!=(96,128,3):raise ValueError('Unexpected DD shape or depth')
                results[int(order[cursor])]=a;skews[int(order[cursor])]=abs(t-wanted[cursor]);cursor+=1
            if cursor>=len(wanted):break
    return results,skews

def process_episode(job):
    ds,ep,rows=job;out=DATA/'episode_cache'/ds['name']/f"episode_{ep['episode']:05d}.npz";out.parent.mkdir(parents=True,exist_ok=True)
    request_hash=sha(json.dumps(rows,sort_keys=True).encode())
    if out.exists():
        z=np.load(out)
        if str(z['request_hash'])==request_hash:return str(out)
        raise RuntimeError('Episode cache plan changed')
    raw=np.load(PROJECT/ep['cache']);dd=np.zeros((len(rows),2,96,128,3),dtype=np.uint16); valid=np.ones(len(rows),dtype=bool)
    skew=np.full((len(rows),2),np.nan);files=[None]*len(rows)
    for h in ['left','right']:
        positions=[i for i,r in enumerate(rows) if r['hand']==h]
        if not positions:continue
        for fi,finger in enumerate(['left','right']):
            key=video_key(h,finger);prefix='videos/'+key
            c=ep['meta'][prefix+'/chunk_index'];f=ep['meta'][prefix+'/file_index'];start=ep['meta'][prefix+'/from_timestamp']
            path=Path(ds['root'])/f'videos/{key}/chunk-{c:03d}/file-{f:03d}.mkv'
            times=[start+float(raw['timestamp'][rows[i]['row']]) for i in positions]
            images,errors=decode_times(path,times,ds['fps'])
            for local,i in enumerate(positions):
                if local not in images:valid[i]=False;continue
                a=images[local]
                # All-zero deformation is exporter missing-data, not unloaded 30000.
                if np.any(np.all(a[:,:,1:]==0,axis=(0,1))):valid[i]=False
                dd[i,fi]=a;skew[i,fi]=errors[local]
            for i in positions:files[i]=str(path.parent.parent.relative_to(Path(ds['root'])))
    forces=np.stack([raw['force'][r['row'],['left','right'].index(r['hand'])] for r in rows])
    records=[]
    for i,r in enumerate(rows):
        j=r['row']; records.append(dict(r,frame_index=int(raw['frame'][j]),timestamp=float(raw['timestamp'][j]),index=int(raw['index'][j]),
            episode_fingerprint=ep['fingerprint'],source_root=ds['root'],dd_alignment_error_s=skew[i].tolist(),alignment='recorded_video_PTS_to_episode_timestamp',
            raw_sensor_clock_available=False,valid_dd=bool(valid[i])))
    temp=out.with_suffix('.tmp.npz');np.savez(temp,dd=dd,force=forces,force_valid=force_valid(forces),valid=valid,records=json.dumps(records),request_hash=request_hash);os.replace(temp,out)
    return str(out)

def contact_mask(image,threshold):
    raw=(image[:,:,0]>threshold).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw,connectivity=8)
    keep=np.zeros(count,dtype=bool)
    if count>1:keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=12
    return keep[labels]

def geometry(image,threshold):
    mask=contact_mask(image,threshold)
    if mask.sum()<12:return np.array([0,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],dtype=np.float32)
    yy,xx=np.nonzero(mask);x=(xx+.5)/image.shape[1];y=(yy+.5)/image.shape[0];cx=x.mean();cy=y.mean()
    return np.array([1,cx,cy,mask.mean(),np.mean((x-cx)**2),np.mean((y-cy)**2),np.mean((x-cx)*(y-cy))],dtype=np.float32)

def finalize(index,plan):
    prepared=DATA/'prepared';prepared.mkdir(exist_ok=True)
    refs={};all_records={};cache={};rng=np.random.default_rng(SEED)
    for entry in plan:
        rows=[]
        for epno in sorted(set(r['episode'] for r in entry['candidates'])):
            path=DATA/'episode_cache'/entry['dataset']/f'episode_{epno:05d}.npz'
            with np.load(path) as z:
                records=json.loads(str(z['records']));valid=z['valid']
            for j,r in enumerate(records):
                if valid[j]:rows.append(dict(r,cache=str(path),cache_row=j))
        candidates={(r['episode'],r['hand'],r['row']) for r in entry['candidates']}
        rows=[r for r in rows if (r['episode'],r['hand'],r['row']) in candidates]
        chosen=choose_stratified(rows,entry['target'],rng)
        refs.setdefault(entry['split'],[]).extend(chosen)
    # Derive depth threshold from low-force *training* samples only; no test tuning.
    background=[]; low_by_dataset=collections.defaultdict(list)
    for r in refs['train']:
        if r['stratum']=='low_force':low_by_dataset[r['dataset']].append(r)
    for name,low in sorted(low_by_dataset.items()):
        local=np.random.default_rng(SEED+sum(name.encode()))
        for index in local.choice(len(low),size=min(64,len(low)),replace=False):
            r=low[int(index)]
            with np.load(r['cache']) as z:
                a=z['dd'][r['cache_row']]
                background.extend(a[:,::4,::4,0].reshape(-1).tolist())
    if not background: raise RuntimeError('No train low-force depth samples for proxy threshold')
    threshold=max(20.,float(np.quantile(background,.995)))
    atomic_json(DATA/'geometry_rule.json',dict(depth_threshold=threshold,min_pixels=12,source='task_balanced_train_low_force_depth_q995',connected_components=8,label_type='proxy_not_independent_ground_truth'))
    stats_sum=np.zeros(3);stats_sq=np.zeros(3);pixel_n=0;force_sum=np.zeros(6);force_sq=np.zeros(6);force_n=np.zeros(6)
    summaries={}
    for split,rows in refs.items():
        rng.shuffle(rows);n=len(rows);tmpdir=prepared/('building_'+split);tmpdir.mkdir(exist_ok=True)
        arrays={
          'dd':np.lib.format.open_memmap(tmpdir/'dd.npy',mode='w+',dtype=np.uint16,shape=(n,2,96,128,3)),
          'force':np.lib.format.open_memmap(tmpdir/'force.npy',mode='w+',dtype=np.float32,shape=(n,2,6)),
          'force_valid':np.lib.format.open_memmap(tmpdir/'force_valid.npy',mode='w+',dtype=bool,shape=(n,2,6)),
          'geometry':np.lib.format.open_memmap(tmpdir/'geometry.npy',mode='w+',dtype=np.float32,shape=(n,2,7))}
        grouped=collections.defaultdict(list)
        for i,r in enumerate(rows):grouped[r['cache']].append((i,r))
        manifest=[None]*n
        for path,items in grouped.items():
            with np.load(path) as z:
                d=z['dd'];f=z['force'];v=z['force_valid']
                for i,r in items:
                    j=r['cache_row'];a=d[j];ff=f[j];vv=v[j]
                    arrays['dd'][i]=a;arrays['force'][i]=ff;arrays['force_valid'][i]=vv
                    arrays['geometry'][i]=np.stack([geometry(x,threshold) for x in a])
                    manifest[i]={k:value for k,value in r.items() if k not in ('cache','cache_row','valid_dd')}
                    if split=='train':
                        x=a.astype(np.float64)-np.array([0,30000,30000]);stats_sum+=x.sum(axis=(0,1,2));stats_sq+=(x*x).sum(axis=(0,1,2));pixel_n+=x.shape[0]*x.shape[1]*x.shape[2]
                        force_sum+=np.where(vv,ff,0).sum(0);force_sq+=np.where(vv,ff.astype(np.float64)**2,0).sum(0);force_n+=vv.sum(0)
        for key,arr in arrays.items():arr.flush()
        contact=arrays['geometry'][:,:,0];summaries[split]=dict(pairs=n,single_fingers=2*n,datasets=dict(collections.Counter(r['dataset'] for r in rows)),empty_finger_fraction=float((contact==0).mean()),force_valid_fraction=float(arrays['force_valid'].mean()))
        text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest);(tmpdir/'manifest.jsonl').write_text(text)
        for key in arrays: os.replace(tmpdir/(key+'.npy'),prepared/f'{split}_{key}.npy')
        os.replace(tmpdir/'manifest.jsonl',prepared/f'{split}_manifest.jsonl');tmpdir.rmdir()
        summaries[split]['manifest_sha256']=sha(text.encode());log('finalized',split=split,**summaries[split])
    mean=stats_sum/pixel_n;std=np.sqrt(np.maximum(stats_sq/pixel_n-mean**2,1.))
    if (force_n==0).any():raise RuntimeError('A force component has no valid training labels')
    fm=force_sum/force_n;fs=np.sqrt(np.maximum(force_sq/force_n-fm**2,1e-8))
    norm=dict(offset=[0,30000,30000],mean=mean.tolist(),std=std.tolist(),force_mean=fm.tolist(),force_std=fs.tolist(),force_count=force_n.tolist(),geometry_names=GEOMETRY,source_split='train',force_units='original_X5_SDK_export_units',no_per_frame_normalization=True,shape=[96,128,3],target_shape=[192,256])
    atomic_json(prepared/'normalization.json',norm)
    atomic_json(prepared/'summary.json',dict(splits=summaries,geometry_rule=json.loads((DATA/'geometry_rule.json').read_text()),source_counts={d['name']:len(d['episodes']) for d in index},alignment_limit='Trainready video/state timestamp alignment checked; raw device clocks unavailable',exact_duplicate_check=True,excluded_dataset='pen: deformation exporter anomaly',seed=SEED))
    atomic_json(prepared/'READY.json',dict(ready=True,completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),splits=summaries))

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--index-only',action='store_true');args=p.parse_args()
    DATA.mkdir(exist_ok=True);index=build_index();plan=make_plan(index)
    if args.index_only:return
    lookup={(d['name'],e['episode']):(d,e) for d in index for e in d['episodes']};grouped=collections.defaultdict(list)
    for entry in plan:
        for r in entry['candidates']:grouped[(r['dataset'],r['episode'])].append(r)
    jobs=[(*lookup[key],rows) for key,rows in grouped.items()]
    log('decode_start',episodes=len(jobs),workers=args.workers)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i,result in enumerate(executor.map(process_episode,jobs),1):
            if i%10==0 or i==len(jobs):log('decoded',completed=i,total=len(jobs),last=result)
    finalize(index,plan);log('ready',directory=str(DATA/'prepared'))


def audit_prepared(recalibrate=True):
    """Task-balanced train-only geometry calibration and transparent proxy-label QA."""
    prepared=DATA/'prepared';assert (prepared/'READY.json').exists()
    atomic_json(prepared/'QA_READY.json',dict(ready=False,status='auditing'))
    manifest=[json.loads(x) for x in (prepared/'train_manifest.jsonl').read_text().splitlines()]
    train=np.load(prepared/'train_dd.npy',mmap_mode='r'); groups=collections.defaultdict(list)
    for i,r in enumerate(manifest):
        if r['stratum']=='low_force':groups[r['dataset']].append(i)
    background=[];background_stats={}
    for name,indices in sorted(groups.items()):
        rng=np.random.default_rng(SEED+sum(name.encode()));chosen=rng.choice(indices,size=min(64,len(indices)),replace=False)
        pixels=np.concatenate([train[int(i),:,::4,::4,0].reshape(-1) for i in chosen]);background.extend(pixels.tolist())
        background_stats[name]=dict(pairs=len(chosen),q50_q95_q995=np.quantile(pixels,[.5,.95,.995]).tolist())
    threshold=max(20.,float(np.quantile(background,.995)))
    if not recalibrate:threshold=json.loads((DATA/'geometry_rule.json').read_text())['depth_threshold']
    rule=dict(depth_threshold=threshold,min_pixels=12,source='task_balanced_train_low_force_depth_q995',connected_components=8,label_type='proxy_not_independent_ground_truth',background_by_dataset=background_stats)
    atomic_json(DATA/'geometry_rule.json',rule)
    report=dict(rule=rule,groups={},split_episode_overlap=False,source_missing_value=9930,source_missing_value_evidence='data/source_metadata_evidence.json')
    split_groups={};examples=[]
    for split in ['train','val','test','ood_funnel','ood_cups']:
        rows=[json.loads(x) for x in (prepared/f'{split}_manifest.jsonl').read_text().splitlines()]
        dd=np.load(prepared/f'{split}_dd.npy',mmap_mode='r');fp=np.load(prepared/f'{split}_force_valid.npy',mmap_mode='r')
        gp=prepared/f'{split}_geometry.npy';geom=np.load(gp,mmap_mode='r+')
        if recalibrate:
            for i in range(len(dd)):geom[i]=np.stack([geometry(x,threshold) for x in dd[i]])
            geom.flush()
        names=collections.defaultdict(list)
        for i,r in enumerate(rows):names[(r['dataset'],r['hand'])].append(i)
        split_groups[split]={r['episode_fingerprint'] for r in rows}
        for (dataset,hand),idx in names.items():
            a=np.asarray(geom[idx]);contacts=a[:,:,0]>0;areas=a[:,:,3];centers=a[:,:,1:3]
            report['groups'][f'{split}/{dataset}/{hand}']=dict(pairs=len(idx),finger_contact_fraction=float(contacts.mean()),both_contact_fraction=float(contacts.all(1).mean()),both_empty_fraction=float((~contacts).all(1).mean()),force_valid_fraction=float(fp[idx].mean()),
                contact_area_q10_q50_q90=np.quantile(areas[contacts],[.1,.5,.9]).tolist() if contacts.any() else None,
                contact_centroid_mean=np.mean(centers[contacts],axis=0).tolist() if contacts.any() else None,
                geometric_finite_fraction=float(np.isfinite(a[:,:,1:][contacts]).mean()) if contacts.any() else None)
            if split=='train':
                # Show a low-contact and a typical-contact example for each dataset/hand.
                pairarea=np.nan_to_num(a[:,:,3],nan=0).sum(1); order=np.argsort(pairarea)
                for j in [order[0],order[len(order)//2]]:examples.append((dataset,hand,rows[idx[int(j)]],np.array(dd[idx[int(j)]])))
        assert np.all(np.isfinite(geom[:,:,1:][geom[:,:,0]>0]))
    names=list(split_groups)
    for i,a in enumerate(names):
        for b in names[i+1:]:assert not(split_groups[a]&split_groups[b]),'Episode leakage '+a+' '+b
    training_groups=[v for k,v in report['groups'].items() if k.startswith('train/')]
    assert all(0<v['finger_contact_fraction']<1 for v in training_groups),'A training dataset/hand lacks contact or no-contact probe labels; inspect'
    summary=json.loads((prepared/'summary.json').read_text());summary['geometry_rule']=rule
    for split in summary['splits']:
        g=np.load(prepared/f'{split}_geometry.npy',mmap_mode='r');summary['splits'][split]['empty_finger_fraction']=float((g[:,:,0]==0).mean())
    atomic_json(prepared/'summary.json',summary)
    qa=DATA/'qa';qa.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(examples),4,figsize=(12,2*len(examples)),squeeze=False)
    for row,(dataset,hand,meta,pair) in enumerate(examples):
        for finger in range(2):
            depth=pair[finger,:,:,0];mask=contact_mask(pair[finger],threshold)
            axes[row,2*finger].imshow(depth,cmap='viridis',vmin=0,vmax=max(100,float(np.quantile(depth,.99))))
            axes[row,2*finger+1].imshow(mask,cmap='gray',vmin=0,vmax=1)
            axes[row,2*finger].set_title(f"{dataset}/{hand} ep{meta['episode']} f{meta['frame_index']} finger{finger}",fontsize=8)
            axes[row,2*finger+1].set_title(f'depth>{threshold:g}; area={mask.mean():.3f}',fontsize=8)
        for ax in axes[row]:ax.axis('off')
    fig.tight_layout();fig.savefig(qa/'contact_examples.png',dpi=100);plt.close(fig)
    atomic_json(qa/'quality.json',report)
    integrity={}
    for path in sorted(prepared.iterdir()):
        if path.suffix not in ('.npy','.jsonl') and path.name!='normalization.json':continue
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while True:
                block=stream.read(16*1024*1024)
                if not block:break
                digest.update(block)
        integrity[path.name]=dict(sha256=digest.hexdigest(),size=path.stat().st_size)
    atomic_json(prepared/'integrity.json',dict(algorithm='sha256',files=integrity))
    atomic_json(prepared/'QA_READY.json',dict(ready=True,depth_threshold=threshold,geometry_rule_sha256=sha(json.dumps(rule,sort_keys=True).encode()),no_episode_overlap=True,source_sentinel_verified=True,qa_report='data/qa/quality.json'))
    log('quality_ready',threshold=threshold,groups=report['groups'])

if __name__=='__main__':
    main()
    if '--index-only' not in sys.argv:
        audit_prepared(recalibrate=True)
