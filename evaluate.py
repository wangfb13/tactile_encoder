"""Frozen tactile probes. Test/OOD data are inaccessible in --selection-only mode."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

GEOMETRY_NAMES = ["cx", "cy", "area", "mxx", "myy", "mxy"]
RIDGE_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0]

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)

def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def manifests(root, split):
    path = Path(root) / (split + "_manifest.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows

def group_key(row):
    return str(row["dataset"]) + ":" + str(row["episode"])

def assert_no_episode_overlap(rows_by_split):
    seen = {}
    source_seen = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            key = group_key(row)
            if key in seen and seen[key] != split:
                raise ValueError("Episode leakage: " + key)
            seen[key] = split
            fingerprint = row.get("episode_fingerprint")
            if fingerprint:
                if fingerprint in source_seen and source_seen[fingerprint] != split:
                    raise ValueError("Source-episode fingerprint leakage")
                source_seen[fingerprint] = split

def standardize(train, *others):
    mean = np.asarray(train, np.float64).mean(axis=0)
    std = np.asarray(train, np.float64).std(axis=0)
    std = np.maximum(std, 1e-4)
    results = [np.asarray((x - mean) / std, np.float32) for x in (train,) + others]
    return results, mean.astype(np.float32), std.astype(np.float32)

def target_stats(y, valid):
    count = valid.sum(axis=0)
    mean = np.where(count > 0, np.where(valid, y, 0).sum(axis=0) / np.maximum(count, 1), 0.)
    var = np.where(valid, (np.nan_to_num(y) - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-4).astype(np.float32), count

def masked_axis_rmse(pred, y, valid):
    count = valid.sum(axis=0)
    sq = np.where(valid, (pred - np.nan_to_num(y)) ** 2, 0).sum(axis=0)
    return np.sqrt(sq / np.maximum(count, 1)), count

def finite_mean(values, default=None):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else default

def regression_metrics(pred, y, valid, scale, groups, datasets, bootstrap=300):
    valid = valid & np.isfinite(y) & np.isfinite(pred)
    rmse, count = masked_axis_rmse(pred, y, valid)
    present = count > 0
    nr = rmse / np.maximum(scale, 1e-4)
    by_task = {}
    datasets = np.asarray(datasets)
    for task in sorted(set(datasets)):
        selected = datasets == task
        r, n = masked_axis_rmse(pred[selected], y[selected], valid[selected])
        by_task[str(task)] = {"n": int(selected.sum()), "nrmse": finite_mean((r / scale)[n > 0])}
    result = {
        "rmse_per_component": [float(v) if n > 0 else None for v, n in zip(rmse, count)],
        "valid_per_component": count.astype(int).tolist(),
        "nrmse": finite_mean(nr[present]), "by_task": by_task,
        "macro_task_nrmse": finite_mean([x["nrmse"] for x in by_task.values()]),
    }
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    if bootstrap and len(unique) >= 2:
        sse = np.zeros((len(unique), y.shape[1]), np.float64)
        cnt = np.zeros_like(sse)
        np.add.at(sse, inverse, np.where(valid, (pred - np.nan_to_num(y)) ** 2, 0))
        np.add.at(cnt, inverse, valid.astype(np.float64))
        task_by_group = [str(datasets[np.flatnonzero(inverse == i)[0]]) for i in range(len(unique))]
        task_groups = {task: np.flatnonzero(np.asarray(task_by_group) == task) for task in sorted(set(task_by_group))}
        rng = np.random.default_rng(20261112)
        scores = []
        for _ in range(bootstrap):
            task_scores = []
            for ix in task_groups.values():
                drawn = rng.choice(ix, len(ix), replace=True)
                n = cnt[drawn].sum(axis=0)
                r = np.sqrt(sse[drawn].sum(axis=0) / np.maximum(n, 1)) / scale
                task_scores.append(finite_mean(r[n > 0]))
            scores.append(finite_mean(task_scores))
        scores = [v for v in scores if v is not None and np.isfinite(v)]
        if scores:
            result["macro_task_nrmse_episode_bootstrap_95ci"] = np.quantile(scores, [.025, .975]).tolist()
    result["episode_count"] = int(len(unique))
    return result

def contact_metrics(prob, y, groups, datasets, bootstrap=300):
    pred = prob >= .5
    y = y >= .5
    datasets = np.asarray(datasets)
    def stats(p, t, probs):
        recall = []
        f1 = []
        for j in range(t.shape[1]):
            for label in [False, True]:
                truth = t[:, j] == label
                guess = p[:, j] == label
                tp = np.sum(truth & guess); fn = np.sum(truth & ~guess); fp = np.sum(~truth & guess)
                if tp + fn:
                    recall.append(tp / (tp + fn))
                    f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        return {"balanced_accuracy": finite_mean(recall), "macro_f1": finite_mean(f1),
                "brier": float(np.mean((probs - t) ** 2))}
    r = stats(pred, y, prob)
    r["by_task"] = {str(k): stats(pred[datasets == k], y[datasets == k], prob[datasets == k])
                    for k in sorted(set(datasets))}
    r["macro_task_balanced_accuracy"] = finite_mean([v["balanced_accuracy"] for v in r["by_task"].values()])
    # Resample complete episodes, stratified by task; both fingers stay together.
    unique, inverse = np.unique(np.asarray(groups), return_inverse=True)
    r["episode_count"] = int(len(unique))
    if bootstrap and len(unique) >= 2:
        # Per-episode confusion counts avoid copying many frames at each draw.
        counts = np.zeros((len(unique), y.shape[1], 4), np.float64)
        for j in range(y.shape[1]):
            for k, mask in enumerate([pred[:, j] & y[:, j], ~pred[:, j] & ~y[:, j],
                                       pred[:, j] & ~y[:, j], ~pred[:, j] & y[:, j]]):
                np.add.at(counts[:, j, k], inverse, mask.astype(float))
        task_group = np.array([datasets[np.flatnonzero(inverse == i)[0]] for i in range(len(unique))])
        rng = np.random.default_rng(20261112)
        scores = []
        for _ in range(bootstrap):
            task_scores = []
            for task in sorted(set(task_group)):
                ix = np.flatnonzero(task_group == task)
                c = counts[rng.choice(ix, len(ix), replace=True)].sum(0)
                recalls = []
                for tp, tn, fp, fn in c:
                    if tp + fn: recalls.append(tp / (tp + fn))
                    if tn + fp: recalls.append(tn / (tn + fp))
                task_scores.append(finite_mean(recalls))
            scores.append(finite_mean(task_scores))
        r["macro_task_balanced_accuracy_episode_bootstrap_95ci"] = np.quantile(scores, [.025, .975]).tolist()
    return r

def fit_ridge(x, y, valid, xv, yv, vv, device="cpu", alphas=RIDGE_ALPHAS):
    """L2 ridge, loss averaged over valid rows. Alpha selected only on val."""
    mean, scale, counts = target_stats(y, valid)
    usable = counts >= 2
    xt = torch.as_tensor(x, device=device)
    candidates = [{"weight": np.zeros((x.shape[1], y.shape[1]), np.float32),
                   "bias": mean.copy(), "alpha": float(alpha)} for alpha in alphas]
    # Group targets with identical validity so eigendecomposition is shared.
    masks = {}
    for j in np.flatnonzero(usable):
        masks.setdefault(valid[:, j].tobytes(), []).append(int(j))
    for js in masks.values():
        mask = valid[:, js[0]]
        xx = xt[torch.as_tensor(mask, device=device)]
        xm = xx.mean(0)
        xx = xx - xm
        yy = torch.as_tensor((y[mask][:, js] - mean[js]) / scale[js], device=device)
        cov = (xx.T @ xx) / len(xx)
        cov = (cov + cov.T) * .5
        rhs = (xx.T @ yy) / len(xx)
        vals, vecs = torch.linalg.eigh(cov)
        vals = vals.clamp_min(0)
        transformed = vecs.T @ rhs
        for alpha, candidate in zip(alphas, candidates):
            w = vecs @ (transformed / (vals[:, None] + float(alpha)))
            raw_w = w * torch.as_tensor(scale[js], device=device)[None, :]
            b = torch.as_tensor(mean[js], device=device) - xm @ raw_w
            candidate["weight"][:, js] = raw_w.detach().cpu().numpy()
            candidate["bias"][js] = b.detach().cpu().numpy()
    scores = []
    for c in candidates:
        prediction = xv @ c["weight"] + c["bias"]
        errors, n = masked_axis_rmse(prediction, yv, vv & usable[None, :])
        scores.append(finite_mean((errors / scale)[n > 0], float("inf")))
    if not np.isfinite(scores).any():
        raise ValueError("No valid validation labels for ridge probe")
    best = candidates[int(np.argmin(scores))]
    best.update({"target_scale": scale, "target_mean": mean, "usable": usable,
                 "val_nrmse": float(min(scores)), "selection_scores": scores, "valid_train_counts": counts})
    return best

def predict_ridge(model, x):
    return x @ model["weight"] + model["bias"]

def fit_contact(x, y, xv, yv, device="cpu"):
    xt, yt = torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)
    candidates = []
    for alpha in [1e-4, 1e-2, 1.]:
        w = torch.zeros((x.shape[1], y.shape[1]), device=device, requires_grad=True)
        prior = np.clip(y.mean(0), 1e-4, 1-1e-4)
        b = torch.tensor(np.log(prior / (1-prior)), device=device, requires_grad=True)
        count_pos = yt.sum(0)
        pos_weight = (len(y) - count_pos) / count_pos.clamp_min(1)
        pos_weight = pos_weight.clamp(.05, 20)
        optimizer = torch.optim.LBFGS([w,b], lr=1, max_iter=60, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(xt @ w + b, yt, pos_weight=pos_weight)
            loss = loss + alpha * w.square().sum() / (2 * y.shape[1])
            loss.backward()
            return loss
        optimizer.step(closure)
        c = {"weight":w.detach().cpu().numpy(), "bias":b.detach().cpu().numpy(), "alpha":alpha}
        for j in range(y.shape[1]):
            if np.unique(y[:,j]).size == 1:
                c["weight"][:,j] = 0
                c["bias"][j] = math.log(prior[j] / (1-prior[j]))
        probs = predict_contact(c, xv)
        metrics = contact_metrics(probs,yv,np.arange(len(yv)),["val"]*len(yv),bootstrap=0)
        c["val_balanced_accuracy"] = metrics["balanced_accuracy"]
        candidates.append(c)
    return max(candidates, key=lambda c:c["val_balanced_accuracy"])

def predict_contact(model, x):
    logits = np.clip(x @ model["weight"] + model["bias"], -30, 30)
    return 1 / (1 + np.exp(-logits))

def extract_features(model, data_root, split, cache, signature, device, batch_size, workers):
    from dataset import TactileDataset
    from torch.utils.data import DataLoader
    ds = TactileDataset(data_root, split=split, image_size=(192,256))
    rows = manifests(data_root, split)
    if len(ds) != len(rows)*2:
        raise ValueError("Dataset/paired manifest length mismatch")
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    output = cache / (split + "_features.npy")
    marker = cache / (split + "_features.json")
    if output.exists() and marker.exists() and json.loads(marker.read_text()).get("signature") == signature:
        features = np.load(output, mmap_mode="r")
        if len(features) == len(ds):
            return features, rows
    tmp = cache / (split + "_features.partial.npy")
    arr = None
    loader = DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,
                        pin_memory=str(device).startswith("cuda"))
    offset=0
    with torch.inference_mode():
        for batch in loader:
            images=batch["image"].to(device,non_blocking=True)
            ctx=torch.autocast("cuda",dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
            with ctx: z=model.encode(images)
            z=z.float().cpu().numpy()
            if not np.isfinite(z).all(): raise FloatingPointError("Nonfinite encoded features")
            if arr is None: arr=np.lib.format.open_memmap(tmp,mode="w+",dtype=np.float32,shape=(len(ds),z.shape[1]))
            indexes=np.asarray(batch["index"])
            if not np.array_equal(indexes,np.arange(offset,offset+len(z))):
                raise ValueError("Feature extraction index order changed")
            arr[offset:offset+len(z)]=z;offset+=len(z)
    if offset != len(ds): raise ValueError("Incomplete feature extraction")
    arr.flush(); del arr
    os.replace(tmp,output)
    write_json(marker,{"signature":signature,"samples":offset,"split":split})
    return np.load(output,mmap_mode="r"),rows

def labels_for(data_root,split,rows,per_finger=True):
    root=Path(data_root)
    geom=np.asarray(np.load(root/(split+"_geometry.npy")),np.float32)
    raw=np.asarray(np.load(root/(split+"_force.npy")),np.float32)
    fv=np.asarray(np.load(root/(split+"_force_valid.npy")),bool)
    norm=json.loads((root/"normalization.json").read_text())
    force=(raw-np.asarray(norm["force_mean"],np.float32))/np.asarray(norm["force_std"],np.float32)
    if geom.shape != (len(rows),2,7) or raw.shape != (len(rows),2,6) or fv.shape!=raw.shape:
        raise ValueError("Paired label shape mismatch")
    groups=[group_key(r) for r in rows];tasks=[str(r["dataset"]) for r in rows]
    if per_finger:
        geometry=geom[:,:,1:].reshape(-1,6);force=force.reshape(-1,6);fv=fv.reshape(-1,6)
        contacts=geom[:,:,0].reshape(-1,1)
        groups=np.repeat(groups,2);tasks=np.repeat(tasks,2)
    else:
        geometry=geom[:,:,1:].reshape(-1,12);force=force.reshape(-1,12);fv=fv.reshape(-1,12)
        contacts=geom[:,:,0]
    return {"geometry":geometry,"geometry_valid":np.isfinite(geometry),"force":force,
            "force_valid":fv&np.isfinite(force),"contact":contacts.astype(np.float32),
            "groups":np.asarray(groups),"datasets":np.asarray(tasks)}

def feature_health(features):
    mean=np.asarray(features).mean(0)
    x=np.asarray(features)-mean
    std=x.std(0)
    # 4096 deterministically sampled observations suffice for a diagnostic.
    ids=np.linspace(0,len(x)-1,min(len(x),4096),dtype=int)
    s=np.linalg.svd(x[ids],compute_uv=False)
    p=s*s/max(float(np.sum(s*s)),1e-12);p=p[p>0]
    return {"mean_feature_std":float(std.mean()),"near_constant_dimensions":int((std<1e-4).sum()),
            "effective_rank_covariance":float(np.exp(-(p*np.log(p)).sum())),
            "dimension":int(x.shape[1]),"rank_sample_count":len(ids)}

def evaluate_probe_set(name,xs,labels,out,device,bootstrap=300):
    splits=list(xs)
    standardized,mean,std=standardize(xs["train"],*[xs[s] for s in splits if s!="train"])
    xnorm={"train":standardized[0]}
    xnorm.update(dict(zip([s for s in splits if s!="train"],standardized[1:])))
    train,val=labels["train"],labels["val"]
    models={}
    for family in ["geometry","force"]:
        models[family]=fit_ridge(xnorm["train"],train[family],train[family+"_valid"],
                                xnorm["val"],val[family],val[family+"_valid"],device=device)
    models["contact"]=fit_contact(xnorm["train"],train["contact"],xnorm["val"],val["contact"],device=device)
    artifact=Path(out)/"probes"/(name+".pt");artifact.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"models":models,"feature_mean":mean,"feature_std":std,
                "fit_split":"train","hyperparameter_split":"val"},artifact)
    result={"probe_file":str(artifact),"selected_alphas":{k:v["alpha"] for k,v in models.items()},
            "splits":{}}
    for split in splits:
        if split=="train":continue
        target=labels[split];r={}
        for family in ["geometry","force"]:
            pred=predict_ridge(models[family],xnorm[split])
            valid=target[family+"_valid"]&models[family]["usable"][None,:]
            r[family]=regression_metrics(pred,target[family],valid,models[family]["target_scale"],
                                        target["groups"],target["datasets"],bootstrap)
        r["contact"]=contact_metrics(predict_contact(models["contact"],xnorm[split]),target["contact"],
                                     target["groups"],target["datasets"],bootstrap)
        result["splits"][split]=r
    return result

def paired_representation(features,kind):
    if len(features)%2:raise ValueError("Odd single-finger sample count")
    z=np.asarray(features).reshape(-1,2,features.shape[-1])
    if kind=="single_left":return z[:,0]
    if kind=="single_right":return z[:,1]
    if kind=="concat":return np.concatenate([z[:,0],z[:,1]],axis=1)
    if kind=="concat_difference":return np.concatenate([z[:,0],z[:,1],z[:,0]-z[:,1]],axis=1)
    raise ValueError(kind)

def fit_mlp_ablation(xs,labels,out,kind,device,seed=0,epochs=30,patience=5,budget=120000,bootstrap=300):
    """A matched-parameter nonlinear probe; encoder stays frozen."""
    torch.manual_seed(seed)
    splits=list(xs)
    norm,mean,std=standardize(xs["train"],*[xs[s] for s in splits if s!="train"])
    xx={"train":norm[0]};xx.update(zip([s for s in splits if s!="train"],norm[1:]))
    ys={};vs={}
    for s in splits:
        ys[s]=np.concatenate([labels[s]["geometry"],labels[s]["force"]],axis=1)
        vs[s]=np.concatenate([labels[s]["geometry_valid"],labels[s]["force_valid"]],axis=1)
    ymean,yscale,counts=target_stats(ys["train"],vs["train"])
    usable=counts>=2
    output_dim=ys["train"].shape[1]+labels["train"]["contact"].shape[1]
    input_dim=xx["train"].shape[1]
    width=max(4,(budget-output_dim)//(input_dim+output_dim+1))
    model=torch.nn.Sequential(torch.nn.Linear(input_dim,width),torch.nn.GELU(),
                              torch.nn.Linear(width,output_dim)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    xt=torch.as_tensor(xx["train"],device=device)
    yt=torch.as_tensor(np.nan_to_num((ys["train"]-ymean)/yscale, nan=0., posinf=0., neginf=0.),device=device)
    vt=torch.as_tensor(vs["train"]&usable[None,:],device=device)
    ct=torch.as_tensor(labels["train"]["contact"],device=device)
    xv=torch.as_tensor(xx["val"],device=device)
    yv=torch.as_tensor(np.nan_to_num((ys["val"]-ymean)/yscale, nan=0., posinf=0., neginf=0.),device=device)
    vv=torch.as_tensor(vs["val"]&usable[None,:],device=device)
    cv=torch.as_tensor(labels["val"]["contact"],device=device)
    nreg=yt.shape[1]
    best=float("inf");best_state=None;bad=0;trace=[]
    for epoch in range(epochs):
        model.train()
        perm=torch.randperm(len(xt),device=device)
        for ids in perm.split(512):
            pred=model(xt[ids])
            reg=((pred[:,:nreg]-yt[ids]).square()*vt[ids]).sum()/vt[ids].sum().clamp_min(1)
            loss=reg+F.binary_cross_entropy_with_logits(pred[:,nreg:],ct[ids])
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            pred=model(xv)
            score=float((((pred[:,:nreg]-yv).square()*vv).sum()/vv.sum().clamp_min(1)
                         +F.binary_cross_entropy_with_logits(pred[:,nreg:],cv)).cpu())
        trace.append({"epoch":epoch+1,"val_objective":score})
        if score<best-1e-6:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=patience:break
    model.load_state_dict(best_state);model.eval()
    path=Path(out)/"probes"/("mlp_"+kind+".pt");path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":best_state,"input_dim":input_dim,"hidden_dim":width,"output_dim":output_dim,
                "feature_mean":mean,"feature_std":std,"target_mean":ymean,"target_std":yscale,
                "seed":seed,"validation_trace":trace},path)
    result={"parameter_count":sum(p.numel() for p in model.parameters()),"hidden_width":width,
            "target_parameter_budget":budget,"validation_trace":trace,"probe_file":str(path),"splits":{}}
    for split in splits:
        if split=="train":continue
        with torch.no_grad():
            parts=[]
            for start in range(0,len(xx[split]),2048):
                parts.append(model(torch.as_tensor(xx[split][start:start+2048],device=device)).cpu().numpy())
        pp=np.concatenate(parts)
        regression=pp[:,:nreg]*yscale+ymean
        contact=1/(1+np.exp(-np.clip(pp[:,nreg:],-30,30)))
        target=labels[split];ng=target["geometry"].shape[1]
        result["splits"][split]={
            "geometry":regression_metrics(regression[:,:ng],target["geometry"],target["geometry_valid"]&usable[:ng],
                                         yscale[:ng],target["groups"],target["datasets"],bootstrap),
            "force":regression_metrics(regression[:,ng:],target["force"],target["force_valid"]&usable[ng:],
                                      yscale[ng:],target["groups"],target["datasets"],bootstrap),
            "contact":contact_metrics(contact,target["contact"],target["groups"],target["datasets"],bootstrap)}
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--data-root",required=True)
    parser.add_argument("--out",required=True);parser.add_argument("--device",default="cuda")
    parser.add_argument("--batch-size",type=int,default=128);parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--probe-device",default=None);parser.add_argument("--bootstrap",type=int,default=300)
    parser.add_argument("--mlp-epochs",type=int,default=30);parser.add_argument("--skip-mlp",action="store_true")
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selection-only",action="store_true");mode.add_argument("--final",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    out=Path(args.out).resolve()
    task_root=next((p for p in Path(__file__).resolve().parents if p.name == "tactile_encoder" and (p/"provenance"/"source.json").is_file()), None)
    if task_root is None or task_root not in out.parents:
        raise ValueError("Evaluation output must remain in the task directory")
    out.mkdir(parents=True,exist_ok=True)
    from model import load_model
    checkpoint=Path(args.checkpoint)
    checkpoint_hash=digest_file(checkpoint)
    checkpoint_data=torch.load(checkpoint,map_location="cpu",weights_only=False)
    train_args=checkpoint_data.get("args",{})
    if not isinstance(train_args,dict):train_args=vars(train_args)
    model=load_model(checkpoint,device=args.device);model.eval()
    splits=["train","val"] if args.selection_only else ["train","val","test","ood_funnel","ood_cups"]
    rows={s:manifests(args.data_root,s) for s in splits}
    assert_no_episode_overlap(rows)
    data_provenance={"normalization_sha256":digest_file(Path(args.data_root)/"normalization.json"),
                     "manifest_sha256":{s:digest_file(Path(args.data_root)/(s+"_manifest.jsonl")) for s in splits}}
    data_signature=hashlib.sha256(json.dumps(data_provenance,sort_keys=True).encode()).hexdigest()
    features={}
    for s in splits:
        signature=hashlib.sha256((checkpoint_hash+digest_file(Path(args.data_root)/"normalization.json")
                    +digest_file(Path(args.data_root)/(s+"_manifest.jsonl"))+"192x256-mean-patch-v1").encode()).hexdigest()
        features[s],_=extract_features(model,args.data_root,s,out/"features",signature,
                                       args.device,args.batch_size,args.workers)
        print(json.dumps({"event":"features_ready","split":s,"samples":len(features[s])}),flush=True)
    del model,checkpoint_data
    if str(args.device).startswith("cuda"):torch.cuda.empty_cache()
    probe_device=args.probe_device or args.device
    labels={s:labels_for(args.data_root,s,rows[s],True) for s in splits}
    quality=evaluate_probe_set("per_finger",features,labels,out,probe_device,args.bootstrap)
    # Side-specific scores use exactly the same fitted quality heads.
    heads=torch.load(out/"probes"/"per_finger.pt",map_location="cpu",weights_only=False)
    side_quality={}
    for finger,label in [(0,"left"),(1,"right")]:
        side_quality[label]={}
        for split in splits:
            if split=="train":continue
            idx=np.arange(finger,len(features[split]),2)
            x=(features[split][idx]-heads["feature_mean"])/heads["feature_std"]
            target=labels[split];scores={}
            for family in ["geometry","force"]:
                fitted=heads["models"][family]
                scores[family]=regression_metrics(predict_ridge(fitted,x),target[family][idx],
                    target[family+"_valid"][idx]&fitted["usable"],fitted["target_scale"],
                    target["groups"][idx],target["datasets"][idx],args.bootstrap)
            scores["contact"]=contact_metrics(predict_contact(heads["models"]["contact"],x),
                target["contact"][idx],target["groups"][idx],target["datasets"][idx],args.bootstrap)
            side_quality[label][split]=scores
    result={"schema_version":1,"mode":"selection_only" if args.selection_only else "final",
            "checkpoint":str(checkpoint),"checkpoint_sha256":checkpoint_hash,"data_signature":data_signature,"data_provenance":data_provenance,
            "lambda_force":float(train_args.get("lambda_force",0.)),
            "seed":int(train_args.get("seed",0)),"evaluation_time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
            "split_pair_counts":{s:len(rows[s]) for s in splits},
            "encoder_feature_health":feature_health(features["train"]),
            "per_finger":quality,"per_finger_sides":side_quality,
            "interpretation":["Force targets are X5 SDK estimates, not independent physical force ground truth.",
                              "Contact geometry is a fixed DD-derived proxy, not manually verified truth.",
                              "Single/paired representations are evaluated on identical paired timestamps and target sets.",
                              "Episode bootstrap intervals do not treat individual frames as independent.",
                              "Hyperparameters and checkpoint/lambda selection never use final test or OOD."]}
    write_json(out/"evaluation.partial.json",result)
    if args.final:
        pair_labels={s:labels_for(args.data_root,s,rows[s],False) for s in splits}
        result["paired_same_target"]={}
        for kind in ["single_left","single_right","concat"]:
            xs={s:paired_representation(features[s],kind) for s in splits}
            result["paired_same_target"][kind]=evaluate_probe_set(kind,xs,pair_labels,out,probe_device,args.bootstrap)
            print(json.dumps({"event":"paired_probe_ready","representation":kind}),flush=True)
        # concat+diff has exactly the same linear hypothesis class. Reuse an
        # equivalent solution with W_difference=0; do not mislabel a new signal.
        result["paired_same_target"]["concat_difference"]={
            "equivalent_to":"concat","fit":"analytic_equivalent_linear_head_Wdiff_zero",
            "dimension":3*features["train"].shape[1],
            "splits":result["paired_same_target"]["concat"]["splits"],
            "note":"Explicit difference adds no information or linear expressiveness; separate L2 fits would change regularization."}
        result["mlp_difference_ablation"]={}
        if not args.skip_mlp:
            for kind in ["concat","concat_difference"]:
                xs={s:paired_representation(features[s],kind) for s in splits}
                result["mlp_difference_ablation"][kind]=fit_mlp_ablation(
                    xs,pair_labels,out,kind,probe_device,seed=result["seed"],
                    epochs=args.mlp_epochs,bootstrap=args.bootstrap)
        else:result["mlp_difference_ablation"]["status"]="explicitly_skipped"
    write_json(out/"evaluation.json",result)
    write_json(out/"EVALUATION_COMPLETED.json",{"checkpoint_sha256":checkpoint_hash,"mode":result["mode"]})
    print(json.dumps({"event":"evaluation_completed","output":str(out/"evaluation.json"),
                      "lambda_force":result["lambda_force"],"seed":result["seed"]}),flush=True)

if __name__=="__main__":main()
