"""Choose the auxiliary force weight using development results only."""
from pathlib import Path
import argparse, hashlib, json, datetime
import numpy as np

ROOT=next((p for p in Path(__file__).resolve().parents if p.name == "tactile_encoder" and (p/"provenance"/"source.json").is_file()), None)
if ROOT is None:raise RuntimeError("Cannot find authorized tactile_encoder project root")

def select(results,geometry_tolerance=.05,contact_tolerance=.02,min_force_improvement=.01):
    groups={}
    data_signatures=set()
    for path in results:
        row=json.loads(Path(path).read_text())
        if row["mode"]!="selection_only":
            raise ValueError("Lambda selection accepts selection_only results; final test results are forbidden")
        if set(row["split_pair_counts"])!={"train","val"}:
            raise ValueError("Selection result contains a held-out split")
        if "data_signature" not in row:raise ValueError("Missing immutable data signature")
        data_signatures.add(row["data_signature"])
        score=row["per_finger"]["splits"]["val"]
        values={"geometry":score["geometry"]["macro_task_nrmse"],
                "force":score["force"]["macro_task_nrmse"],
                "contact":score["contact"]["macro_task_balanced_accuracy"]}
        if any(v is None or not np.isfinite(v) for v in values.values()):
            raise ValueError("Nonfinite/missing validation metric")
        groups.setdefault(float(row["lambda_force"]),[]).append({"seed":row["seed"],"file":str(path),**values})
    if len(data_signatures)!=1:raise ValueError("Candidate runs used different data manifests or normalization")
    if 0. not in groups or len(groups)<2:raise ValueError("Need MAE baseline and at least one force candidate")
    aggregate={lam:{key:float(np.mean([r[key] for r in rows])) for key in ["geometry","force","contact"]}
               for lam,rows in groups.items()}
    baseline=aggregate[0.]
    scores=[]
    for lam in sorted(aggregate):
        r=aggregate[lam]
        geometry_ok=r["geometry"]<=baseline["geometry"]*(1+geometry_tolerance)+1e-12
        contact_ok=r["contact"]>=baseline["contact"]-contact_tolerance
        force_ok=r["force"]<baseline["force"]*(1-min_force_improvement)
        scores.append({"lambda_force":lam,**r,"geometry_ok":bool(geometry_ok),"contact_ok":bool(contact_ok),
                       "force_improved":bool(force_ok),"eligible":bool(lam>0 and geometry_ok and contact_ok and force_ok)})
    accepted=[s for s in scores if s["eligible"]]
    if accepted:
        chosen=min(accepted,key=lambda s:s["lambda_force"]);accepted_as_improvement=True
        reason="Smallest weight meeting predeclared validation geometry/contact tolerances and force improvement."
    else:
        # A positive candidate is still needed for an honest formal A/B result.
        chosen=min([s for s in scores if s["lambda_force"]>0],key=lambda s:(s["force"],s["lambda_force"]))
        accepted_as_improvement=False
        reason="No weight met all validation criteria. MAE remains recommended; positive candidate is retained solely for the planned formal comparison."
    return {"schema_version":1,"selected_lambda":chosen["lambda_force"],
            "accepted_as_improvement":accepted_as_improvement,
            "recommended_lambda":chosen["lambda_force"] if accepted_as_improvement else 0.,
            "reason":reason,"selection_split":"val","test_used":False,
            "thresholds":{"geometry_relative_tolerance":geometry_tolerance,"contact_absolute_tolerance":contact_tolerance,
                          "force_relative_improvement":min_force_improvement},
            "validation_scores":scores,"runs":{str(k):v for k,v in groups.items()},
            "screening_uncertainty":"Screening is not final evidence; the formal comparison uses three training seeds."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",nargs="+",required=True);p.add_argument("--out",required=True)
    args=p.parse_args()
    out=Path(args.out).resolve()
    if ROOT not in out.parents:raise ValueError("Output must remain in this task directory")
    result=select(args.results)
    result["created_at_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["input_sha256"]={str(f):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in args.results}
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+".tmp")
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2));tmp.replace(out)
    print(json.dumps({"event":"lambda_selected","lambda":result["selected_lambda"],
                      "accepted":result["accepted_as_improvement"],"output":str(out)}),flush=True)
if __name__=="__main__":main()
