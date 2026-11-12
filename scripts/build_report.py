"""Build the final Chinese report strictly from completed evaluation artifacts."""
from pathlib import Path
import argparse, datetime, hashlib, html, json, math
import numpy as np

ROOT=next((p for p in Path(__file__).resolve().parents if p.name == "tactile_encoder" and (p/"provenance"/"source.json").is_file()), None)
if ROOT is None:raise RuntimeError("Cannot find authorized tactile_encoder project root")
def get_metric(row,kind,split,family):
    score=row["paired_same_target"][kind]["splits"][split][family]
    return score["macro_task_balanced_accuracy"] if family=="contact" else score["macro_task_nrmse"]

def aggregate(rows):
    groups={}
    for row in rows:groups.setdefault(float(row["lambda_force"]),[]).append(row)
    summary=[]
    for lam,runs in sorted(groups.items()):
        for split in ["val","test","ood_funnel","ood_cups"]:
            for kind in ["single_left","single_right","concat","concat_difference"]:
                values={}
                for family in ["geometry","force","contact"]:
                    v=[get_metric(r,kind,split,family) for r in runs]
                    v=[float(x) for x in v if x is not None and np.isfinite(x)]
                    values[family]={"mean":float(np.mean(v)) if v else None,
                                    "std_across_training_seeds":float(np.std(v,ddof=1)) if len(v)>1 else None,
                                    "values":v}
                summary.append({"lambda_force":lam,"split":split,"representation":kind,
                                "seed_count":len(runs),"metrics":values})
    return groups,summary

def number(v):
    return "—" if v is None else f"{v:.4f}"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",nargs="+",required=True);p.add_argument("--selection",required=True)
    p.add_argument("--out-dir","--out",dest="out_dir",required=True);p.add_argument("--allow-partial",action="store_true")
    args=p.parse_args()
    out=Path(args.out_dir).resolve()
    if ROOT not in out.parents:raise ValueError("Report output must remain in task directory")
    rows=[]
    seen_runs=set()
    for f in args.results:
        row=json.loads(Path(f).read_text())
        if row.get("mode")!="final":raise ValueError("Final report cannot substitute screening results")
        if not (Path(f).parent/"EVALUATION_COMPLETED.json").exists():
            raise ValueError("Evaluation completion marker missing")
        identity=(float(row["lambda_force"]),int(row["seed"]))
        if identity in seen_runs:raise ValueError("Duplicate lambda/seed result would bias aggregation")
        seen_runs.add(identity)
        marker=json.loads((Path(f).parent/"EVALUATION_COMPLETED.json").read_text())
        if marker["checkpoint_sha256"]!=row["checkpoint_sha256"]:raise ValueError("Evaluation marker/checkpoint mismatch")
        rows.append(row)
    if len({r.get("data_signature") for r in rows})!=1 or not all(r.get("data_signature") for r in rows):
        raise ValueError("Final runs used missing/different data signatures")
    if len({json.dumps(r["split_pair_counts"],sort_keys=True) for r in rows})!=1:
        raise ValueError("Experiment sample counts differ")
    groups,summary=aggregate(rows)
    selection=json.loads(Path(args.selection).read_text())
    expected={0.,float(selection["selected_lambda"])}
    counts={lam:len({r["seed"] for r in runs}) for lam,runs in groups.items()}
    seed_sets={lam:{r["seed"] for r in runs} for lam,runs in groups.items()}
    complete=(set(groups)==expected and all(counts.get(lam,0)>=3 for lam in expected)
              and len({tuple(sorted(v)) for v in seed_sets.values()})==1)
    if not complete and not args.allow_partial:raise ValueError("Need completed baseline and force runs for at least three distinct seeds each")
    out.mkdir(parents=True,exist_ok=True)
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")
    overview={"status":"complete" if complete else "partial","generated_at_Asia_Shanghai":now,
              "lambda_selection":selection,"seed_counts":{str(k):v for k,v in counts.items()},
              "aggregated_results":summary,
              "sources":[{"file":str(f),"sha256":hashlib.sha256(Path(f).read_bytes()).hexdigest()} for f in args.results],
              "limits":["Force labels are X5 SDK estimates, not an independent force measurement.",
                        "Geometry labels are fixed DD-derived proxies.",
                        "The pair comparison measures observation availability, not different encoder weights.",
                        "Concat+signed-difference has no additional information for a linear model.",
                        "Seed standard deviations and episode bootstrap intervals answer different uncertainty questions.",
                        "Held-out transfer to two tasks does not prove unrestricted cross-task or cross-sensor generalization."]}
    (out/"result_summary.json").write_text(json.dumps(overview,ensure_ascii=False,indent=2,allow_nan=False))
    status="已完成" if complete else "部分完成"
    lines=[f"# 触觉编码器对比实验结果（{status}）","",f"生成时间：{now}。","",
           "实验比较 MAE 与 MAE＋SDK 六维力回归。两组使用相同数据清单、原始 CLIP 初始化、训练轮数和成对随机种子；每个编码器冻结后再比较单指与同手双指表示。",
           "",f"开发集选择的力权重为 **{selection['selected_lambda']}**。"+("该候选满足预先设定的开发集准则。" if selection["accepted_as_improvement"] else "没有候选同时满足全部开发集准则；当前正权重只用于完成正式对照，不能宣称筛选成功。"),
           "", "## 数据与评估约定","",
           "- 训练样本预算为 25,000 个同手双指时刻，展开为 50,000 张单指图；最终数量以每份 evaluation.json 的 split_pair_counts 为准。",
           "- 先按 episode/源轨迹划分，归一化仅使用训练集；开发集用于探针正则和力权重选择。",
           "- 漏斗和三杯任务仅用于最终任务外评估，不用于选择权重或探针超参数。",
           "- 单指左、单指右、双指表示均在同一批时刻预测同一组双侧目标；下表分数才可直接比较。",
           "- 几何与力为各任务等权平均的标准化 RMSE，越低越好；接触为任务平均平衡准确率，越高越好。",
           "- 线性 concat＋difference 行使用与 concat 完全等价的线性解，显式差值的额外效果由参数量匹配的非线性 MLP 单独检验。",
           "", "## 主结果",""]
    for split,label in [("test","训练任务的留出 episode"),("ood_funnel","任务外：漏斗"),("ood_cups","任务外：三杯")]:
        lines.extend([f"### {label}","",
                      "| 预训练 | 表示 | 种子数 | 几何 nRMSE | SDK 力 nRMSE | 接触平衡准确率 |",
                      "|---|---|---:|---:|---:|---:|"])
        for r in summary:
            if r["split"]!=split:continue
            name="MAE" if r["lambda_force"]==0 else f"MAE＋力 λ={r['lambda_force']}"
            values=[]
            for family in ["geometry","force","contact"]:
                m=r["metrics"][family]
                values.append(number(m["mean"])+(f" ± {m['std_across_training_seeds']:.4f}" if m["std_across_training_seeds"] is not None else ""))
            lines.append(f"| {name} | {r['representation']} | {r['seed_count']} | "+" | ".join(values)+" |")
        lines.append("")
    lines.extend(["## 显式左右差值的非线性对照","",
                  "以下两种表示使用近似相同参数总量的小 MLP，编码器保持冻结，验证集决定早停。两者输入维度不同，因此隐藏宽度经过调整。"])
    for lam,runs in sorted(groups.items()):
        lines.extend(["",f"λ={lam}：","",
                      "| 表示 | 参数量范围 | 测试几何 nRMSE | 测试 SDK 力 nRMSE | 测试接触平衡准确率 |",
                      "|---|---:|---:|---:|---:|"])
        for kind in ["concat","concat_difference"]:
            entries=[r.get("mlp_difference_ablation",{}).get(kind) for r in runs]
            entries=[e for e in entries if e]
            if not entries:
                lines.append(f"| {kind} | 未完成 | — | — | — |");continue
            ns=[e["parameter_count"] for e in entries]
            scores=[]
            for family in ["geometry","force","contact"]:
                key="macro_task_balanced_accuracy" if family=="contact" else "macro_task_nrmse"
                values=[e["splits"]["test"][family][key] for e in entries]
                scores.append(number(float(np.mean(values))))
            lines.append(f"| {kind} | {min(ns)}–{max(ns)} | "+" | ".join(scores)+" |")
    lines.extend(["", "## 编码器本身的增益",""])
    for split,label in [("test","训练任务留出"),("ood_funnel","漏斗任务外"),("ood_cups","杯子任务外")]:
        table={}
        for lam,runs in groups.items():
            table[lam]={}
            for family in ["geometry","force","contact"]:
                key="macro_task_balanced_accuracy" if family=="contact" else "macro_task_nrmse"
                vals=[r["per_finger"]["splits"][split][family][key] for r in runs]
                vals=[v for v in vals if v is not None]
                table[lam][family]=float(np.mean(vals)) if vals else None
        if 0. in table and float(selection["selected_lambda"]) in table:
            baseline=table[0.];aux=table[float(selection["selected_lambda"])]
            changes=[]
            for family,name in [("geometry","几何"),("force","SDK 力")]:
                if baseline[family] is not None and aux[family] is not None:
                    relative=(aux[family]/max(baseline[family],1e-12)-1)*100
                    changes.append(f"{name} nRMSE {number(baseline[family])} → {number(aux[family])}（{relative:+.2f}%）")
            changes.append(f"接触平衡准确率 {number(baseline['contact'])} → {number(aux['contact'])}")
            lines.append(f"- {label}："+"；".join(changes)+"。")
    lines.append("")
    lines.extend(["", "## 结果解读与限制","",
                  "是否受益应分别检查：力监督是否改善每指几何与力探针；双指是否改善相同双侧目标；显式差值是否在等参数 MLP 中稳定受益。不能仅凭 SDK 力误差降低就宣称触觉编码器整体更好。",
                  "",
                  "六维标签来自 X5 SDK 的触觉估计，几何标签来自固定 DD 规则；这些结果不是独立物理测力精度或人工标注接触真值。三个训练种子的标准差反映训练波动；每份评估 JSON 另有以完整 episode 为单位的 bootstrap 区间。",
                  "",
                  "若结果跨种子不稳定或几何指标退化，应如实保留 MAE 基线。漏斗/杯子转移只能支持本次评估范围内的跨任务证据，不能保证任意任务、传感器或盖板都直接泛化。",
                  "", "## 产物","",
                  "- result_summary.json：聚合结果、选择准则和输入文件校验值。",
                  "- 各 runs 目录的 encoder.pt / checkpoint：编码器及训练配置。",
                  "- 各评估目录的 evaluation.json：完整分任务、左右指、单/双指和置信区间。",
                  "- 各评估目录的 probes：冻结特征上的线性/MLP 探针及训练统计。",
                  "- 所有原始训练数据与原编码器目录保持只读。",""])
    markdown="\n".join(lines)
    (out/"report.md").write_text(markdown)
    # Standalone readable HTML, no remote resources.
    html_body="<h1>触觉编码器对比实验结果</h1><p>状态："+html.escape(status)+"</p><pre>"+html.escape(markdown)+"</pre>"
    (out/"report.html").write_text("<!doctype html><html lang='zh'><meta charset='utf-8'><title>触觉编码器对比报告</title><style>body{max-width:1200px;margin:32px auto;font:16px/1.7 sans-serif;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:20px}img{max-width:100%}</style>"+html_body+"</html>")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        kinds=["single_left","single_right","concat"]
        for ax,family in zip(axes,["geometry","force","contact"]):
            for i,(lam,runs) in enumerate(sorted(groups.items())):
                selected=[next(r for r in summary if r["lambda_force"]==lam and r["split"]=="test" and r["representation"]==kind) for kind in kinds]
                means=[r["metrics"][family]["mean"] for r in selected]
                errors=[r["metrics"][family]["std_across_training_seeds"] or 0 for r in selected]
                ax.bar(np.arange(3)+(i-.5)*.36,means,yerr=errors,width=.34,label=f"lambda={lam}")
            ax.set_xticks(np.arange(3),["Single L","Single R","Dual concat"])
            ax.set_title(family+" (test, mean +/- seed SD)");ax.legend()
        fig.tight_layout();fig.savefig(out/"test_comparison.png",dpi=160);plt.close(fig)
    except ImportError:
        pass
    print(json.dumps({"event":"report_written","status":overview["status"],"path":str(out/"report.html")}),flush=True)
if __name__=="__main__":main()
