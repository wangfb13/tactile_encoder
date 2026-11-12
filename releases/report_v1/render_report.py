"""Render immutable experiment artifacts as a standalone, evidence-based HTML report.

This script does not train, evaluate, choose a weight, edit existing artifacts, or
query cloud credentials. It only reads project-local evidence and writes a new
reviewed_report.html plus a rendering receipt.
"""
from __future__ import annotations
import argparse, base64, datetime, hashlib, html, json, math, os
from pathlib import Path

REPRESENTATIONS = {"single_left":"单指 L", "single_right":"单指 R",
                   "concat":"双指拼接", "concat_difference":"双指拼接＋差值"}
SPLITS = [("test","训练任务的留出 episode"),
          ("ood_funnel","任务外：漏斗"),("ood_cups","任务外：三杯")]
TASK_NAMES = {"green_normal":"绿瓶 normal","green_guo":"绿瓶 guo","vase":"花瓶",
              "wire":"整理线缆","pinggai":"瓶盖","cable":"拾取线缆","wipe":"擦板",
              "funnel":"漏斗","cups":"三杯"}

def esc(value):
    return html.escape(str(value),quote=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def inside(path,root):
    try:Path(path).resolve().relative_to(Path(root).resolve());return True
    except ValueError:return False

def load_json(path,default=None):
    p=Path(path)
    return json.loads(p.read_text()) if p.is_file() else default

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def average(values):
    values=[float(v) for v in values if finite(v)]
    return sum(values)/len(values) if values else None

def sample_sd(values):
    values=[float(v) for v in values if finite(v)]
    if len(values)<2:return None
    m=average(values)
    return math.sqrt(sum((x-m)**2 for x in values)/(len(values)-1))

def number(value,digits=4):
    return f"{value:.{digits}f}" if finite(value) else "—"

def mean_sd(values):
    m,s=average(values),sample_sd(values)
    return number(m)+(f" ± {s:.4f}" if s is not None else "")

def percentage_change(after,before):
    return 100*(after/before-1) if finite(after) and finite(before) and before>0 else None

def table(headers,rows,caption=None):
    cap=f"<caption>{esc(caption)}</caption>" if caption else ""
    head="".join("<th scope='col'>"+esc(h)+"</th>" for h in headers)
    body="".join("<tr>"+"".join("<td>"+esc(c)+"</td>" for c in row)+"</tr>" for row in rows)
    return "<div class='table-wrap'><table>"+cap+"<thead><tr>"+head+"</tr></thead><tbody>"+body+"</tbody></table></div>"

def badge(ok):
    return "<span class='badge "+("pass" if ok else "pending")+"'>"+("通过" if ok else "未完整核验")+"</span>"

def metric(row,section,split,family,representation=None):
    obj=row.get(section,{})
    if representation is not None:obj=obj.get(representation,{})
    obj=obj.get("splits",{}).get(split,{}).get(family,{})
    key="macro_task_balanced_accuracy" if family=="contact" else "macro_task_nrmse"
    return obj.get(key)

def load_sources(summary,root):
    rows=[];evidence=[]
    seen=set()
    for entry in summary.get("sources",[]):
        path=Path(entry["file"])
        if not path.is_absolute():path=root/path
        path=path.resolve()
        if not inside(path,root):raise ValueError("Evaluation source escapes the project directory")
        if not path.is_file():raise FileNotFoundError("Missing evaluation source: "+str(path))
        actual=sha(path)
        if entry.get("sha256")!=actual:raise ValueError("Evaluation source hash differs from result summary: "+str(path))
        row=load_json(path)
        identity=(float(row["lambda_force"]),int(row["seed"]))
        if identity in seen:raise ValueError("Duplicate lambda/seed source")
        seen.add(identity)
        if row.get("mode")!="final":raise ValueError("A screening artifact cannot be rendered as a final result")
        marker=load_json(path.parent/"EVALUATION_COMPLETED.json",{})
        marker_ok=(marker.get("checkpoint_sha256")==row.get("checkpoint_sha256")
                   and marker.get("mode")=="final")
        run_dir=path.parent.parent
        job_marker=load_json(run_dir/"JOB_COMPLETE.json",{})
        trained=load_json(run_dir/"COMPLETED.json",{})
        training_summary=load_json(run_dir/"summary.json",{})
        job_ok=(job_marker.get("phase")=="final"
                and float(job_marker.get("lambda_force",-1))==identity[0]
                and int(job_marker.get("seed",-1))==identity[1]
                and int(job_marker.get("epochs",0))==100
                and int(job_marker.get("max_steps",-1))==0)
        train_ok=(bool(trained) and training_summary.get("status")=="completed"
                  and int(training_summary.get("epochs",0))==100
                  and float(training_summary.get("lambda_force",-1))==identity[0]
                  and int(training_summary.get("seed",-1))==identity[1])
        evidence.append({"file":str(path),"sha256":actual,"lambda_force":identity[0],"seed":identity[1],
                         "evaluation_marker_ok":marker_ok,"job_marker_ok":job_ok,
                         "training_complete_ok":train_ok,"run_directory":str(run_dir)})
        rows.append(row)
    return rows,evidence

def grouped(rows):
    groups={}
    for row in rows:groups.setdefault(float(row["lambda_force"]),[]).append(row)
    for runs in groups.values():runs.sort(key=lambda r:int(r["seed"]))
    return groups

def completion_evidence(summary,rows,evidence,root):
    selection=summary.get("lambda_selection",{})
    selected=selection.get("selected_lambda")
    groups=grouped(rows)
    expected={0.,float(selected)} if finite(selected) and selected>0 else {0.}
    seed_sets={lam:{int(r["seed"]) for r in runs} for lam,runs in groups.items()}
    three=(len(expected)==2 and set(groups)==expected
           and all(seed_sets.get(lam)=={0,1,2} for lam in expected))
    all_markers=(len(evidence)==6 and all(e["evaluation_marker_ok"] and e["job_marker_ok"]
                    and e["training_complete_ok"] for e in evidence))
    signatures={r.get("data_signature") for r in rows}
    same_data=(bool(rows) and None not in signatures and len(signatures)==1)
    qa=load_json(root/"data/prepared/QA_READY.json",{})
    prep=load_json(root/"data/prepared/summary.json",{})
    expected_counts={s:d.get("pairs") for s,d in prep.get("splits",{}).items()}
    matching_counts=bool(rows) and bool(expected_counts) and all(r.get("split_pair_counts")==expected_counts for r in rows)
    prep_ok=(qa.get("ready") is True and qa.get("no_episode_overlap") is True and matching_counts)
    selection_ok=(selection.get("selection_split")=="val" and selection.get("test_used") is False
                  and finite(selected) and selected>0)
    full=(summary.get("status")=="complete" and three and all_markers and same_data and prep_ok and selection_ok)
    return {"complete":full,"summary_status":summary.get("status"),"three_matched_seeds":three,
            "six_completed_jobs_and_evaluations":all_markers,"identical_data_signature":same_data,
            "prepared_data_qa_and_counts":prep_ok,"validation_only_weight_selection":selection_ok,
            "seed_sets":{str(k):sorted(v) for k,v in seed_sets.items()},
            "data_summary":prep,"data_qa":qa}

def findings(groups,selection,complete):
    selected=selection.get("selected_lambda")
    if not finite(selected) or 0. not in groups or float(selected) not in groups:
        return ["当前材料未包含两类编码器的完整对照，不能判断辅助力监督是否有效。"]
    selected=float(selected)
    base,aux=groups[0.],groups[selected]
    values={}
    for label,runs in [("mae",base),("force",aux)]:
        values[label]={family:average([metric(r,"per_finger","test",family) for r in runs])
                       for family in ["geometry","force","contact"]}
    b,a=values["mae"],values["force"]
    result=[]
    for family,name in [("geometry","接触几何"),("force","SDK 六维力")]:
        delta=percentage_change(a[family],b[family])
        if delta is not None:
            direction="降低" if delta<0 else ("升高" if delta>0 else "持平")
            result.append(f"冻结单指特征的{name} nRMSE：MAE {number(b[family])}，MAE＋力 {number(a[family])}；均值{direction} {abs(delta):.2f}%。")
    if finite(b["contact"]) and finite(a["contact"]):
        result.append(f"接触平衡准确率：{number(b['contact'])} → {number(a['contact'])}，变化 {(a['contact']-b['contact'])*100:+.2f} 个百分点。")
    if not complete:
        result.insert(0,"本页未满足全部完成判据，以下是现有材料的描述性统计，不作最终方法优劣结论。")
    elif all(finite(v) for v in list(b.values())+list(a.values())):
        geometry_ok=a["geometry"]<=b["geometry"]*1.05
        force_better=a["force"]<b["force"]*.99
        contact_ok=a["contact"]>=b["contact"]-.02
        if geometry_ok and force_better and contact_ok:
            result.append("在本次固定探针和留出 episode 上，辅助监督的力误差均值改善，几何与接触指标没有超过预定工程容忍线；这支持其在本次评估范围内的用途，不能直接视为独立测力精度或统计显著性证明。")
        elif not geometry_ok:
            result.append("辅助监督的几何误差超过相对基线 5% 的工程容忍线；不能仅凭力误差变化宣称通用触觉表示更好，位置敏感用途应优先保留 MAE 基线。")
        elif not force_better:
            result.append("没有观察到至少 1% 的 SDK 力误差均值改善；本次结果不足以支持增加力监督带来明确收益。")
        else:
            result.append("接触识别均值低于预定容忍线，辅助监督没有同时满足全部指标要求。")
    for lam,runs in sorted(groups.items()):
        singles=average(paired_values(runs,"test","geometry","single_mean"))
        dual=average(paired_values(runs,"test","geometry","concat"))
        change=percentage_change(dual,singles)
        if change is not None:
            result.append(f"λ={lam} 的相同双侧目标对照：单指 L/R 平均几何 nRMSE 为 {number(singles)}，双指拼接为 {number(dual)}（变化 {change:+.2f}%）；这里的单指平均先在每个训练种子内平均两侧成绩，再跨种子汇总，不挑选更好的一侧。")
    result.append("上述百分比来自训练种子的均值比较；未执行显著性检验，不能把均值变化写成确定的因果收益。")
    return result

def source_table(evidence):
    rows=[]
    for e in sorted(evidence,key=lambda x:(x["lambda_force"],x["seed"])):
        rows.append(["MAE" if e["lambda_force"]==0 else f"MAE＋力 λ={e['lambda_force']}",
                     e["seed"],Path(e["run_directory"]).name,
                     "✓" if e["training_complete_ok"] else "缺失/未匹配",
                     "✓" if e["job_marker_ok"] else "缺失/未匹配",
                     "✓" if e["evaluation_marker_ok"] else "缺失/未匹配",
                     e["sha256"][:16]+"…"])
    return table(["方法","种子","运行目录","训练标记","完整作业标记","评估标记","评估文件 SHA-256"],rows)

def paired_values(runs,split,family,kind):
    if kind!="single_mean":
        return [metric(r,"paired_same_target",split,family,kind) for r in runs]
    values=[]
    for row in runs:
        left=metric(row,"paired_same_target",split,family,"single_left")
        right=metric(row,"paired_same_target",split,family,"single_right")
        values.append((left+right)/2 if finite(left) and finite(right) else None)
    return values

def main_result_tables(groups):
    sections=[]
    for split,title in SPLITS:
        rows=[]
        for lam,runs in sorted(groups.items()):
            views=[("single_left","单指 L"),("single_right","单指 R"),
                   ("single_mean","单指 L/R 平均"),("concat","双指拼接"),
                   ("concat_difference","双指拼接＋差值")]
            for kind,label in views:
                vals=[mean_sd(paired_values(runs,split,f,kind))
                      for f in ["geometry","force","contact"]]
                rows.append(["MAE" if lam==0 else f"MAE＋力 λ={lam}",label,len(runs),*vals])
        sections.append("<h3>"+esc(title)+"</h3>"+table(
            ["预训练","观测表示","训练种子数","几何 nRMSE ↓","SDK 力 nRMSE ↓","接触平衡准确率 ↑"],rows))
    return "".join(sections)

def per_finger_table(groups):
    rows=[]
    for split,title in SPLITS:
        for lam,runs in sorted(groups.items()):
            vals=[mean_sd([metric(r,"per_finger",split,f) for r in runs])
                  for f in ["geometry","force","contact"]]
            rows.append([title,"MAE" if lam==0 else f"MAE＋力 λ={lam}",len(runs),*vals])
    return table(["评估集合","编码器","种子数","几何 nRMSE ↓","SDK 力 nRMSE ↓","接触平衡准确率 ↑"],rows)

def mlp_table(groups):
    rows=[];notes=[]
    for lam,runs in sorted(groups.items()):
        scores={}
        for kind in ["concat","concat_difference"]:
            entries=[r.get("mlp_difference_ablation",{}).get(kind) for r in runs]
            entries=[e for e in entries if isinstance(e,dict)]
            if not entries:
                rows.append(["MAE" if lam==0 else f"MAE＋力 λ={lam}",REPRESENTATIONS[kind],"未完成","—","—","—"])
                continue
            params=[int(e["parameter_count"]) for e in entries]
            values={}
            for family in ["geometry","force","contact"]:
                key="macro_task_balanced_accuracy" if family=="contact" else "macro_task_nrmse"
                values[family]=[e.get("splits",{}).get("test",{}).get(family,{}).get(key) for e in entries]
            scores[kind]={k:average(v) for k,v in values.items()}
            rows.append(["MAE" if lam==0 else f"MAE＋力 λ={lam}",REPRESENTATIONS[kind],
                         str(params[0]) if min(params)==max(params) else f"{min(params)}–{max(params)}",
                         *[mean_sd(values[f]) for f in ["geometry","force","contact"]]])
        if set(scores)=={"concat","concat_difference"}:
            delta=percentage_change(scores["concat_difference"]["geometry"],scores["concat"]["geometry"])
            if delta is not None:
                notes.append(f"λ={lam}：增加显式差值后，等参数 MLP 的测试几何 nRMSE 均值变化 {delta:+.2f}%；这描述输入参数化的效果，没有增加新的观测信息。")
    return table(["编码器","MLP 输入","参数量","几何 nRMSE ↓","SDK 力 nRMSE ↓","接触平衡准确率 ↑"],rows)+"".join("<p>"+esc(n)+"</p>" for n in notes)

def data_section(completion,rows):
    prep=completion["data_summary"]
    train=prep.get("splits",{}).get("train",{})
    data_rows=[]
    total=train.get("pairs",0)
    for task,n in sorted(train.get("datasets",{}).items()):
        data_rows.append([TASK_NAMES.get(task,task),f"{n:,}",f"{2*n:,}",f"{100*n/total:.1f}%" if total else "—"])
    if total:data_rows.append(["合计",f"{total:,}",f"{2*total:,}","100%"])
    table_html=table(["训练数据源","同手双指时刻","单指图","训练比例"],data_rows)
    split_rows=[]
    for split,d in prep.get("splits",{}).items():
        split_rows.append([split,d.get("pairs"),d.get("single_fingers"),
                           f"{100*d['empty_finger_fraction']:.2f}%" if finite(d.get("empty_finger_fraction")) else "—"])
    text="<p>一个双指样本是同一只夹爪的两根手指在同一时刻的观测，不是机器人左右手的拼接。先按源轨迹/episode 划分，再抽样；同一时刻两指不能跨集合。</p>"
    text+=table_html+table(["集合","双指时刻","单指图","空载指端比例"],split_rows)
    if not completion["prepared_data_qa_and_counts"]:
        text+="<p class='warning'>当前评估结果未与项目完整预处理清单完成一致性核验；上表是项目清单，不能冒充当前结果实际使用的数据量。</p>"
    rule=prep.get("geometry_rule",{})
    if rule:
        text+=("<p>几何代理标签使用固定规则：深度阈值 "+esc(rule.get("depth_threshold"))+
               "，每个接触连通域至少 "+esc(rule.get("min_pixels"))+" 像素。阈值来自训练数据，最终测试不调阈值。</p>")
    return text

def cloud_snapshot(root,evidence):
    status=load_json(root/"cloud/status.json",{})
    pipeline=load_json(root/"cloud/pipeline_state.json",{})
    if not status:return "<p>没有本任务的云端状态快照；本报告不推断作业资源状态。</p>"
    logicals={Path(e["run_directory"]).name for e in evidence}
    jobs=[j for j in status.get("jobs",[]) if j.get("logical") in logicals]
    rows=[[j.get("logical"),j.get("JobId"),j.get("Status"),j.get("GmtFinishTime") or "—"] for j in jobs]
    text="<p>本任务台账快照更新时间："+esc(status.get("updated_at","未知"))+"。快照记录活动任务 "+esc(status.get("active_jobs","未知"))+" 个、申请 GPU "+esc(status.get("active_gpus","未知"))+" 张。</p>"
    text+="<p>调度状态文件当前阶段："+esc(pipeline.get("phase","未知"))+"。报告渲染成功后调度器才会写入 complete；因此本页按训练、作业及评估标记核验，不单凭这个阶段字符串判断是否完成。</p>"
    if rows:text+=table(["本任务运行","DLC Job ID","快照状态","结束时间（UTC）"],rows)
    text+="<p class='muted'>这是本项目最后一次查询的记录，不是实时云端查询，也不代表其他用户的作业或资源状态。</p>"
    return text

def render(report_dir,project_root):
    root=Path(project_root).resolve();folder=Path(report_dir).resolve()
    if not root.is_dir() or not (root/"provenance/source.json").is_file():
        raise ValueError("Project root is missing its provenance marker")
    if not inside(folder,root):raise ValueError("Report directory must remain within the project")
    summary_path=folder/"result_summary.json"
    markdown_path=folder/"report.md"
    if not summary_path.is_file() or not markdown_path.is_file():raise FileNotFoundError("Need result_summary.json and report.md")
    input_hashes={"result_summary.json":sha(summary_path),"report.md":sha(markdown_path)}
    summary=load_json(summary_path)
    rows,evidence=load_sources(summary,root)
    groups=grouped(rows)
    completion=completion_evidence(summary,rows,evidence,root)
    selection=summary.get("lambda_selection",{})
    generated=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")
    completed=completion["complete"]
    title="触觉编码器对比实验结果"
    label="完整结果已核验" if completed else "结果材料未完整核验"
    findings_html="<ul class='findings'>"+"".join("<li>"+esc(s)+"</li>" for s in findings(groups,selection,completed))+"</ul>"
    stages=[
        ("数据准备","QA_READY、源轨迹划分和评估样本数量一致",completion["prepared_data_qa_and_counts"]),
        ("权重筛选","只使用 val；最终 test 与任务外集合不参与",completion["validation_only_weight_selection"]),
        ("正式训练","MAE/MAE＋力采用相同的 0、1、2 三个训练种子",completion["three_matched_seeds"]),
        ("产物验收","六份完整训练、JOB_COMPLETE 和 EVALUATION_COMPLETED 标记匹配",completion["six_completed_jobs_and_evaluations"]),
        ("数据一致性","六次结果使用同一份 manifest 与归一化签名",completion["identical_data_signature"])]
    stage_html="<div class='stage-list'>"+"".join("<div class='stage'>"+badge(ok)+"<strong>"+esc(name)+"</strong><span>"+esc(desc)+"</span></div>" for name,desc,ok in stages)+"</div>"
    figure=""
    image_path=folder/"test_comparison.png"
    if image_path.is_file():
        image=image_path.read_bytes()
        if not image.startswith(b"\x89PNG\r\n\x1a\n") or len(image)>16*1024*1024:
            raise ValueError("Comparison figure must be a PNG smaller than 16 MiB")
        input_hashes["test_comparison.png"]=sha(image_path)
        figure=("<figure><img alt='训练任务留出集合的单指与双指对照，误差线为训练种子标准差' src='data:image/png;base64,"+
                base64.b64encode(image).decode("ascii")+"'><figcaption>训练任务留出集合。柱高为训练种子均值，误差线为训练种子标准差；不是 episode bootstrap 置信区间。</figcaption></figure>")
    choice=("开发集候选满足预先设定的筛选准则。" if selection.get("accepted_as_improvement") else
            "开发集没有确认同时满足全部准则的改进候选；保留正权重是为了完成正式对照，不表示筛选已经证明力监督更好。")
    score_rows=[]
    for v in selection.get("validation_scores",[]):
        score_rows.append([v.get("lambda_force"),number(v.get("geometry")),number(v.get("force")),number(v.get("contact")),
                           "是" if v.get("eligible") else "否"])
    selection_table=table(["λ","验证几何 nRMSE","验证 SDK 力 nRMSE","验证接触准确率","满足全部准则"],score_rows) if score_rows else ""
    limitations=[
        "六维力来自 X5 SDK 对触觉的估计，属于辅助监督，不是独立六维力传感器的真值。力探针提升仅直接说明 SDK 力更易解码。",
        "接触质心、面积和二阶矩来自固定 DD 处理规则，属于几何代理标签，不是人工逐帧确认的接触真值。",
        "单指与双指使用同一套冻结编码器；双指的收益反映观测增加，不应解释为重新训练出了更强的编码器。",
        "拼接＋左右差值没有增加信息，对线性探针也没有增加表达能力。线性表中的两行使用等价解；非线性 MLP 对照另行匹配参数量。",
        "同一源轨迹的帧并非独立样本。各 evaluation.json 提供以完整 episode 为单位的 bootstrap 区间；本页 ± 为训练种子标准差，两者含义不同。",
        "漏斗与三杯只有有限数量的源 episode；这些结果不能保证任意新任务、传感器个体或更换盖板后都直接泛化。",
        "已核对录制视频 PTS 与状态表时间对齐，但原始传感器时钟不可用，不能把这一检查描述成完成了设备级同步标定。"]
    css="""*{box-sizing:border-box}body{margin:0;color:#172635;background:#f3f6f9;font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:36px 28px 64px}header{padding:28px 32px;background:#102f49;color:white;border-radius:12px}h1{font-size:29px;margin:0 0 8px}h2{font-size:21px;color:#123a55;margin:0 0 14px}h3{font-size:17px;margin:24px 0 10px}.subtitle{color:#d7e4ed;margin:8px 0}.pill{display:inline-block;background:#eef5ff;color:#183f62;padding:3px 12px;border-radius:20px;font-weight:650}.pill.pending{background:#fff0cb;color:#744f00}section{background:#fff;margin-top:22px;padding:26px 30px;border:1px solid #e1e8ef;border-radius:10px}.findings{margin:0;padding-left:22px}.findings li{margin:9px 0}.table-wrap{overflow:auto;margin:12px 0 18px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}caption{text-align:left;font-weight:600;margin-bottom:8px}th{text-align:left;background:#edf3f7;color:#24455d;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #e2e8ee;vertical-align:top}tbody tr:nth-child(even){background:#f8fafc}td:nth-child(n+3){font-variant-numeric:tabular-nums}.muted,figcaption{color:#60717e;font-size:13px}.warning{border-left:4px solid #c58b20;background:#fff8e8;padding:12px 16px;color:#714f17}figure{margin:20px 0}img{max-width:100%;height:auto;border:1px solid #e5ebef;border-radius:6px}figcaption{margin-top:6px}.stage{display:grid;grid-template-columns:112px 94px 1fr;align-items:start;gap:12px;padding:11px 0;border-bottom:1px solid #e6ecf1}.badge{display:inline-block;border-radius:5px;padding:2px 8px;font-size:12px;text-align:center;white-space:nowrap}.badge.pass{background:#e9f6ef;color:#18633d}.badge.pending{background:#fff1d4;color:#795719}code{background:#f1f4f7;padding:2px 5px;border-radius:4px;font-size:13px}footer{color:#667885;font-size:12px;padding-top:22px;overflow-wrap:anywhere}ul{padding-left:22px}a{color:#1d628f}@media(max-width:700px){main{padding:16px 12px}header,section{padding:20px 16px}h1{font-size:24px}.stage{grid-template-columns:100px 1fr}.stage span:last-child{grid-column:1/-1}}@media print{body{background:white}main{max-width:none;padding:0}header{background:white;color:#172635;border:1px solid #ccd6df}.subtitle{color:#506775}section{break-inside:auto;border:none;padding:18px 0}.table-wrap{overflow:visible}tr,figure,.stage{break-inside:avoid}}"""
    body=("<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
          "<title>"+title+"</title><style>"+css+"</style></head><body><main>"
          "<header><h1>"+title+"</h1><span class='pill "+("" if completed else "pending")+"'>"+esc(label)+"</span>"
          "<p class='subtitle'>共享单指 ViT-B/16 · MAE 与 MAE＋SDK 六维力 · 单指／同手双指表示</p>"
          "<p class='subtitle'>原始汇总："+esc(summary.get("generated_at_Asia_Shanghai","未知"))+"　｜　渲染："+esc(generated)+"</p></header>"
          "<section><h2>结果要点</h2>"+findings_html+"</section>"
          "<section><h2>完成判据</h2>"+stage_html+
          "<p class='muted'>结果汇总文件状态为 "+esc(summary.get("status","未知"))+"。只有标记、样本和六个成对种子均匹配时，本页才标为完整结果。</p></section>"
          "<section><h2>数据与配比</h2>"+data_section(completion,rows)+"</section>"
          "<section><h2>力监督权重的选择</h2><p>选择 λ＝<strong>"+esc(selection.get("selected_lambda","未知"))+
          "</strong>。"+esc(choice)+"</p>"+selection_table+
          "<p>MAE 为主损失；六维力只做回归，经过投影 MLP 后预测，下游使用投影之前的 768 维平均池化特征。"
          "力权重候选为 0.1、0.3、0.5，几何和接触容忍线是预先约定的工程准则，不是理论保证。</p></section>"
          "<section><h2>编码器本身：冻结单指特征的探针</h2><p>两类编码器预测相同的单指接触、几何和 SDK 力目标。"
          "各任务等权汇总；nRMSE 越低越好，接触平衡准确率越高越好。</p>"+per_finger_table(groups)+"</section>"
          "<section><h2>主对照：同一双侧目标，单指还是双指</h2><p>每种表示都在相同的双指时刻预测相同的双侧目标。"
          "表中 ± 是三个训练种子的样本标准差；只有一个种子时不显示标准差。</p>"
          "<p>“单指 L/R 平均”是 HTML 中新增的派生统计：对每个训练种子的左右单指同目标指标先取算术平均，再计算跨种子的均值和标准差。"
          "它不是合用两侧特征，也不是平均两个标准差；L/R 明细仍单独列出，原始 JSON 不变。</p>"+figure+main_result_tables(groups)+"</section>"
          "<section><h2>显式左右差值的非线性对照</h2><p>拼接与拼接＋差值使用近似相同参数总量的小 MLP，"
          "编码器完全冻结，验证集决定早停。输入维数不同，隐藏宽度经过调整；实际参数量列在表中。</p>"+mlp_table(groups)+"</section>"
          "<section><h2>适用范围与证据限制</h2><ul>"+"".join("<li>"+esc(s)+"</li>" for s in limitations)+"</ul></section>"
          "<section><h2>运行产物与云端快照</h2>"+source_table(evidence)+cloud_snapshot(root,evidence)+"</section>"
          "<footer>原始 report.md 与 result_summary.json 保持不变。本 HTML 内嵌样式及图像，不加载外部资源。"
          "<br>result_summary.json SHA-256："+esc(input_hashes["result_summary.json"])+
          "<br>report.md SHA-256："+esc(input_hashes["report.md"])+"</footer></main></body></html>")
    for name,digest in input_hashes.items():
        if sha(folder/name)!=digest:raise RuntimeError("Input artifact changed while rendering")
    destination=folder/"reviewed_report.html"
    tmp=destination.with_suffix(".html.tmp");tmp.write_text(body);os.replace(tmp,destination)
    receipt={"rendered_at_Asia_Shanghai":generated,"complete_evidence":completed,
             "input_sha256":input_hashes,"output_sha256":sha(destination),"source_count":len(rows),
             "completion_checks":{k:v for k,v in completion.items() if k not in ["data_summary","data_qa"]},
             "render_script_sha256":sha(Path(__file__)),"output":str(destination)}
    rp=folder/"reviewed_report_receipt.json";rt=rp.with_suffix(".json.tmp")
    rt.write_text(json.dumps(receipt,ensure_ascii=False,indent=2));os.replace(rt,rp)
    return receipt

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report-dir",required=True);p.add_argument("--project-root",required=True)
    a=p.parse_args()
    result=render(a.report_dir,a.project_root)
    print(json.dumps({"event":"reviewed_report_rendered","output":result["output"],
                      "complete_evidence":result["complete_evidence"],"sha256":result["output_sha256"]}),flush=True)

if __name__=="__main__":main()
