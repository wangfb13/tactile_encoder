"""Renderer checks; every fixture stays under this project's tests directory."""
from pathlib import Path
import base64, hashlib, importlib.util, json, sys, tempfile, unittest
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("render_report",ROOT/"scripts/render_report.py")
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)

class Tags(HTMLParser):
    def __init__(self):super().__init__();self.tags=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        self.tags.append(tag)
        if tag=="img":self.images.append(dict(attrs))

class RenderTests(unittest.TestCase):
    def test_existing_fixture_renders_real_tables_inline_png_and_keeps_inputs(self):
        folder=ROOT/"tests/evaluation_fixture/release_report"
        before={n:renderer.sha(folder/n) for n in ["report.md","result_summary.json"]}
        result=renderer.render(folder,ROOT)
        self.assertFalse(result["complete_evidence"])
        page=(folder/"reviewed_report.html").read_text()
        tags=Tags();tags.feed(page)
        self.assertGreaterEqual(tags.tags.count("table"),7)
        self.assertNotIn("pre",tags.tags)
        self.assertTrue(tags.images)
        self.assertTrue(all(i["src"].startswith("data:image/png;base64,") for i in tags.images))
        self.assertNotIn("src='https://",page)
        self.assertIn("结果材料未完整核验",page)
        self.assertEqual(before,{n:renderer.sha(folder/n) for n in before})
    def fixture(self,root,geometry=1.2):
        (root/"provenance").mkdir()
        (root/"provenance/source.json").write_text("{}")
        prepared=root/"data/prepared";prepared.mkdir(parents=True)
        counts={s:10 for s in ["train","val","test","ood_funnel","ood_cups"]}
        splits={s:{"pairs":10,"single_fingers":20,"datasets":{"vase":10},
                   "empty_finger_fraction":.15} for s in counts}
        (prepared/"summary.json").write_text(json.dumps({"splits":splits,"geometry_rule":{"depth_threshold":30,"min_pixels":12}}))
        (prepared/"QA_READY.json").write_text(json.dumps({"ready":True,"no_episode_overlap":True}))
        report=root/"reports/final";report.mkdir(parents=True)
        (report/"report.md").write_text("# ORIGINAL\n\nDo not change this source.\n")
        sources=[]
        for lam in [0.,.3]:
            for seed in [0,1,2]:
                run=root/"runs"/(f"final_{'mae' if lam==0 else 'force'}_s{seed}")
                ev=run/"evaluation";ev.mkdir(parents=True)
                def scores():
                    return {"geometry":{"macro_task_nrmse":1. if lam==0 else geometry},
                            "force":{"macro_task_nrmse":1. if lam==0 else .8},
                            "contact":{"macro_task_balanced_accuracy":.9}}
                probe={"splits":{s:scores() for s in ["val","test","ood_funnel","ood_cups"]}}
                row={"mode":"final","lambda_force":lam,"seed":seed,"checkpoint_sha256":f"checkpoint-{lam}-{seed}",
                     "data_signature":"same-synthetic-fixture","split_pair_counts":counts,"per_finger":probe,
                     "paired_same_target":{k:probe for k in renderer.REPRESENTATIONS},"mlp_difference_ablation":{}}
                file=ev/"evaluation.json";file.write_text(json.dumps(row))
                (ev/"EVALUATION_COMPLETED.json").write_text(json.dumps({"mode":"final","checkpoint_sha256":row["checkpoint_sha256"]}))
                (run/"JOB_COMPLETE.json").write_text(json.dumps({"phase":"final","lambda_force":lam,"seed":seed,"epochs":100,"max_steps":0}))
                (run/"COMPLETED.json").write_text(json.dumps({"status":"completed"}))
                (run/"summary.json").write_text(json.dumps({"status":"completed","lambda_force":lam,"seed":seed,"epochs":100}))
                sources.append({"file":str(file.relative_to(root)),"sha256":renderer.sha(file)})
        summary={"status":"complete","generated_at_Asia_Shanghai":"2026-11-12T00:00:00+08:00",
                 "lambda_selection":{"selected_lambda":.3,"accepted_as_improvement":True,
                                     "selection_split":"val","test_used":False},"sources":sources}
        (report/"result_summary.json").write_text(json.dumps(summary))
        return report
    def test_three_seed_markers_and_actual_worse_geometry_do_not_claim_force_better(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            root=Path(d);report=self.fixture(root,geometry=1.2)
            result=renderer.render(report,root)
            self.assertTrue(result["complete_evidence"])
            page=(report/"reviewed_report.html").read_text()
            self.assertIn("位置敏感用途应优先保留 MAE 基线",page)
            self.assertIn("均值升高 20.00%",page)
            self.assertIn("完整结果已核验",page)
    def test_missing_job_marker_overrides_summary_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            root=Path(d);report=self.fixture(root,geometry=.98)
            (root/"runs/final_mae_s0/JOB_COMPLETE.json").unlink()
            result=renderer.render(report,root)
            self.assertFalse(result["complete_evidence"])
            self.assertIn("不作最终方法优劣结论",(report/"reviewed_report.html").read_text())
    def test_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            root=Path(d);report=self.fixture(root)
            file=root/"runs/final_mae_s0/evaluation/evaluation.json"
            file.write_text(file.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"hash differs"):
                renderer.render(report,root)
    def test_paths_cannot_escape_project(self):
        with self.assertRaisesRegex(ValueError,"within the project"):
            renderer.render("/tmp",ROOT)
    def test_single_finger_average_is_computed_within_each_seed(self):
        rows=[]
        for left,right in [(1.,7.),(3.,1.),(2.,6.)]:
            rows.append({"paired_same_target":{
                "single_left":{"splits":{"test":{"geometry":{"macro_task_nrmse":left}}}},
                "single_right":{"splits":{"test":{"geometry":{"macro_task_nrmse":right}}}}}})
        values=renderer.paired_values(rows,"test","geometry","single_mean")
        self.assertEqual(values,[4.,2.,4.])
        self.assertAlmostEqual(renderer.average(values),10/3)
        self.assertAlmostEqual(renderer.sample_sd(values),(4/3)**.5)
        page=renderer.main_result_tables({0.:rows})
        self.assertIn("单指 L/R 平均",page)
        self.assertIn("3.3333 ± 1.1547",page)
    def test_html_content_is_escaped(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            root=Path(d);report=self.fixture(root)
            p=report/"result_summary.json";summary=json.loads(p.read_text())
            summary["generated_at_Asia_Shanghai"]="<script>alert(1)</script>"
            p.write_text(json.dumps(summary))
            renderer.render(report,root)
            page=(report/"reviewed_report.html").read_text()
            self.assertNotIn("<script>alert",page)
            self.assertIn("&lt;script&gt;alert",page)

if __name__=="__main__":unittest.main()
