import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate import (assert_no_episode_overlap,paired_representation,fit_ridge,predict_ridge,
                      fit_contact,predict_contact,regression_metrics,contact_metrics,
                      target_stats,evaluate_probe_set,fit_mlp_ablation,feature_health)
spec=importlib.util.spec_from_file_location("select_lambda",ROOT/"scripts"/"select_lambda.py")
selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def test_episode_and_source_leakage(self):
        row={"dataset":"a","episode":1,"episode_fingerprint":"same"}
        with self.assertRaises(ValueError):
            assert_no_episode_overlap({"train":[row],"val":[row]})
        with self.assertRaises(ValueError):
            assert_no_episode_overlap({"train":[row],"val":[dict(row,dataset="b",episode=2)]})
        assert_no_episode_overlap({"train":[row],"val":[dict(row,episode=2,episode_fingerprint="different")]})
    def test_pairing_and_linear_difference_identity(self):
        z=np.arange(48,dtype=np.float32).reshape(6,8)
        concat=paired_representation(z,"concat")
        diff=paired_representation(z,"concat_difference")
        np.testing.assert_array_equal(diff[:,:16],concat)
        np.testing.assert_array_equal(diff[:,16:],concat[:,:8]-concat[:,8:])
        rng=np.random.default_rng(1)
        w=rng.normal(size=(24,2))
        effective=np.concatenate([w[:8]+w[16:],w[8:16]-w[16:]],axis=0)
        np.testing.assert_allclose(diff@w,concat@effective,atol=1e-10)
    def test_valid_label_stats_do_not_turn_missing_into_zero(self):
        y=np.array([[1,np.nan],[3,100],[np.nan,200]],np.float32)
        valid=np.isfinite(y)
        m,s,n=target_stats(y,valid)
        np.testing.assert_allclose(m,[2,150])
        np.testing.assert_array_equal(n,[2,2])
        np.testing.assert_allclose(s,[1,50])
    def test_ridge_known_linear_relation_and_missing_labels(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(120,8)).astype(np.float32)
        w=rng.normal(size=(8,3)).astype(np.float32)
        y=x@w+np.array([1,2,3],np.float32)
        valid=np.ones_like(y,dtype=bool);valid[::4,1]=False
        y[~valid]=np.nan
        model=fit_ridge(x[:80],y[:80],valid[:80],x[80:],y[80:],valid[80:],alphas=[1e-6,1.])
        self.assertEqual(model["alpha"],1e-6)
        pred=predict_ridge(model,x)
        self.assertLess(np.max(np.abs(pred[valid]-y[valid])),.001)
    def test_contact_probe_constant_class_and_binary(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(120,4)).astype(np.float32)
        y=np.stack([x[:,0]>0,np.ones(120,dtype=bool)],axis=1).astype(np.float32)
        model=fit_contact(x[:80],y[:80],x[80:],y[80:])
        pred=predict_contact(model,x[80:])
        self.assertTrue(np.isfinite(pred).all())
        self.assertGreater(np.mean((pred>=.5)==y[80:]),.95)
    def test_episode_bootstrap_and_no_contact_geometry(self):
        y=np.array([[0],[0],[1],[1]],np.float32)
        pred=np.array([[.1],[.1],[1.2],[1.2]],np.float32)
        result=regression_metrics(pred,y,np.ones_like(y,bool),np.ones(1),["a","a","b","b"],["task"]*4,40)
        self.assertEqual(result["episode_count"],2)
        self.assertIn("macro_task_nrmse_episode_bootstrap_95ci",result)
        contact=contact_metrics(np.array([[.1],[.1],[.9],[.9]]),y,["a","a","b","b"],["task"]*4,40)
        self.assertEqual(contact["balanced_accuracy"],1.)
        empty=regression_metrics(np.zeros((4,1)),np.full((4,1),np.nan),np.zeros((4,1),bool),
                                 np.ones(1),["a","a","b","b"],["task"]*4,20)
        self.assertIsNone(empty["nrmse"])
        self.assertNotIn("macro_task_nrmse_episode_bootstrap_95ci",empty)
    def test_probe_suite_runs_and_saves_frozen_linear_heads(self):
        rng=np.random.default_rng(6)
        xs={s:rng.normal(size=(n,8)).astype(np.float32) for s,n in [("train",80),("val",40),("test",40)]}
        labels={}
        for s,x in xs.items():
            g=x[:,:6]*.1
            f=x[:,:6]*.5+.2
            labels[s]={"geometry":g,"geometry_valid":np.ones_like(g,bool),"force":f,
                       "force_valid":np.ones_like(f,bool),"contact":(x[:,:1]>0).astype(np.float32),
                       "groups":np.array([s+str(i//4) for i in range(len(x))]),"datasets":np.array(["a"]*len(x))}
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            r=evaluate_probe_set("unit",xs,labels,d,"cpu",bootstrap=10)
            self.assertLess(r["splits"]["test"]["geometry"]["nrmse"],.01)
            self.assertTrue((Path(d)/"probes"/"unit.pt").exists())
            mlp=fit_mlp_ablation(xs,labels,d,"unit","cpu",epochs=2,budget=500,bootstrap=0)
            self.assertTrue(np.isfinite(mlp["splits"]["test"]["force"]["nrmse"]))
            self.assertLessEqual(mlp["parameter_count"],500)
    def test_lambda_selection_prefers_smallest_and_rejects_test(self):
        def row(lam,g,f,c=.9,mode="selection_only"):
            return {"mode":mode,"data_signature":"fixture","split_pair_counts":{"train":100,"val":20},"lambda_force":lam,"seed":0,
                    "per_finger":{"splits":{"val":{"geometry":{"macro_task_nrmse":g},
                                  "force":{"macro_task_nrmse":f},"contact":{"macro_task_balanced_accuracy":c}}}}}
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            paths=[]
            for i,r in enumerate([row(0,1,1),row(.1,1.02,.9),row(.3,1.03,.8),row(.5,1.2,.7)]):
                p=Path(d)/(str(i)+".json");p.write_text(json.dumps(r));paths.append(p)
            chosen=selection.select(paths)
            self.assertEqual(chosen["selected_lambda"],.1)
            self.assertTrue(chosen["accepted_as_improvement"])
            paths[-1].write_text(json.dumps(row(.5,1.2,.7,mode="final")))
            with self.assertRaises(ValueError):selection.select(paths)
    def test_lambda_fallback_reports_no_improvement(self):
        def row(lam,g,f):
            return {"mode":"selection_only","data_signature":"fixture","split_pair_counts":{"train":100,"val":20},"lambda_force":lam,"seed":0,
                    "per_finger":{"splits":{"val":{"geometry":{"macro_task_nrmse":g},
                                  "force":{"macro_task_nrmse":f},"contact":{"macro_task_balanced_accuracy":.9}}}}}
        with tempfile.TemporaryDirectory(dir=ROOT/"tests") as d:
            paths=[]
            for i,r in enumerate([row(0,1,1),row(.1,1.2,.8)]):
                p=Path(d)/(str(i)+".json");p.write_text(json.dumps(r));paths.append(p)
            chosen=selection.select(paths)
            self.assertFalse(chosen["accepted_as_improvement"])
            self.assertEqual(chosen["recommended_lambda"],0.)
            self.assertEqual(chosen["selected_lambda"],.1)

if __name__=="__main__":unittest.main()
