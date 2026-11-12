import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from dataset import TactileDataset
from prepare_data import geometry,force_valid,choose_stratified

class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.root=Path(self.tmp.name)
        raw=np.zeros((2,2,96,128,3),dtype=np.uint16);raw[:,:,:,:,1:]=30000
        raw[0,0,:,:,1]=29000;raw[0,1,:,:,1]=31000
        np.save(self.root/'train_dd.npy',raw)
        forces=np.zeros((2,2,6),dtype=np.float32);forces[0,0,0]=9930
        np.save(self.root/'train_force.npy',forces);np.save(self.root/'train_force_valid.npy',force_valid(forces))
        np.save(self.root/'train_geometry.npy',np.zeros((2,2,7),dtype=np.float32))
        norm=dict(offset=[0,30000,30000],mean=[0,0,0],std=[1000,1000,1000],force_mean=[0]*6,force_std=[1]*6)
        (self.root/'normalization.json').write_text(json.dumps(norm))
    def tearDown(self):self.tmp.cleanup()
    def test_signed_and_pair_order(self):
        d=TactileDataset(self.root)
        self.assertEqual(len(d),4);self.assertEqual(d.pair_count,2)
        self.assertEqual(tuple(d[0]['image'].shape),(3,192,256))
        self.assertAlmostEqual(float(d[0]['image'][1].mean()),-1.)
        self.assertAlmostEqual(float(d[1]['image'][1].mean()),1.)
        self.assertEqual(d[2]['pair_index'],1);self.assertEqual(d[2]['finger'],0)
        self.assertTrue(torch.equal(d[0]['image'],d[0]['image']))
    def test_zero_force_is_valid_missing_not_zero(self):
        d=TactileDataset(self.root)
        self.assertFalse(d[0]['force_valid'][0]);self.assertTrue(d[0]['force_valid'][1])
        self.assertTrue(torch.isfinite(d[0]['force']).all())
    def test_aspect_ratio(self):
        with self.assertRaises(ValueError):TactileDataset(self.root,image_size=(224,224))
    def test_geometry(self):
        x=np.zeros((96,128,3),dtype=np.uint16);g=geometry(x,20)
        self.assertEqual(g[0],0);self.assertTrue(np.isnan(g[1:]).all())
        x[24:72,32:96,0]=100;g=geometry(x,20)
        np.testing.assert_allclose(g[1:4],[.5,.5,.25],atol=1e-6)
    def test_disconnected_noise_is_not_contact(self):
        x=np.zeros((96,128,3),dtype=np.uint16)
        x[::10,::10,0]=100
        self.assertEqual(geometry(x,20)[0],0)
    def test_sampling_unique(self):
        rows=[dict(episode=i//20,hand='left',row=i,stratum='active' if i%3 else 'low_force') for i in range(100)]
        selected=choose_stratified(rows,80,np.random.default_rng(1))
        self.assertEqual(len(selected),80);self.assertEqual(len({x['row'] for x in selected}),80)
if __name__=='__main__':unittest.main()
