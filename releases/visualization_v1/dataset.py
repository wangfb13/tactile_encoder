"""Single-finger views of the immutable, paired uint16 training manifest."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class TactileDataset(Dataset):
    def __init__(self, root, split='train', image_size=(192, 256)):
        self.root=Path(root); self.split=split; self.image_size=tuple(image_size)
        if self.image_size[0]*4 != self.image_size[1]*3:
            raise ValueError('DD resizing must preserve the 3:4 height:width aspect ratio')
        self.normalization=json.loads((self.root/'normalization.json').read_text())
        self.dd=np.load(self.root/f'{split}_dd.npy',mmap_mode='r')
        self.force=np.load(self.root/f'{split}_force.npy',mmap_mode='r')
        self.force_valid=np.load(self.root/f'{split}_force_valid.npy',mmap_mode='r')
        self.geometry=np.load(self.root/f'{split}_geometry.npy',mmap_mode='r')
        self.pair_count=len(self.dd)
        assert self.dd.dtype==np.uint16 and self.dd.shape[1:]==(2,96,128,3)
        assert self.force.shape==self.force_valid.shape==(self.pair_count,2,6)
        assert self.geometry.shape==(self.pair_count,2,7)
        self.offset=np.asarray(self.normalization['offset'],dtype=np.float32)
        self.mean=np.asarray(self.normalization['mean'],dtype=np.float32)
        self.std=np.asarray(self.normalization['std'],dtype=np.float32)
        self.force_mean=np.asarray(self.normalization['force_mean'],dtype=np.float32)
        self.force_std=np.asarray(self.normalization['force_std'],dtype=np.float32)
        assert np.all(self.std>0) and np.all(self.force_std>0)
    def __len__(self): return self.pair_count*2
    def __getitem__(self,index):
        index=int(index); pair,finger=divmod(index,2)
        # Cast before subtraction: uint16 cannot represent signed deformation.
        x=(np.asarray(self.dd[pair,finger],dtype=np.float32)-self.offset-self.mean)/self.std
        x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)))
        x=F.interpolate(x[None],size=self.image_size,mode='bilinear',align_corners=False)[0]
        raw=np.array(self.force[pair,finger],dtype=np.float32,copy=True)
        valid=np.array(self.force_valid[pair,finger],dtype=bool,copy=True)
        normalized=np.where(valid,(raw-self.force_mean)/self.force_std,0).astype(np.float32)
        return dict(image=x,force=torch.from_numpy(normalized),force_raw=torch.from_numpy(raw),
                    force_valid=torch.from_numpy(valid),geometry=torch.from_numpy(np.array(self.geometry[pair,finger],copy=True)),
                    pair_index=pair,finger=finger,index=index)
