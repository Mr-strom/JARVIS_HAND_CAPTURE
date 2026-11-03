"""CPU inference adapter, preserving Phase 2 filtering/loss fallback."""
import numpy as np
import torch
from retargeting import Retargeter,joint_features as angles
from personal_data import canonical
from train_personal import Network
from simulate import ROOT

def neural_features(world):
    return np.r_[canonical(world).reshape(-1),angles(world)]

class NeuralRetargeter(Retargeter):
    def __init__(self,names,ctrlrange,calibration=None):
        super().__init__(names,ctrlrange,calibration)
        torch.set_num_threads(2)
        checkpoint=torch.load(ROOT/'private/learning/personal_mlp.pt',map_location='cpu',weights_only=True)
        assert tuple(checkpoint['names'])==tuple(names),'Actuator convention mismatch'
        self.net=Network();self.net.load_state_dict(checkpoint['state']);self.net.eval()
        self.mean=np.array(checkpoint['mean']);self.std=np.array(checkpoint['std'])
        self.low=np.array(checkpoint['low']);self.high=np.array(checkpoint['high'])

    def map_features(self,features):
        x=np.asarray(features)
        if x.shape!=(79,) or not np.isfinite(x).all():raise ValueError('Invalid neural hand input')
        with torch.no_grad():normalized=self.net(torch.tensor((x-self.mean)/self.std,dtype=torch.float32)[None])[0].numpy()
        return np.clip(self.low+(self.high-self.low)*normalized,self.limits[:,0],self.limits[:,1])
