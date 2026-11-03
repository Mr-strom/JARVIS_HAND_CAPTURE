"""Plots and per-recording metrics; private paths stay out of public aggregate CSV."""
import json,csv,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
result=json.loads((root/'reports/personal_model_results.json').read_text());history=json.loads((root/'reports/training_history.json').read_text());z=np.load(root/'private/learning/predictions.npz')
for method in ['baseline','optimizer','neural']:
    accelerations=[]
    for clip in np.unique(z['clip_ids'][z['held_out']]):
        mask=z['clip_ids']==clip;q=z[method][mask];t=z['timestamps'][mask];dt=np.diff(t)
        v=np.diff(q,axis=0)/dt[:,None];valid=(dt[1:]<.3)&(dt[:-1]<.3)
        if valid.any():accelerations.extend(np.linalg.norm(np.diff(v,axis=0)[valid]/((dt[1:]+dt[:-1])/2)[valid,None],axis=1))
    result[method]['mean_joint_acceleration_norm_rad_s2']=float(np.mean(accelerations)) if accelerations else None
with (root/'reports/method_comparison.csv').open('w',newline='') as f:
    keys=['method','held_out_fingertip_error_mm','held_out_pinch_error_mm','joint_limit_violations','mean_joint_vector_speed_rad_s','mean_joint_acceleration_norm_rad_s2'];w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader()
    for method in ['baseline','optimizer','neural']:w.writerow({'method':method,**result[method]})
(root/'reports/personal_model_results.json').write_text(json.dumps(result,indent=2))
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
ax[0].plot([e['training_loss'] for e in history],label='train');ax[0].plot([e['validation_loss'] for e in history],label='validation');ax[0].set(title='Personal MLP learning',xlabel='Epoch',ylabel='Normalized teacher MSE');ax[0].legend()
methods=['baseline','optimizer','neural']
for a,key,title in [(ax[1],'held_out_fingertip_error_mm','Held-out geometry target error'),(ax[2],'held_out_pinch_error_mm','Held-out pinch-distance error')]:
 a.bar(methods,[result[m][key] for m in methods]);a.set(title=title,ylabel='mm (kinematic proxy)');a.tick_params(axis='x',rotation=15)
fig.tight_layout();fig.savefig(root/'reports/personal_learning_results.png',dpi=160);plt.close(fig)
print(json.dumps(result,indent=2))
