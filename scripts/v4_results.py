"""Aggregate executed humanoid experiments without promoting rejected contacts."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'reports/bimanual_task_results.json').read_text());summary={}
for assisted in [False,True]:
    items=[r for r in rows if r['assisted']==assisted];name='assisted' if assisted else 'single'
    summary[name]={'attempts':len(items),'valid_grasps':sum(r['success'] for r in items),'integrity_aborts':sum(not r['physics_integrity_valid'] for r in items),'valid_table_stabilizations':sum(r['shared_stabilization_success'] for r in items),'max_lift_m':max(r['max_lift_m'] for r in items),'max_joint_excursion_rad':max(r['max_joint_limit_excursion_rad'] for r in items),'max_penetration_m_including_aborted_attempts':max(r['max_penetration_m'] for r in items),'mean_control_step_ms':float(np.mean([r['mean_control_step_ms'] for r in items])),'mean_real_time_factor':float(np.mean([r['real_time_factor'] for r in items]))}
(root/'reports/v4_experiment_summary.json').write_text(json.dumps(summary,indent=2))
with (root/'reports/v4_experiments.csv').open('w',newline='') as f:
    keys=['configuration']+list(summary['single']);w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    for name,values in summary.items():w.writerow({'configuration':name,**values})
fig,axes=plt.subplots(1,3,figsize=(12,3.7));names=list(summary)
for ax,key,title in [(axes[0],'valid_grasps','Valid lift-and-hold (primary task)'),(axes[1],'valid_table_stabilizations','Preliminary table stabilization'),(axes[2],'integrity_aborts','Integrity-aborted attempts')]:
    values=[summary[n][key] for n in names];ax.bar(names,values,color=['#577d9d','#d49243']);ax.set_ylim(0,5.8);ax.set_title(title,fontsize=10);ax.set_ylabel('Count out of five attempts')
    for i,v in enumerate(values):ax.text(i,v+.1,str(v),ha='center')
fig.suptitle('No reliable grasp advantage established; all failed/aborted trials retained',fontsize=11);fig.tight_layout();fig.savefig(root/'reports/v4_experiments.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
