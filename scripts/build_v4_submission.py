"""Package public source/results; private approval manifest is intentionally separate."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'submission/v4';out.mkdir(parents=True,exist_ok=True)
names=['README.md','HUMANOID_V4_REPORT.md','LEARNING_AND_MANIPULATION.md','RESEARCH_REUSE.md','THIRD_PARTY_NOTICES.md',
       'requirements.in','requirements.lock','requirements-learning.in','requirements-learning.lock','.gitignore',
       'live_mirror.py','neural_live.py','simulate.py','retargeting.py','physics_guard.py','personal_data.py',
       'train_personal.py','replay_personal.py','manipulation.py','humanoid_control.py','embodiment_adapter.py',
       'humanoid_personal_replay.py','bimanual_task.py','diagnose_grasp.py','grasp_recovery.py']
paths=[ROOT/x for x in names]
for directory in ['learning','scripts','tests','licenses']:
    paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
reports=['personal_model_results.json','training_history.csv','training_history.json','method_comparison.csv','personal_learning_results.png',
         'fk_validation.json','inference_benchmark.json','camera_benchmark.json','g1_asset_manifest.json','g1_initial_probe.json',
         'g1_stability_results.json','g1_render_control_results.json','g1_actuator_sweep.json','g1_personal_transfer_results.json',
         'g1_transfer_repeatability.json','grasp_diagnosis.json','grasp_recovery_results.json','bimanual_task_results.json','v4_demo_manifest.json',
         'v4_experiment_summary.json','v4_experiments.csv','v4_experiments.png','v4_tests.log']
paths.extend(ROOT/'reports'/name for name in reports)
with zipfile.ZipFile(out/'JARVIS-Hand-v4-public-code-results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if p.exists():z.write(p,p.relative_to(ROOT))
inventory={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
(out/'public_manifest.json').write_text(json.dumps({'files':inventory,'excluded':['data','private','weights','videos','checkpoints','raw research snapshots'],'published':False},indent=2))
audit=json.loads((ROOT/'reports/dataset_audit.json').read_text());train=[e for e in audit['split'] if e['partition']=='train'];test=[e for e in audit['split'] if e['partition']=='test']
samples=[max(train,key=lambda e:e['gesture_counts'].get('pinch',0)),max(test,key=lambda e:e['valid_frames'])]
private=[]
for e in samples:
    p=Path(e['source']);private.append({'role':'representative original personal recording','path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'categories_from_geometry':e['gesture_counts'],'partition':e['partition'],'approval':False})
for name,role in [('private/learning/personal_mlp.pt','trained checkpoint'),('private/learning/personal_comparison.mp4','personal source/LEAP comparison'),('private/JARVIS_Hand_v4_technical_demo.mp4','captioned technical demo'),('private/g1_personal_transfer.mp4','experimental G1 adapter replay')]:
    p=ROOT/name
    if p.exists():private.append({'role':role,'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'approval':False})
(out/'private_artifact_manifest.json').write_text(json.dumps({'publication_approval_required':True,'nothing_published':True,'privacy_review':'Review people/background/audio in raw footage before sharing. Do not publish this manifest or the private folder automatically.','proposed_artifacts':private},indent=2))
print(out/'JARVIS-Hand-v4-public-code-results.zip')
