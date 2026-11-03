"""Build a reviewable code/results archive using an explicit privacy-safe allowlist."""
from pathlib import Path
import zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
root_files=['README.md','LEARNING_AND_MANIPULATION.md','RESEARCH_REUSE.md','THIRD_PARTY_NOTICES.md',
            'requirements.in','requirements.lock','requirements-learning.in','requirements-learning.lock','.gitignore',
            'live_mirror.py','simulate.py','physics_guard.py','retargeting.py','personal_data.py','train_personal.py',
            'replay_personal.py','neural_live.py','manipulation.py']
report_files=['personal_model_results.json','method_comparison.csv','training_history.csv','training_history.json',
              'personal_learning_results.png','fk_validation.json','inference_benchmark.json','camera_benchmark.json',
              'manipulation_results.json','perception_results.json','hand_landmarker_manifest.json']
paths=[ROOT/p for p in root_files]
for folder in ['learning','tests','licenses','scripts']:
    paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
paths += [ROOT/'reports'/p for p in report_files]
out=ROOT/'submission';out.mkdir(exist_ok=True)
with zipfile.ZipFile(out/'JARVIS-Hand-code-and-results.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in paths:
        if p.exists():z.write(p,p.relative_to(ROOT))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
(out/'manifest.json').write_text(json.dumps({'files':manifest,'excluded':['data','private','checkpoints','downloads','assets','.venv','research source snapshots'],'published':False},indent=2))
print(out/'JARVIS-Hand-code-and-results.zip')
