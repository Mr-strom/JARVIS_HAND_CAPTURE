"""Separate public source/results from private release candidates. Never uploads."""
from pathlib import Path
import hashlib,json,re,zipfile,importlib.metadata,sys,platform
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission/final'


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env={'python':sys.version,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},'new_packages_installed_in_final_stage':False}
    (ROOT/'reports/verified_environment.json').write_text(json.dumps(env,indent=2))
    files=list(ROOT.glob('*.py'))
    files += [ROOT/x for x in ['README.md','FINAL_SUBMISSION_REPORT.md','HUMANOID_V4_REPORT.md','LEARNING_AND_MANIPULATION.md','RESEARCH_REUSE.md','PHASE1_REPORT.md','PHASE2_REPORT.md','THIRD_PARTY_NOTICES.md','requirements.in','requirements.lock','requirements-learning.in','requirements-learning.lock','.gitignore']]
    for directory in ['learning','scripts','tests','licenses','docs','experiments']:
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    reports=['personal_model_results.json','training_history.csv','training_history.json','method_comparison.csv','personal_learning_results.png',
        'fk_validation.json','inference_benchmark.json','camera_benchmark.json','leap_asset_manifest.json','hand_landmarker_manifest.json',
        'g1_asset_manifest.json','g1_initial_probe.json','g1_stability_results.json','g1_render_control_results.json','g1_actuator_sweep.json',
        'g1_personal_transfer_results.json','g1_transfer_repeatability.json','grasp_diagnosis.json','grasp_recovery_results.json','manipulation_results.json',
        'bimanual_task_results.json','v4_experiment_summary.json','v4_experiments.csv','v4_experiments.png','leap_demo.mp4',
        'cross_embodiment_results.json','cross_embodiment.csv','cross_portable_validation.json','final_grasp_recovery.json','final_grasp_summary.json',
        'final_grasp.csv','final_evaluation.png','final_demo_manifest.json','final_tests.log','final_archive_smoke.json','verified_environment.json','privacy_preparation.json',
        'final_grasp_render_validation.json']
    files.extend(ROOT/'reports'/x for x in reports)
    payload={str(p.relative_to(ROOT)).replace('\\','/'):p.read_bytes() for p in files if p.is_file()}
    # Public split evidence is sufficient for leakage regression; no source paths.
    audit=json.loads((ROOT/'reports/dataset_audit.json').read_text())
    clean={k:v for k,v in audit.items() if k!='split'}
    clean['split']=[{k:e[k] for k in ['index','group','partition','sha256','valid_frames','sampled_frames']} for e in audit['split']]
    payload['reports/dataset_audit.json']=json.dumps(clean,indent=2).encode()
    readme='''# Review before publication

The public ZIP contains source, licenses and aggregate results. It contains no personal media, weights, calibration or landmark arrays. The private ZIP must remain private until approved. Original-code licensing still needs owner confirmation. Third-party notices retain their original terms.

The private ZIP proposes one silent hand-masked sample and its timestamped landmarks, the preserved checkpoint and calibration, five held-out landmark caches, and the final masked demonstration. It is enough to reproduce held-out Shadow evaluation after restoring at its supplied paths. It cannot reproduce full training without the complete private dataset. Full original videos are not proposed for release because they visibly contain a face and room; audio was not cleared.

Do not publish the entire submission directory. The private manifest itself is private. No publication, push or upload has occurred. Approval must identify artifacts and destination. The owner should review the sample and final video, even after masking.
'''
    (OUT/'REVIEW_BEFORE_PUBLICATION.md').write_text(readme,encoding='utf-8')
    payload['PUBLICATION_STATUS.md']=('Private video/data/checkpoint release is pending owner approval. See README.\nOriginal project-code license selection is pending; third-party terms remain unchanged.\n').encode()
    for name,data in list(payload.items()):
        if name.endswith(('.md','.json','.py','.csv','.log','.txt')):
            data=data.replace(str(ROOT).encode(),b'C:\\path\\jarvis-hand')
            data=data.replace(str(ROOT).replace('\\','\\\\').encode(),b'C:\\\\path\\\\jarvis-hand')
            payload[name]=data
    suspicious=[]
    patterns=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[A-Z0-9]{16}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'sk-[A-Za-z0-9]{32,}']
    for name,data in payload.items():
        if name.endswith(('.md','.json','.py','.csv','.log','.txt','.lock','.in')):
            for pattern in patterns:
                if re.search(pattern,data):suspicious.append({'file':name,'pattern':pattern.decode()})
    if suspicious:raise RuntimeError('Potential secret pattern found; inspect locally before packaging')
    public=OUT/'JARVIS-Hand-final-public-code-results.zip'
    with zipfile.ZipFile(public,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):z.writestr(name,data)
    (OUT/'public_manifest.json').write_text(json.dumps({'files':{name:sha(data) for name,data in payload.items()},'archive_sha256':sha(public.read_bytes()),'published':False,'secret_pattern_hits':suspicious,'text_workspace_path_redacted':True,'license_status':'original-code owner choice pending; third-party notices preserved'},indent=2))
    private=[ROOT/p for p in ['private/final/personal_hand_sample.mp4','private/final/personal_hand_landmarks.json','private/learning/personal_mlp.pt','private/calibration.json','private/final/JARVIS_Hand_final_demo.mp4']]
    private.extend(p for p in (ROOT/'private/final/heldout_landmarks').iterdir() if p.is_file())
    private_zip=OUT/'JARVIS-Hand-private-review-bundle.zip'
    with zipfile.ZipFile(private_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in private:z.write(p,p.relative_to(ROOT))
        z.writestr('PRIVATE_RELEASE_REVIEW.md',readme)
    manifest={'published':False,'publication_approval_required':True,'minimum_rgb_sample':'One derived hand-masked clip from a genuine held-out recording; no raw originals proposed.',
        'reproducibility':'checkpoint + calibration + five held-out landmark caches reproduce Shadow evaluation. Small RGB sample cannot reproduce 34-video training.',
        'privacy':'Raw originals contain face/body/room and uncleared audio. Derivatives have no audio and mask non-hand pixels; manual owner review still required. Landmarks and weights are personal derived artifacts.',
        'private_archive_sha256':sha(private_zip.read_bytes()),'artifacts':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'approval':False} for p in private]}
    (OUT/'private_artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'public_archive':str(public),'public_files':len(payload),'public_bytes':public.stat().st_size,'private_review_archive':str(private_zip),'private_bytes':private_zip.stat().st_size,'uploaded':False},indent=2))

if __name__=='__main__':main()
