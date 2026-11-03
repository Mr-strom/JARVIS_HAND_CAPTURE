"""Assemble one LOCAL pending-approval v5 candidate; never publish/apply license."""
from pathlib import Path
import hashlib,json,re,zipfile
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'submission/release-audit'

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=ROOT/'submission/final/JARVIS-Hand-final-public-code-results.zip'
    manifest=json.loads((ROOT/'submission/final/public_manifest.json').read_text())
    assert digest(frozen.read_bytes())==manifest['archive_sha256']
    with zipfile.ZipFile(frozen) as z:payload={name:z.read(name) for name in z.namelist()}
    for name,data in payload.items():assert digest(data)==manifest['files'][name]
    for name in ['docs/RELEASE_LICENSE_REVIEW.md','docs/RELEASE_QUICKSTART.md','docs/GITHUB_AND_SUBMISSION_DRAFT.md',
                 'scripts/restore_release_artifacts.py','scripts/release_sample_smoke.py','scripts/release_leap_metrics_smoke.py','scripts/build_release_candidate.py',
                 'reports/release_privacy_review.json','reports/release_dependency_inventory.json','reports/release_personal_artifact_inventory.json','reports/release_tests.log']:
        payload[name]=(ROOT/name).read_bytes()
    approvals=[]
    old=json.loads((ROOT/'submission/final/private_artifact_manifest.json').read_text())
    for item in old['artifacts']:
        runtime=item['path'];p=Path(runtime)
        release='release_artifacts/heldout_landmarks/'+p.name if 'heldout_landmarks' in p.parts else 'release_artifacts/'+p.name
        data=(ROOT/runtime).read_bytes();assert digest(data)==item['sha256']
        if p.suffix=='.mp4':risk='Personal hand appearance/motion, residual room fragments; no audio. Demo contains upstream robot appearance/branding with attribution.'
        elif p.suffix=='.pt':risk='Personalized model and training distribution information; owner rights/reuse terms not yet granted; upstream software notices remain separate.'
        elif p.name=='calibration.json':risk='Personal joint range and calibration statistics; needed for exact ROM baseline; no direct name/location.'
        else:risk='Derived personal hand geometry, relative timing and source hashes; can be linked to matching originals. Not anonymous.'
        entry={'release_path':release,'runtime_path':runtime,'sha256':item['sha256'],'bytes':len(data),'publication_approved':False,'privacy_and_rights_risk':risk,'reuse_license':'not selected or applied'}
        approvals.append(entry);payload[release]=data
    for filename in ['leap_heldout_reference.npz','leap_heldout_provenance.json']:
        runtime='private/release_audit/'+filename;data=(ROOT/runtime).read_bytes();release='release_artifacts/'+filename
        entry={'release_path':release,'runtime_path':runtime,'sha256':digest(data),'bytes':len(data),'publication_approved':False,
            'privacy_and_rights_risk':'Personal held-out features and robot command trajectories/target geometry (NPZ), or linkage hash (JSON); no training clips or raw media. Needed to reproduce archived three-way LEAP metric comparison without raw videos.',
            'reuse_license':'not selected or applied'}
        approvals.append(entry);payload[release]=data
    payload['release_artifacts/MANIFEST.json']=json.dumps({'publication_status':'REQUIRES OWNER APPROVAL; local testing is not publication permission','artifacts':approvals},indent=2).encode()
    payload['RELEASE_APPROVAL_REQUIRED.json']=json.dumps({'status':'not approved; not published','code_and_results_release_approved':False,
        'original_code_license_recommendation':'MIT for original code only; copyright name and choice awaiting owner approval; no license applied',
        'personal_artifacts':approvals,'excluded':['all34 raw recordings','unmasked media','training/validation recordings','entire original private folder','model assets','third-party wheels/binaries','secrets','full prediction caches','research snapshots'],
        'reproducibility_limit':'Inference and held-out comparisons reproducible from selected derived data; full personal retraining requires private original dataset.',
        'publication_operation':'GitHub destination and any competition upload require separate explicit approval; none performed'},indent=2).encode()
    intro=('> **v5 RC1: LOCAL APPROVAL CANDIDATE — NOT CLEARED FOR PUBLICATION.** No original-code license has been applied.\n> Start with [installation and demo links](docs/RELEASE_QUICKSTART.md), [licensing review](docs/RELEASE_LICENSE_REVIEW.md) and `RELEASE_APPROVAL_REQUIRED.json`.\n> Personal files in `release_artifacts/` are proposed release copies; all approval flags remain false. Raw recordings are excluded.\n\n')
    payload['README.md']=intro.encode()+payload['README.md']
    payload['PUBLICATION_STATUS.md']=b'LOCAL v5 RC1 ONLY. All personal artifact release, original-code license selection and GitHub/upload approval remain pending. Do not publish this ZIP or its contents. See RELEASE_APPROVAL_REQUIRED.json.\n'
    for name,data in list(payload.items()):
        if name.endswith(('.md','.json','.py','.log','.txt','.csv')):
            data=data.replace(str(ROOT).encode(),b'C:\\path\\jarvis-hand').replace(str(ROOT).replace('\\','\\\\').encode(),b'C:\\\\path\\\\jarvis-hand')
            payload[name]=data
    suspicious=[]
    for name,data in payload.items():
        assert not any(p in ('private','data','assets','.venv','checkpoints') for p in Path(name).parts),name
        if name.endswith(('.md','.json','.py','.log','.txt','.csv','.lock','.in')):
            for pat in [rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'AKIA[A-Z0-9]{16}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'sk-[A-Za-z0-9]{32,}']:
                if re.search(pat,data):suspicious.append(name)
    assert not suspicious,suspicious
    # Only file hashes are stored inside; archive hash lives outside to avoid self-reference.
    filehash={name:digest(data) for name,data in payload.items()}
    payload['RELEASE_FILE_HASHES.json']=json.dumps(filehash,indent=2).encode()
    candidate=OUT/'JARVIS-Hand-v5-RC1-REQUIRES-APPROVAL.zip'
    with zipfile.ZipFile(candidate,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(payload.items()):
            info=zipfile.ZipInfo(name,date_time=(2026,11,3,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
    approval_payload=json.loads(payload['RELEASE_APPROVAL_REQUIRED.json'])
    (OUT/'APPROVAL_REQUIRED.json').write_text(json.dumps({**approval_payload,'candidate_sha256':digest(candidate.read_bytes()),'candidate_bytes':candidate.stat().st_size,'files':len(payload)},indent=2))
    print(json.dumps({'candidate':str(candidate),'sha256':digest(candidate.read_bytes()),'bytes':candidate.stat().st_size,'files':len(payload),'personal_artifacts':len(approvals),'license_applied':False,'published':False},indent=2))

if __name__=='__main__':main()
