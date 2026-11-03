"""Read-only preservation, archive and media verification; write aggregate audit."""
from pathlib import Path
import json,hashlib,zipfile,subprocess
import cv2
ROOT=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    before=json.loads((ROOT/'checkpoints/pre_final_hashes.json').read_text());protected={}
    for rel,item in before.items():
        p=Path(rel);parts=p.parts
        if p.suffix=='.py' or parts[0] in ('assets','data') or rel.replace('\\','/') in ('private/calibration.json','private/learning/personal_mlp.pt','private/learning/predictions.npz') or (parts[0]=='private' and p.suffix in ('.mp4','.npz')):
            protected[rel]=digest(ROOT/rel)==item['sha256']
    assert all(protected.values()),[k for k,v in protected.items() if not v]
    assert digest(ROOT/'docs/README_V4_ARCHIVE.md')==before['README.md']['sha256']
    backup=ROOT/'checkpoints/pre_final_verified.zip'
    with zipfile.ZipFile(backup) as z:assert z.testzip() is None
    out=ROOT/'submission/final';public=json.loads((out/'public_manifest.json').read_text());private=json.loads((out/'private_artifact_manifest.json').read_text())
    with zipfile.ZipFile(out/'JARVIS-Hand-final-public-code-results.zip') as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(public['files'])
        for name in z.namelist():
            assert not any(p in ('private','data','checkpoints','.venv','assets') for p in Path(name).parts),name
            assert hashlib.sha256(z.read(name)).hexdigest()==public['files'][name],name
    assert digest(out/'JARVIS-Hand-final-public-code-results.zip')==public['archive_sha256']
    with zipfile.ZipFile(out/'JARVIS-Hand-private-review-bundle.zip') as z:
        assert z.testzip() is None
        for item in private['artifacts']:
            assert item['approval'] is False
            assert digest(ROOT/item['path'])==item['sha256']
            assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
    assert digest(out/'JARVIS-Hand-private-review-bundle.zip')==private['private_archive_sha256']
    media=[]
    for rel in ['private/final/JARVIS_Hand_final_demo.mp4','private/final/personal_hand_sample.mp4','private/final/cross_embodiment_masked.mp4','private/final/leap_replay_masked.mp4','private/final/grasp_recovery.mp4']:
        p=ROOT/rel;c=cv2.VideoCapture(str(p));fps=c.get(cv2.CAP_PROP_FPS);expected=round(c.get(cv2.CAP_PROP_FRAME_COUNT));n=0
        while True:
            ok,f=c.read()
            if not ok:break
            n+=1
        c.release();assert n==expected and n>0,(rel,n,expected)
        streams=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(p)]))['streams']
        assert not any(s['codec_type']=='audio' for s in streams),rel
        media.append({'path':rel,'decoded_frames':n,'fps':fps,'audio_streams':0,'sha256':digest(p)})
    (ROOT/'reports/final_video_validation.json').write_text(json.dumps({'media':media,'private':True,'published':False},indent=2))
    report={'protected_files_checked':len(protected),'all_original_code_models_recordings_and_weights_unchanged':all(protected.values()),'original_readme_archived_exactly':True,
        'checkpoint_crc_passed':True,'checkpoint_sha256':digest(backup),'public_archive_files':len(public['files']),
        'public_archive_private_artifact_check':'passed','all_archive_and_artifact_hashes_match':True,'final_video_frames':media[0]['decoded_frames'],'final_video_seconds':media[0]['decoded_frames']/media[0]['fps'],
        'current_project_tests':(ROOT/'reports/final_tests.log').read_text().strip(),
        'archive_smoke':json.loads((ROOT/'reports/final_archive_smoke.json').read_text()),
        'portable_replay':json.loads((ROOT/'reports/cross_portable_validation.json').read_text()),
        'publication_approval_pending':True,'pushed_or_uploaded':False,'original_code_license':'owner selection pending; no reuse rights granted by packaging'}
    (ROOT/'reports/final_delivery_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
