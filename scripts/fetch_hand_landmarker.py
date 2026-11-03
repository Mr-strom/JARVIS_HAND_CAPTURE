"""Fetch the official versioned CPU-compatible MediaPipe Hand Landmarker asset."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
target = ROOT/'assets/models/hand_landmarker.task'
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    with urllib.request.urlopen(URL, timeout=90) as response:
        target.write_bytes(response.read())
digest = hashlib.sha256(target.read_bytes()).hexdigest()
assert digest == 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1', 'Unexpected official model checksum'
manifest = ROOT/'reports/hand_landmarker_manifest.json'
if manifest.exists():
    assert json.loads(manifest.read_text())['sha256'] == digest, 'Model checksum changed'
manifest.write_text(json.dumps({'url':URL,'sha256':digest,'bytes':target.stat().st_size},indent=2),encoding='utf-8')
print(manifest.read_text())
