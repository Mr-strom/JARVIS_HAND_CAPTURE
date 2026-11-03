"""Download only the pinned Menagerie LEAP folder, including its MIT notice."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REV = 'c1a4eeb85694ae1dffe33ff1797d4e528928a133'
BASE = 'https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/' + REV + '/'


def fetch(item):
    path = item['path']
    target = ROOT / 'assets' / 'menagerie' / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(BASE + path, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
    data = target.read_bytes()
    git_blob_hash = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
    if git_blob_hash != item['sha']:
        raise ValueError(f'Git object hash mismatch: {path}')
    return {'path': path, 'url': BASE + path, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def main():
    req = urllib.request.Request(f'https://api.github.com/repos/google-deepmind/mujoco_menagerie/git/trees/{REV}?recursive=1', headers={'User-Agent':'JARVIS-Hand-phase1'})
    with urllib.request.urlopen(req, timeout=30) as response:
        tree = json.load(response)
    paths = [item for item in tree['tree'] if item['type']=='blob' and item['path'].startswith('leap_hand/')]
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, paths))
    (ROOT / 'reports/leap_asset_manifest.json').write_text(json.dumps({'revision':REV,'files':files},indent=2), encoding='utf-8')
    print(f'Downloaded {len(files)} files; {sum(item["bytes"] for item in files)} bytes. Revision {REV}', flush=True)


if __name__ == '__main__':
    main()
