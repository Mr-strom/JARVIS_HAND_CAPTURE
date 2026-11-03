"""Fetch pinned upstream source/assets; never execute their installer scripts."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REV = "c4525805a82d1d6991bbed3ef74923f5d86d23b6"
ASSET_REV = "94cedd86c275bd4c7b92d61b5e5568c2c57f6483"


def download(url, name, expected=None):
    path = ROOT / "downloads" / name
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        print(f"Downloading {name}", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "JARVIS-Hand-phase1"})
        with urllib.request.urlopen(req, timeout=120) as response, path.with_suffix(path.suffix + '.part').open('wb') as out:
            shutil.copyfileobj(response, out)
        path.with_suffix(path.suffix + '.part').replace(path)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    if expected and digest != expected:
        raise ValueError(f"SHA256 mismatch: {name}")
    return path, {"url": url, "sha256": digest, "bytes": path.stat().st_size}


def safe_path(root, name):
    dest = (root / name).resolve()
    if not dest.is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe archive path: {name}")
    return dest


def main():
    source, source_info = download(f"https://codeload.github.com/BotRunner64/somehand/zip/{SOURCE_REV}", "somehand.zip", 'c636bdbe19e6af447cb2992708e7707acc756df278faabf4bbc1a35e87963eb1')
    source_root = ROOT / "third_party" / "somehand"
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            relative = '/'.join(member.filename.split('/')[1:])
            if not relative or member.is_dir():
                continue
            target = safe_path(source_root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as inp, target.open('wb') as out:
                shutil.copyfileobj(inp, out)
    url = f"https://huggingface.co/api/models/12e21/somehand-assets/tree/{ASSET_REV}/archives"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    entry = next(item for item in metadata if item['path'] == 'archives/mjcf_assets.tar.gz')
    expected = entry.get('lfs', {}).get('oid')
    asset, asset_info = download(f"https://huggingface.co/12e21/somehand-assets/resolve/{ASSET_REV}/archives/mjcf_assets.tar.gz", "mjcf_assets.tar.gz", expected)
    manifest = []
    with tarfile.open(asset, 'r:gz') as archive:
        for member in archive:
            manifest.append({'name': member.name, 'bytes': member.size})
            # Preserve model files and any bundled attribution, without extracting other robots.
            name = member.name.lstrip('./')
            selected = 'dex5_right/' in name or any(k in Path(name).name.lower() for k in ['license', 'notice', 'readme', 'copyright'])
            if not selected or not member.isfile():
                continue
            target = safe_path(ROOT / 'assets', name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as inp, target.open('wb') as out:
                shutil.copyfileobj(inp, out)
    report = {'source_revision': SOURCE_REV, 'asset_revision': ASSET_REV, 'source': source_info, 'assets': asset_info, 'upstream_lfs_sha256': expected, 'archive_members': manifest}
    (ROOT / 'reports' / 'asset_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'archive_members'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
