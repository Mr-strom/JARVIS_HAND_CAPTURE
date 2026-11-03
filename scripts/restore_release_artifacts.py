"""Restore reviewed distribution artifacts to the frozen application's paths.

This is local packaging support, not publication permission. Never overwrites a
different existing calibration, checkpoint or dataset file.
"""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'release_artifacts/MANIFEST.json').read_text())
    for entry in manifest['artifacts']:
        source=(ROOT/entry['release_path']).resolve();destination=(ROOT/entry['runtime_path']).resolve()
        if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT):raise ValueError('Artifact path escapes project')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Release artifact checksum mismatch')
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest()!=entry['sha256']:raise FileExistsError(f'Refusing to replace existing {entry["runtime_path"]}')
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
    print(f'Restored/verified {len(manifest["artifacts"])} approved sample artifacts locally; personal-data reuse terms are separate.')

if __name__=='__main__':main()
