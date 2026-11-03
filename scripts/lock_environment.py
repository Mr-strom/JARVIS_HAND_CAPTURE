"""Freeze installed versions and hash-lock resolved PyPI wheels from pip's report."""
import importlib.metadata as meta
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT/'reports/pip_install.json').read_text(encoding='utf-8'))
lines = ['# Windows x64 / CPython 3.11; exact wheels resolved in Phase 1.', '# somehand source is installed separately from its verified pinned archive.']
for entry in sorted(report['install'], key=lambda item:item['metadata']['name'].lower()):
    name = entry['metadata']['name']
    version = entry['metadata']['version']
    assert meta.version(name) == version, name
    digest = entry['download_info']['archive_info']['hashes']['sha256']
    lines.append(f'{name}=={version} --hash=sha256:{digest}')
(ROOT/'requirements.lock').write_text('\n'.join(lines)+'\n', encoding='utf-8')
freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
(ROOT/'reports/installed_versions.txt').write_text(freeze,encoding='utf-8')
print(f'Locked {len(report["install"])} PyPI distributions; somehand provenance recorded separately.')
