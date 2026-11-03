"""Fetch pinned preassembled G1/Shadow assets, retaining composite model licenses."""
from pathlib import Path
import urllib.request,zipfile,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
REV='249acdb565b4040fbfff7d5265629d61ae82b5d0'
URL=f'https://codeload.github.com/darth-alexus/humanoid_shadow_g1/zip/{REV}'
archive=ROOT/'downloads/g1_shadow.zip';archive.parent.mkdir(exist_ok=True)
if not archive.exists():
    with urllib.request.urlopen(URL,timeout=90) as response,archive.open('wb') as out:shutil.copyfileobj(response,out)
target=ROOT/'assets/g1_shadow';files=[]
if hashlib.sha256(archive.read_bytes()).hexdigest()!='671300ff916197a4942ab010047198f47cd139ffae43b93502a3a44c1bd078ca':
    raise ValueError('Pinned G1 archive checksum mismatch')
with zipfile.ZipFile(archive) as z:
    for entry in z.infolist():
        relative='/'.join(entry.filename.split('/')[1:])
        if not relative or entry.is_dir():continue
        dest=(target/relative).resolve()
        if not dest.is_relative_to(target.resolve()):raise ValueError('Unsafe archive path')
        dest.parent.mkdir(parents=True,exist_ok=True);data=z.read(entry);dest.write_bytes(data)
        files.append({'path':relative,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
manifest={'repository':'darth-alexus/humanoid_shadow_g1','revision':REV,'url':URL,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':files}
(ROOT/'reports/g1_asset_manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copyfile(target/'LICENSE',ROOT/'licenses/g1-shadow-COMPOSITE.txt')
print(f'Fetched {len(files)} files, {sum(f["bytes"] for f in files)} bytes')
