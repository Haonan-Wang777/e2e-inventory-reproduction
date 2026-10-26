"""Audit that released source files were preserved byte for byte."""
from __future__ import annotations

import hashlib
import json
import platform
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ZIP = Path(os.environ.get('E2E_AUTHOR_ZIP', str(ROOT / 'author_package.zip')))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    files = []
    if ORIGINAL_ZIP.exists():
        with zipfile.ZipFile(ORIGINAL_ZIP) as archive:
            for item in archive.infolist():
                parts = PurePosixPath(item.filename).parts
                if item.is_dir() or any(x.startswith('.') or x.startswith('__MACOSX') or x.startswith('~$') for x in parts):
                    continue
                relative = Path(*parts[1:])
                target = ROOT / 'author_package' / relative
                expected = hashlib.sha256(archive.read(item)).hexdigest()
                actual = digest(target)
                if actual != expected:
                    raise AssertionError(f'Original author file changed: {relative}')
                files.append(dict(file=relative.as_posix(), bytes=target.stat().st_size, sha256=actual,
                                  identical_to_archive=True))
        archive_record = dict(path=str(ORIGINAL_ZIP), bytes=ORIGINAL_ZIP.stat().st_size,
                              sha256=digest(ORIGINAL_ZIP), available_for_recheck=True)
    else:
        # A moved project can validate against its saved manifest without the external zip.
        prior = json.loads((out / 'provenance.json').read_text(encoding='utf-8'))
        for item in prior['author_files']:
            target = ROOT / 'author_package' / item['file']
            if digest(target) != item['sha256']:
                raise AssertionError(f'Author source hash changed: {target}')
        files = prior['author_files']
        archive_record = dict(prior['archive'], available_for_recheck=False)
    names = ['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'statsmodels', 'torch', 'pypdf']
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  archive=archive_record, author_files=files,
                  preserved_author_file_count=len(files),
                  paper=dict(file='paper/source_paper.pdf',
                             sha256=digest(ROOT / 'paper/source_paper.pdf') if (ROOT / 'paper/source_paper.pdf').exists()
                             else json.loads((out / 'provenance.json').read_text(encoding='utf-8'))['paper']['sha256'],
                             available_for_recheck=(ROOT / 'paper/source_paper.pdf').exists()),
                  environment=dict(python=sys.version, executable=sys.executable,
                                   platform=platform.platform(), packages=packages),
                  scripts={str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'scripts').glob('*.py'))})
    (out / 'provenance.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'PASS: {len(files)} author files match the supplied archive byte for byte')


if __name__ == '__main__':
    main()
