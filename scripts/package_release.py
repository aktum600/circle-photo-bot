"""Package only explicitly approved source files, never workspace secrets."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    '.dockerignore', '.gitignore', '.env.example', 'Dockerfile',
    'README.md', 'THIRD_PARTY.md', 'requirements.txt', 'render.yaml',
    'scripts/export_model.py', 'scripts/package_release.py',
]
files = [ROOT / name for name in FILES]
for folder, pattern in [('bot', '*.py'), ('tests', '*.py'), ('docs', '*.md')]:
    files.extend(sorted((ROOT / folder).glob(pattern)))
destination = ROOT / 'runtime' / 'release' / 'circle-photo-bot.zip'
destination.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed: {path.name}')
        archive.write(path, path.relative_to(ROOT).as_posix())
with ZipFile(destination) as archive:
    assert archive.testzip() is None
    print(f'Prepared {len(archive.namelist())} source files: {destination}')
