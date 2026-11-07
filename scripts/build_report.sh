#!/usr/bin/env bash
# Build the REVIEW-CANDIDATE report only. Never installs, publishes, submits, or runs experiments.
# Remote use: source scripts/project_env.sh before invoking this script.
# Only report/ and this script should be synced while other agents change the repo.
# All products, logs, snapshots, temporary files, and TeX caches stay in:
#   ../local-materials/report-build/ or ../submission/
set -euo pipefail
if [[ $# -gt 0 ]]; then
    printf '%s\n' 'Usage: bash scripts/build_report.sh (no arguments; fixed safe output paths)' >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'BLOCKED: python3 is required. No packages were installed.' >&2
    exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec python3 - "${BASH_SOURCE[0]}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

script = Path(sys.argv[1]).resolve()
research = script.parent.parent
project = research.parent
allowed = {
    Path('/Users/ssdd/academics/comp597'),
    Path('/home/ssdd/academics/comp597'),
}
if project not in allowed or research.name != 'research':
    sys.exit('BLOCKED: this build is confined to the authorized COMP 597 project roots.')
if platform.system() == 'Linux' and project != Path('/home/ssdd/academics/comp597'):
    sys.exit('BLOCKED: remote writes require /home/ssdd/academics/comp597.')

# Reject symlink escapes before making any output directory.
def confined_dir(path):
    if path.resolve() != path:
        sys.exit(f'BLOCKED: output path contains a symlink: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path

build_root = confined_dir(project / 'local-materials' / 'report-build')
submission = confined_dir(project / 'submission')
source = research / 'report'
bibliography = research / 'literature' / 'references.bib'
required = [source / 'main.tex', source / 'vendor' / 'IEEEtai.cls',
            source / 'vendor' / 'IEEEtran.bst', bibliography]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    sys.exit('BLOCKED: missing build inputs:\n' + '\n'.join(missing))

expected_vendor = {
    'IEEEtai.cls': 'e46d339baa0c9fbb9913bae0fd5bc6c3b382f08eb67761a24ce6a72570b42564',
    'IEEEtran.bst': '314f0ece704568faf827011bac498650691b2b5ee06320720830e782416d5a5f',
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for name, expected in expected_vendor.items():
    if sha(source / 'vendor' / name) != expected:
        sys.exit(f'BLOCKED: vendored {name} changed; preserve the supplied template and notices.')
original_class = project / 'local-materials' / 'course-template' / 'IEEEtai.cls'
if original_class.exists() and sha(original_class) != expected_vendor['IEEEtai.cls']:
    sys.exit('BLOCKED: the original course class differs from the pinned unchanged copy.')

# Reject manually changed or stale generated results before compiling.
generation_path = source / 'generated' / 'results-manifest.json'
if not generation_path.is_file():
    sys.exit('BLOCKED: run report/generate_results.py against the reviewed outputs first.')
generation = json.loads(generation_path.read_text())
if generation.get('scientific_results_review') != 'PASS' or generation.get('training_review') != 'PASS':
    sys.exit('BLOCKED: scientific and training evidence must have PASS reviews.')
if sha(source / 'generate_results.py') != generation['generator_sha256']:
    sys.exit('BLOCKED: report generator changed since generation; regenerate the inputs.')
for name, digest in generation['output_sha256'].items():
    target = source / 'generated' / name
    if Path(name).name != name or not target.is_file() or sha(target) != digest:
        sys.exit('BLOCKED: generated report input does not match its CSV-bound manifest: ' + name)

missing_tools = [name for name in ('pdflatex', 'bibtex', 'kpsewhich') if not shutil.which(name)]
if missing_tools:
    sys.exit('BLOCKED: missing tools: ' + ', '.join(missing_tools) +
             '. Supply them through an existing TeX installation or project-local setup; no global installs are performed.')

run = Path(tempfile.mkdtemp(prefix='review-candidate-', dir=build_root))
work = run / 'source'
work.mkdir()
manifest = {}
# Stage only report-owned inputs. Do not snapshot the changing experiment repo.
for path in sorted(source.rglob('*')):
    if path.is_symlink():
        sys.exit(f'BLOCKED: report source symlink is not an allowed build input: {path}')
    if not path.is_file():
        continue
    relative = path.relative_to(source)
    if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
        continue
    # Explicit allowlist excludes logs, caches, archives, and source-paper PDFs.
    # PDF figures are permitted only in the reserved generated-input directory.
    if path.suffix not in ('.tex', '.cls', '.bst', '.txt', '.py', '.json') and not (
        relative.parts[0] == 'generated' and path.suffix in ('.pdf', '.png', '.jpg', '.jpeg')
    ):
        continue
    target = work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    manifest['report/' + relative.as_posix()] = sha(target)
shutil.copyfile(bibliography, work / 'references.bib')
manifest['literature/references.bib'] = sha(work / 'references.bib')
shutil.copyfile(script, run / 'build_report.sh')
manifest['scripts/build_report.sh'] = sha(run / 'build_report.sh')

# Override inherited project caches for this build. No automatic font generation
# or shell escape is allowed; dependencies must already be installed.
env = os.environ.copy()
for name, suffix in {
    'TMPDIR': 'tmp', 'TMP': 'tmp', 'TEMP': 'tmp',
    'TEXMFVAR': 'texmf-var', 'TEXMFCONFIG': 'texmf-config',
    'TEXMFCACHE': 'texmf-cache', 'VARTEXFONTS': 'texmf-fonts',
    'XDG_CACHE_HOME': 'cache', 'XDG_CONFIG_HOME': 'config',
    'XDG_DATA_HOME': 'data',
}.items():
    directory = run / suffix
    directory.mkdir(exist_ok=True)
    env[name] = str(directory)
env.update({
    'TEXINPUTS': str(work / 'vendor') + os.pathsep + str(work) + os.pathsep,
    'BSTINPUTS': str(work / 'vendor') + os.pathsep,
    'BIBINPUTS': str(work) + os.pathsep,
    'TEXMFOUTPUT': str(work), 'MKTEXPK': '0', 'MKTEXTFM': '0',
    'openout_any': 'p', 'shell_escape': '0',
})

metadata = {
    'status': 'REVIEW-CANDIDATE BUILD STARTED; not experiment evidence or a submission',
    'started_utc': datetime.now(timezone.utc).isoformat(),
    'build_directory': str(run), 'source_sha256': manifest,
    'original_class_checked': original_class.exists(),
    'commands': [], 'tool_versions': {},
    'results_manifest_sha256': sha(generation_path),
    'results_generation': generation,
    'independent_report_audit': 'PENDING',
}
def save_metadata():
    (run / 'build-manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')

def fail(message):
    metadata['status'] = 'BLOCKED: ' + message
    save_metadata()
    sys.exit(f'{message}\nBuild logs: {run}\nNo new PDF was published. Any existing REVIEW-CANDIDATE PDF may be stale.')

for tool in ('pdflatex', 'bibtex'):
    result = subprocess.run([tool, '--version'], cwd=work, env=env,
                            capture_output=True, text=True)
    metadata['tool_versions'][tool] = result.stdout.splitlines()[:2]
save_metadata()

needed = []
for package in ('amsmath.sty', 'array.sty', 'graphicx.sty', 'cite.sty', 'hyperref.sty', 'ptmr7t.tfm'):
    found = subprocess.run(['kpsewhich', package], cwd=work, env=env,
                           capture_output=True, text=True)
    if found.returncode or not found.stdout.strip():
        needed.append(package)
if needed:
    detail = (' The missing ptmr7t.tfm belongs to the TeX Live times package (URW Base 35 fonts).'
              if 'ptmr7t.tfm' in needed else '')
    fail('Missing TeX dependencies: ' + ', '.join(needed) + '.' + detail +
         ' Supply dependencies project-locally or through an existing TeX installation. No install was attempted.')

latex = ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
         '-halt-on-error', '-file-line-error', '-recorder', 'main.tex']
commands = [('latex-1', latex), ('bibtex', ['bibtex', 'main']),
            ('latex-2', latex), ('latex-3', latex)]
for label, command in commands:
    metadata['commands'].append({'label': label, 'argv': command, 'cwd': str(work)})
    save_metadata()
    with (run / (label + '.txt')).open('w') as stream:
        result = subprocess.run(command, cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print((run / (label + '.txt')).read_text(errors='replace')[-7000:], file=sys.stderr)
        fail(f'{label} failed with exit {result.returncode}. See the saved log for the missing package or TeX error.')

log = (work / 'main.log').read_text(errors='replace')
if re.search(r'Citation .+ undefined|Reference .+ undefined|There were undefined references|Rerun to get cross-references right', log):
    fail('Unresolved citations or cross-references remain after the final LaTeX pass.')
blg = (work / 'main.blg').read_text(errors='replace')
if 'Warning--' in blg:
    fail('BibTeX reported a warning; inspect main.blg before exporting the REVIEW-CANDIDATE.')

# Verify the actual output recorder, not only the requested output directory.
for line in (work / 'main.fls').read_text(errors='replace').splitlines():
    if line.startswith('OUTPUT '):
        output = Path(line[7:])
        if not output.is_absolute():
            output = work / output
        try:
            output.resolve().relative_to(run)
        except ValueError:
            fail('TeX recorded an output outside this isolated build: ' + str(output))

aux = (work / 'main.aux').read_text(errors='replace')
page_match = re.search(r'\\newlabel\{report-body-end\}\{\{[^}]*\}\{(\d+)\}', aux)
body_pages = int(page_match.group(1)) if page_match else None
metadata['body_pages_excluding_references'] = body_pages
metadata['page_target_met'] = body_pages is not None and 4 <= body_pages <= 6
metadata['layout_warnings'] = [line for line in log.splitlines() if
                               'Overfull' in line or 'Underfull' in line]
if not metadata['page_target_met']:
    fail('Review candidate must have 4 to 6 body pages, excluding references.')
if any('Overfull' in line for line in metadata['layout_warnings']):
    fail('Review candidate contains overfull boxes; fix layout before exporting.')
metadata['status'] = 'REVIEW-CANDIDATE BUILD SUCCEEDED; independent report audit and submission approval pending'
metadata['pdf_sha256'] = sha(work / 'main.pdf')
metadata['finished_utc'] = datetime.now(timezone.utc).isoformat()
save_metadata()

# Atomic exports only after all build checks succeed. Keep previous artifacts on
# failure and explicitly warn that they may be stale. No final-named PDF is made.
def export(src, name):
    target = submission / name
    if target.is_symlink():
        fail('Refusing a symlink export destination: ' + str(target))
    fd, temporary = tempfile.mkstemp(prefix='.review-candidate-', dir=submission)
    os.close(fd)
    temporary = Path(temporary)
    try:
        shutil.copyfile(src, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

export(work / 'main.pdf', 'midproject-report-REVIEW-CANDIDATE.pdf')
export(run / 'build-manifest.json', 'midproject-report-REVIEW-CANDIDATE.build.json')
print('REVIEW-CANDIDATE BUILD SUCCEEDED. Scientific evidence is reviewed; report audit and submission approval remain pending.')
print('PDF: ' + str(submission / 'midproject-report-REVIEW-CANDIDATE.pdf'))
print('Manifest: ' + str(submission / 'midproject-report-REVIEW-CANDIDATE.build.json'))
print('Logs and isolated source snapshot: ' + str(run))
print('Body pages excluding references: ' + str(body_pages))
if not metadata['page_target_met']:
    print('PENDING: the final report must have 4 to 6 body pages, excluding references.')
print('Layout warning count: ' + str(len(metadata['layout_warnings'])))
print('PENDING: independent report audit, user-controlled GitHub access, and final submission approval. Previous DRAFT PDFs are not overwritten.')
PY
