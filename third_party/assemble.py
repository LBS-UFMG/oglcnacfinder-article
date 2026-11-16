"""Assemble the scripts that contain code of LM-OGlcNAc-Site and Stack-OglyPred-PLM.

Neither repository has a licence, so their code is not stored here. Four scripts of this study run
their code with our changes (tagged [modified] / [added] / [removed]); this repository keeps them as
templates (*.template.py) in which every line taken from those repositories is replaced by a
reference. This script clones both repositories at the commits used in the study, fills in the
references and writes the runnable scripts next to the templates (same name without '.template').
Each assembled script is checked against the SHA-256 of the script that was run.

Template syntax:
  #@ <repository> <file> <first>-<last> shift=<n>
      lines <first> to <last> (1-based) of <file>, each indented by n more spaces (n < 0: fewer);
      blank lines stay blank.
  {{<repository> <file> <line>}}
      inside a line: the text of that line of <file>, without its indentation.

usage: python third_party/assemble.py        (from the repository root; needs git)
"""
import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOS = {
    'LM-OGlcNAc-Site': ('https://github.com/KCLabMTU/LM-OGlcNAc-Site.git',
                        'ff032df049c75812a9ad070580c60ea19cb8d941'),
    'Stack-OglyPred-PLM': ('https://github.com/PakhrinLab/Stack-OglyPred-PLM.git',
                           'f9934d21144ebe34a110c0668904972e532b947c'),
}
# SHA-256 of each assembled script (UTF-8, LF line endings), filled in by the release build
TEMPLATES = {
    'analysis/06_benchmark_LM_OGlcNAc_Site.template.py':
        '3bcea9cfbd2c178e41cdc748b6530eece228d2bd54f96daf72a0c98c17d94ad3',
    'scripts/lm_oglcnac_site_external.template.py':
        'd504f7a79ab1c84866577477f2d2a9eaac9ad2c94d8db912dfbe28b1fbccc4c8',
    'analysis/10_benchmark_Stack-OglyPred-PLM.template.py':
        '17c1c599ca90fc5a4bf3891c4a8aa20c1c585a875637488ec796276f5512f573',
    'scripts/stack_oglypred_plm_long_proteins.template.py':
        '08c94186f855ddbaec5b452aa387976454e8115942a7ceaa6f0f4559303d1178',
}
RANGE = re.compile(r'^#@ (\S+) (.+?) (\d+)-(\d+) shift=(-?\d+)$')
INLINE = re.compile(r'\{\{(\S+) (.+?) (\d+)\}\}')


def fetch(name):
    url, commit = REPOS[name]
    path = ROOT / 'third_party' / name
    if not path.exists():
        subprocess.run(['git', 'clone', '--quiet', url, str(path)], check=True)
    subprocess.run(['git', '-C', str(path), 'checkout', '--quiet', commit], check=True)
    head = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()
    assert head == commit, f'{name}: at {head}, expected {commit}'
    return path


def assemble(template, sources):
    def lines_of(repo, file):
        key = (repo, file)
        if key not in sources:
            sources[key] = (fetch(repo) / file).read_text(encoding='utf-8').splitlines()
        return sources[key]

    out = []
    for line in template.read_text(encoding='utf-8').splitlines():
        m = RANGE.match(line)
        if m:
            repo, file, first, last, shift = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
            for text in lines_of(repo, file)[first - 1:last]:
                if not text.strip():
                    out.append('')
                    continue
                indent = len(text) - len(text.lstrip(' ')) + shift
                assert indent >= 0, f'{template.name}: negative indentation at {file}:{first}-{last}'
                out.append(' ' * indent + text.strip())
            continue
        out.append(INLINE.sub(lambda g: lines_of(g.group(1), g.group(2))[int(g.group(3)) - 1].strip(), line))
    return '\n'.join(out) + '\n'


def main():
    sources = {}
    for rel, sha in TEMPLATES.items():
        template = ROOT / rel
        text = assemble(template, sources)
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        assert digest == sha, f'{rel}: assembled script differs from the one that was run'
        target = template.with_name(template.name.replace('.template.py', '.py'))
        with open(target, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        print(f'{target.relative_to(ROOT)}: assembled, identical to the script that was run')


if __name__ == '__main__':
    main()
