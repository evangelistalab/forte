#!/usr/bin/env python

import subprocess
from pathlib import Path

p = Path('notebooks/')
for file in p.glob('**/*.ipynb'):
    # skip checkpoint files
    if 'checkpoint' in file.name:
        continue
    print(f'Converting notebook {file}')    
    output_dir = Path('source') / Path(*file.parts[1:-1])
    cmd = ['jupyter-nbconvert','--to','rst' ,'--output-dir', f'{output_dir}', f'{file}']    
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    