"""Quick validation: execute both notebooks and report errors."""
import subprocess, sys, os

os.chdir('/Users/stevenhill/Work/EORC/Courses/cubes_and_clouds_python/eo_python_foundations/notebooks')

notebooks = [
    '01_python_basics_and_jupyter_v4_course_ready.ipynb',
    '02_data_analysis_basics_v2_toolbox_eo.ipynb',
]

for nb in notebooks:
    print(f'\n{"="*60}')
    print(f'RUNNING: {nb}')
    print('='*60)
    r = subprocess.run(
        [sys.executable, '-m', 'jupyter', 'nbconvert',
         '--to', 'notebook', '--execute',
         '--ExecutePreprocessor.timeout=120',
         '--output', f'/tmp/test_{nb}',
         nb],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f'  ✅ PASSED')
    else:
        print(f'  ❌ FAILED (exit code {r.returncode})')
        print(f'  STDERR: {r.stderr[-2000:]}')
    if r.stdout.strip():
        print(f'  stdout: {r.stdout[:500]}')

print('\nDone.')
