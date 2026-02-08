import json

notebooks = [
    '04_stac_fundamentals.ipynb',
    '05_stac_xarray_satellite_data.ipynb',
    'A1_cloud_data_formats.ipynb',
    'A2_optional_dask_parallel.ipynb',
]
keywords = ['try it', 'checkpoint', 'exercise 1', 'exercise 2', 'exercise 3', 'exercise 4', 'try it:']

for nb_file in notebooks:
    with open(f'notebooks/{nb_file}') as f:
        nb = json.load(f)
    print(f'\n{"="*80}')
    print(f'NOTEBOOK: {nb_file}')
    print(f'{"="*80}')
    cells = nb['cells']
    for i, cell in enumerate(cells):
        src = ''.join(cell['source'])
        src_lower = src.lower()
        if cell['cell_type'] == 'markdown' and any(kw in src_lower for kw in keywords):
            # Skip table-of-contents cells and cells that just mention exercises in passing
            if 'table of contents' in src_lower:
                continue
            if 'skip ahead to the exercises' in src_lower:
                continue
            # Skip section headers like "## 10) Exercises" that don't have actual exercise content
            stripped = src.strip()
            if stripped in ['---\n\n## 10) Exercises', '## 10) Exercises', '---\n\n## 8) Exercises', '## 8) Exercises', '---\n\n## 7) Exercises', '## 7) Exercises']:
                continue
            if stripped.startswith('---') and 'Exercises' in stripped and len(stripped.split('\n')) <= 4:
                continue

            cell_id = cell.get('id', f'NO_ID_{i}')
            cell_num = i + 1  # 1-based for VSCode
            print(f'\n{"~"*60}')
            print(f'EXERCISE MARKDOWN: cell_num={cell_num} (0-idx={i}), json_id={cell_id}')
            print(f'{"~"*60}')
            for line in cell['source']:
                print(line, end='')
            print()

            # Print ALL following code cells until next markdown
            j = i + 1
            while j < len(cells) and cells[j]['cell_type'] == 'code':
                code_src = ''.join(cells[j]['source'])
                code_id = cells[j].get('id', f'NO_ID_{j}')
                code_num = j + 1
                print(f'\n  >> PAIRED CODE: cell_num={code_num} (0-idx={j}), json_id={code_id}')
                for line in cells[j]['source']:
                    print(f'  {line}', end='')
                print()
                j += 1

            # If the markdown had no code cell following, note it
            if i + 1 < len(cells) and cells[i+1]['cell_type'] != 'code':
                print(f'  >> (no code cell follows - next is markdown)')
