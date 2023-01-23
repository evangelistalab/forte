from collections import defaultdict

header  = """.. _`sec:options`:

List of Forte options
=====================

.. sectionauthor:: Francesco A. Evangelista
"""

import forte

options = forte.forte_options.dict()

grouped_options = defaultdict(list)
for k, v in options.items():
    grouped_options[v['group']].append((k,v))

groups = sorted(grouped_options.keys())

lines = []
for g in groups:
    label = 'General' if len(g) == 0 else g
    head = f'{label} options'
    lines.append(f"\n{head}\n{'=' * len(head)}")
    opts = sorted(grouped_options[g])
    for label, descr in opts:
        lines.append(f'\n**{label}**')
        lines.append(f"\n{descr['description']}")
        lines.append(f"\nType: {descr['type']}")
        lines.append(f"\nDefault value: {descr['default_value']}")
        lines.append(f"\nAllowed values: {descr['allowed_values']}") if 'allowed_values' in descr else None
            
content = '\n'.join(lines)

with open('source/options.rst','w+') as f:
    f.write(f"{header}\n{content}")