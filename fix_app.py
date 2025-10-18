#!/usr/bin/env python3
"""Fix the app.py indentation issues"""

import re

# Read the file
with open('app.py.backup', 'r') as f:
    content = f.read()

# The problem is lines 140-157 and 167+ have wrong indentation
# We need to find the generation section and results section and fix them

# Split by the key marker
parts = content.split('if st.session_state.generation_result is not None:')

if len(parts) == 2:
    before = parts[0]
    after = parts[1]

    # Fix the 'after' part by removing excess indentation
    lines = after.split('\n')
    fixed_lines = []
    for line in lines:
        # If line starts with many spaces (more than 12), reduce to 8
        if line.startswith('                    '):  # 20 spaces
            fixed_lines.append('        ' + line.lstrip())
        else:
            fixed_lines.append(line)

    # Reconstruct
    fixed_after = '\n'.join(fixed_lines)
    fixed_content = before + 'if st.session_state.generation_result is not None:' + fixed_after

    # Write back
    with open('app.py', 'w') as f:
        f.write(fixed_content)

    print("Fixed app.py")
else:
    print("Could not find the marker")
