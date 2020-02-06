# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Format notebook code cells using yapf google style."""
import glob
import nbformat
import yapf

# Must be run from the top level of the `TFQuantum` repo.
NOTEBOOKS = glob.glob("docs/tutorials/*.ipynb")
for fname in NOTEBOOKS:
    nb = nbformat.read(fname, as_version=nbformat.NO_CONVERT)
    all_cells = nb.get('cells')
    for i, cell in enumerate(all_cells):
        if cell.get('cell_type') != 'code':
            continue
        lines = cell.get('source')
        # This will safely skip over cells containing !% magic
        try:
            fmt_lines = yapf.yapf_api.FormatCode(''.join(lines),
                                                 style_config="google")[0]
        except SyntaxError:
            continue
        # google style always adds an EOF newline; undo this.
        all_cells[i]['source'] = fmt_lines[:-1]

    nb['cells'] = all_cells
    nbformat.write(nb, fname, version=nbformat.NO_CONVERT)
