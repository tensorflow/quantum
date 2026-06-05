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
"""Validate all Jupyter notebooks in the repository using nbformat."""
import sys
from pathlib import Path

import nbformat
from nbformat.validator import NotebookValidationError


def main():
    """Check all notebooks for valid nbformat structure."""
    failed = False
    notebooks = list(Path('.').rglob('*.ipynb'))
    print(f"Found notebooks: {notebooks}")
    for notebook_path in notebooks:
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            nbformat.validate(nb)
            print(f"✓ {notebook_path}")
        except (NotebookValidationError, OSError) as e:
            print(f"✗ {notebook_path} failed validation: {e}")
            failed = True

    if failed:
        sys.exit(1)
    else:
        print("All notebooks passed nbformat validation.")


if __name__ == "__main__":
    main()
