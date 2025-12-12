import sys
import nbformat
from pathlib import Path

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
        except Exception as e:
            print(f"✗ {notebook_path} failed validation: {e}")
            failed = True

    if failed:
        sys.exit(1)
    else:
        print("All notebooks passed nbformat validation.")

if __name__ == "__main__":
    main()
