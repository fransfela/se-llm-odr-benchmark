"""Create a clean zip of everything Overleaf needs."""

import zipfile
from pathlib import Path
from datetime import datetime


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"emnlp2026_industry_{timestamp}.zip"
    zip_path = Path(zip_name)

    paper = Path("paper")
    required = [
        paper / "emnlp2026_industry.tex",
        paper / "acl.sty",
        paper / "acl_natbib.bst",
        paper / "custom.bib",
    ]

    missing = [f for f in required if not f.exists()]
    if missing:
        print(f"MISSING files: {missing}")
        print("Cannot create zip — fix missing files first.")
        return

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in required:
            zf.write(f, f.name)
            print(f"  Added: {f.name}")

        for fig in sorted((paper / "figures").glob("*.pdf")):
            zf.write(fig, f"figures/{fig.name}")
            print(f"  Added: figures/{fig.name}")
        for fig in sorted((paper / "figures").glob("*.png")):
            zf.write(fig, f"figures/{fig.name}")

        for tbl in sorted((paper / "tables").glob("*.tex")):
            zf.write(tbl, f"tables/{tbl.name}")
            print(f"  Added: tables/{tbl.name}")

    size_kb = zip_path.stat().st_size / 1024
    print(f"\nCreated: {zip_name} ({size_kb:.1f} KB)")
    print("Upload to Overleaf: New Project > Upload Project > select zip")


if __name__ == "__main__":
    main()
