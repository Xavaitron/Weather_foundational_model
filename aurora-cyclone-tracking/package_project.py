"""Create a small transfer bundle without data, checkpoints or the local venv."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="../output/advection_finetuning.zip")
    args = p.parse_args()
    root, output = Path(__file__).resolve().parent, Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = list(root.glob("*.py"))+list(root.glob("*.md"))+list(root.glob("*.txt"))+list(root.glob("*.sh"))
    for directory in ("advection_ft", "configs", "tests"):
        candidates += [p for p in (root/directory).rglob("*") if p.is_file() and p.suffix in (".py", ".json")]
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(candidates):
            archive.write(path, Path(root.name)/path.relative_to(root))
    print(f"Bundled {len(candidates)} source/config/test files: {output}")


if __name__ == "__main__":
    main()
