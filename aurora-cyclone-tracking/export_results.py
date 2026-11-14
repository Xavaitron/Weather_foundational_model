import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    p = argparse.ArgumentParser(description="Bundle metrics/configs/logs for copying back; checkpoints are optional.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--output", default="advection_results.zip")
    p.add_argument("--include-checkpoints", action="store_true")
    args = p.parse_args()
    root, output = Path(args.runs).resolve(), Path(args.output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(output)
    extensions = {".json", ".jsonl", ".csv", ".log", ".txt"}
    if args.include_checkpoints:
        extensions.add(".pt")
    files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink() and p.suffix in extensions]
    if not files:
        raise ValueError("No run metrics/configs/logs found.")
    with ZipFile(output, "w", ZIP_DEFLATED) as z:
        for path in files:
            z.write(path, Path("runs")/path.relative_to(root))
    print(f"Exported {len(files)} files to {output}")


if __name__ == "__main__":
    main()
