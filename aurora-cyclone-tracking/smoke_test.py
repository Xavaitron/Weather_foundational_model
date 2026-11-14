"""CPU test of the actual Aurora encoder/backbone/decoder; no downloads."""
import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"]))
