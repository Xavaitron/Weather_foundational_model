"""Save an ADS token privately from an interactive terminal, without echoing it."""
import getpass
import json
import os
from pathlib import Path
import sys


def main():
    if not sys.stdin.isatty():
        raise SystemExit("Run this yourself in an interactive SSH terminal. Do not send the token in chat.")
    destination = Path(".secrets/ads.json")
    if destination.exists():
        raise SystemExit("ADS configuration already exists; edit it privately to replace it.")
    print("Register/sign in at https://ads.atmosphere.copernicus.eu/how-to-api")
    print("Accept the CAMS EAC4 and global composition dataset terms on their download pages.")
    token = getpass.getpass("Paste your ADS personal access token (hidden): ").strip()
    if not token or any(c.isspace() for c in token):
        raise SystemExit("Token must be nonempty and contain no whitespace.")
    destination.parent.mkdir(mode=0o700, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"url": "https://ads.atmosphere.copernicus.eu/api", "key": token}, f)
    print("Saved private ADS configuration. The token was not printed.")


if __name__ == "__main__":
    main()
