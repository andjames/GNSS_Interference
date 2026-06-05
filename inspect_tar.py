"""
Inspect the contents of a downloaded tar file to see exact filenames
"""
import tarfile
import sys
from pathlib import Path

# Find any downloaded tar file
data_dir = Path("./gnss_data")
tars = list(data_dir.rglob("*.tar"))

if not tars:
    print("No tar files found in ./gnss_data/")
    sys.exit(1)

# Inspect the first one found
tar_path = tars[0]
print(f"Inspecting: {tar_path}")
print(f"Size: {tar_path.stat().st_size / 1024 / 1024:.1f} MB\n")

with tarfile.open(tar_path, 'r') as tf:
    members = tf.getmembers()
    print(f"Files inside ({len(members)} total):")
    for m in members[:20]:
        print(f"  {m.name}  ({m.size / 1024:.0f} KB)")
    if len(members) > 20:
        print(f"  ... and {len(members)-20} more")
