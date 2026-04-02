#!/usr/bin/env python3
"""Download RoboCasa PickPlaceCounterToCabinet demonstrations."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DEPS = REPO_ROOT / "deps"

# Prefer submodule installs over system packages
for dep in ("robosuite", "robocasa"):
    dep_path = DEPS / dep
    if dep_path.exists():
        sys.path.insert(0, str(dep_path))

try:
    from robocasa.scripts.download_datasets import download_datasets
    from robocasa.utils.dataset_registry_utils import get_ds_path
except ImportError:
    sys.exit(
        "RoboCasa not found. Either:\n"
        "  git submodule update --init --recursive\n"
        "  cd deps/robosuite && pip install -e . && cd ../robocasa && pip install -e ."
    )

TASK = "PickPlaceCounterToCabinet"

# Note: RoboCasa's downloader outputs LeRobot-format data (parquet + MP4 videos),
# not the legacy robosuite HDF5 format. The downloaded path ends in .../lerobot/.
# scripts/robocasa_to_lewm.py reads this LeRobot layout directly.
download_datasets(tasks=[TASK], split=["pretrain"], source=["human"])

dataset_path = Path(get_ds_path(TASK, source="human", split="pretrain"))
if dataset_path.exists():
    print(f"\nRaw dataset: {dataset_path}")
    print(f"\nNext: convert to LeWM format:")
    print(f"  python scripts/robocasa_to_lewm.py --dataset-path {dataset_path}")
    print(f"\nThen set:")
    print(f"  export STABLEWM_HOME=~/data/robocasa")
else:
    sys.exit(f"Download failed: {dataset_path} not found")
