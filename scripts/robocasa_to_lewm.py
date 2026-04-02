#!/usr/bin/env python3
"""
Convert RoboCasa LeRobot dataset to LeWM HDF5 format.

Reads parquet files (action/state) and MP4s (images) directly — no LeRobot import.

Output schema required by stable_worldmodel HDF5Dataset:
  ep_len    (E,)            int64
  ep_offset (E,)            int64
  pixels    (T, H, W, 3)   uint8
  action    (T, 7)          float32
  proprio   (T, 9)          float32

Field layout verified against meta/modality.json for PickPlaceCounterToCabinet/20250819:
  action[5:12]  eef_pos(3) + eef_rot(3) + gripper(1)   <- arm control only
  state[7:16]   eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)

If you use a different task or dataset version, confirm slices by inspecting
meta/modality.json in the downloaded lerobot directory.

Usage:
    python scripts/robocasa_to_lewm.py --dataset-path /path/to/lerobot --output ~/data/robocasa/robocasa_pickplace_train.h5
"""

import argparse
import json
import h5py
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ACTION_SLICE = slice(5, 12)  # eef_pos(3) + eef_rot(3) + gripper(1); verified vs modality.json
PROPRIO_SLICE = slice(7, 16) # eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2); verified vs modality.json


def read_video_frames(mp4_path: Path, img_size: int) -> np.ndarray:
    """Read all frames from an MP4 sequentially. Returns (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != img_size or frame.shape[1] != img_size:
            frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def main(args):
    dataset_path = Path(args.dataset_path).expanduser()

    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    chunks_size = info["chunks_size"]
    total_frames = info["total_frames"]

    episodes = sorted(
        [json.loads(l) for l in (dataset_path / "meta" / "episodes.jsonl").open()],
        key=lambda e: e["episode_index"],
    )
    ep_lengths = np.array([e["length"] for e in episodes], dtype=np.int64)
    ep_offsets = np.zeros(len(episodes), dtype=np.int64)
    ep_offsets[1:] = np.cumsum(ep_lengths[:-1])

    camera_key = f"observation.images.{args.camera}"
    assert camera_key in info["features"], \
        f"Camera '{camera_key}' not in {list(info['features'])}"

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Episodes: {len(episodes)},  Frames: {total_frames},  Output: {output_path}")

    with h5py.File(output_path, "w") as f:
        f.create_dataset("ep_len", data=ep_lengths)
        f.create_dataset("ep_offset", data=ep_offsets)
        pixels_ds = f.create_dataset(
            "pixels", shape=(total_frames, args.img_size, args.img_size, 3),
            dtype=np.uint8, chunks=(32, args.img_size, args.img_size, 3),
        )
        action_ds  = f.create_dataset("action",  shape=(total_frames, 7), dtype=np.float32)
        proprio_ds = f.create_dataset("proprio", shape=(total_frames, 9), dtype=np.float32)

        for ep in tqdm(episodes, desc="Episodes"):
            ep_idx = ep["episode_index"]
            chunk  = ep_idx // chunks_size
            start, end = ep_offsets[ep_idx], ep_offsets[ep_idx] + ep_lengths[ep_idx]

            vpath = dataset_path / "videos" / f"chunk-{chunk:03d}" / camera_key / f"episode_{ep_idx:06d}.mp4"
            frames = read_video_frames(vpath, args.img_size)
            assert len(frames) == ep_lengths[ep_idx], \
                f"Episode {ep_idx}: expected {ep_lengths[ep_idx]} frames, got {len(frames)}"
            pixels_ds[start:end] = frames

            ppath = dataset_path / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
            df = pd.read_parquet(ppath, columns=["action", "observation.state"])
            action_ds[start:end]  = np.stack(df["action"].to_numpy())[:, ACTION_SLICE].astype(np.float32)
            proprio_ds[start:end] = np.stack(df["observation.state"].to_numpy())[:, PROPRIO_SLICE].astype(np.float32)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Path to LeRobot-format RoboCasa dataset (e.g. deps/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceCounterToCabinet/20250819/lerobot)")
    parser.add_argument("--output",       default="~/data/robocasa/robocasa_pickplace_train.h5")
    parser.add_argument("--img-size",     type=int, default=96)
    parser.add_argument("--camera",       default="robot0_agentview_left")
    args = parser.parse_args()
    main(args)
