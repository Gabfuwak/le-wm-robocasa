import os

os.environ["MUJOCO_GL"] = "egl"

try:
    import robocasa  # registers robocasa gym envs
except ImportError:
    pass

import time
from pathlib import Path

import imageio
import gymnasium as gym
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from functools import partial
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.world import MegaWrapper, SyncWorld, VariationWrapper, _make_env


def _episode_col(dataset):
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


class FlatActionWrapper(gym.ActionWrapper):
    """Flattens RoboCasa's Dict action space into a flat Box for CEM planning.

    Maps flat 7-dim actions [eef_pos(3), eef_rot(3), gripper(1)] to the
    RoboCasa Dict action space. Base motion and control mode are zeroed out.
    Must be applied BEFORE MegaWrapper so EverythingToInfoWrapper sees flat actions.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def action(self, action: np.ndarray) -> dict:
        return {
            "action.end_effector_position": action[0:3],
            "action.end_effector_rotation": action[3:6],
            "action.gripper_close": action[6:7],
            "action.base_motion": np.zeros(4, dtype=np.float32),
            "action.control_mode": np.zeros(1, dtype=np.float32),
        }


class RoboCasaWorld(swm.World):
    """World subclass that applies FlatActionWrapper before MegaWrapper.

    stable_worldmodel's EverythingToInfoWrapper (inside MegaWrapper) raises
    NotImplementedError if info['action'] is a dict. FlatActionWrapper must
    be innermost so MegaWrapper sees a flat ndarray action.
    """

    def __init__(
        self,
        env_name,
        num_envs,
        image_shape,
        goal_transform=None,
        image_transform=None,
        seed=2349867,
        history_size=1,
        frame_skip=1,
        max_episode_steps=100,
        **kwargs,
    ):
        wrappers = [
            FlatActionWrapper,
            partial(
                MegaWrapper,
                image_shape=image_shape,
                pixels_transform=image_transform,
                goal_transform=goal_transform,
                history_size=history_size,
                frame_skip=frame_skip,
                separate_goal=True,
            ),
        ]
        env_fn = partial(_make_env, env_name, max_episode_steps, wrappers, **kwargs)
        self.envs = VariationWrapper(SyncWorld([env_fn] * num_envs))
        self.envs.unwrapped.autoreset_mode = gym.vector.AutoresetMode.DISABLED

        self._history_size = history_size
        self.policy = None
        self.states = None
        self.infos = {}
        self.rewards = None
        self.terminateds = None
        self.truncateds = None
        self.seed = seed
        self._sticky_infos: dict = {}  # keys to preserve across env steps

    def _rebuild_proprio(self) -> None:
        """Reconstruct `proprio` in self.infos from individual env state keys."""
        eef_pos = self.infos.get("state.end_effector_position_relative")
        eef_rot = self.infos.get("state.end_effector_rotation_relative")
        gripper = self.infos.get("state.gripper_qpos")
        if eef_pos is not None and eef_rot is not None and gripper is not None:
            self.infos["proprio"] = np.concatenate([eef_pos, eef_rot, gripper], axis=-1)

    def _rebuild_pixels_eih(self) -> None:
        """Expose eye-in-hand camera obs as `pixels_eih` for dual-camera encoding."""
        eih = self.infos.get("video.robot0_eye_in_hand")
        if eih is not None:
            self.infos["pixels_eih"] = eih

    def step(self) -> None:
        """Step envs, then restore proprio (and goal) which env doesn't output."""
        for k in list(self.infos.keys()):
            if k.startswith("goal"):
                self._sticky_infos[k] = self.infos[k]

        super().step()
        self._rebuild_proprio()
        self._rebuild_pixels_eih()
        self.infos.update(self._sticky_infos)


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = _episode_col(dataset)
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def run_fresh_eval(world, dataset, cfg, eval_episodes, eval_start_idx, results_path):
    """Fresh-reset eval: env resets to its own random state, only the goal comes from the dataset."""
    eval_budget = cfg.eval.eval_budget
    goal_offset = cfg.eval.goal_offset_steps
    history_size = cfg.world.history_size

    successes = np.zeros(cfg.eval.num_eval, dtype=bool)
    results_path.mkdir(parents=True, exist_ok=True)

    for i, (ep_id, start_step) in enumerate(zip(eval_episodes.tolist(), eval_start_idx.tolist())):
        # --- sample goal frame from dataset ---
        goal_row_data = dataset.load_chunk(
            np.array([ep_id]), np.array([start_step + goal_offset]), np.array([start_step + goal_offset + 1])
        )
        goal_pixels = goal_row_data[0]["pixels"][-1].permute(1, 2, 0).numpy()  # (H, W, C) uint8
        goal_proprio = goal_row_data[0]["proprio"][-1].numpy()  # (9,)

        # broadcast goal to history shape: (1, history, H, W, C)
        goal_pixels_hist = np.broadcast_to(
            goal_pixels[None, None], (1, history_size, *goal_pixels.shape)
        ).copy()
        goal_proprio_hist = np.broadcast_to(
            goal_proprio[None, None], (1, history_size, *goal_proprio.shape)
        ).copy()

        sticky = {"goal": goal_pixels_hist, "goal_proprio": goal_proprio_hist}

        if "pixels_eih" in goal_row_data[0]:
            goal_pixels_eih = goal_row_data[0]["pixels_eih"][-1].permute(1, 2, 0).numpy()
            sticky["goal_pixels_eih"] = np.broadcast_to(
                goal_pixels_eih[None, None], (1, history_size, *goal_pixels_eih.shape)
            ).copy()

        # --- fresh env reset ---
        world.reset()
        world._rebuild_proprio()
        world._rebuild_pixels_eih()
        world._sticky_infos = sticky
        world.infos.update(world._sticky_infos)

        ep_frames = []
        ep_success = False

        for _ in range(eval_budget):
            # record frame — convert to channel-last uint8
            def to_uint8_hwc(f):
                if isinstance(f, torch.Tensor):
                    f = f.numpy()
                if f.dtype != np.uint8:
                    f = (f * 255).clip(0, 255).astype(np.uint8)
                if f.ndim == 3 and f.shape[0] in (1, 3):
                    f = f.transpose(1, 2, 0)
                return f

            frame = to_uint8_hwc(world.infos["pixels"][0, -1])
            if "pixels_eih" in world.infos:
                import cv2
                eih = to_uint8_hwc(world.infos["pixels_eih"][0, -1])
                h = frame.shape[0]
                eih = cv2.resize(eih, (h, h))
                frame = np.concatenate([frame, eih], axis=1)
            ep_frames.append(frame)

            world.step()

            if world.terminateds is not None and world.terminateds[0]:
                ep_success = True
                break

        successes[i] = ep_success

        video_path = results_path / f"eval_ep{ep_id}_step{start_step}.mp4"
        imageio.mimwrite(str(video_path), ep_frames, fps=10)
        print(f"Eval {i}: ep={ep_id} start={start_step} success={ep_success} frames={len(ep_frames)} video={video_path}")

    return {
        "success_rate": float(successes.mean()),
        "episode_successes": successes,
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    img_size = cfg.eval.img_size
    world = RoboCasaWorld(**cfg.world, image_shape=(img_size, img_size))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "pixels_eih": img_transform(cfg),
        "goal": img_transform(cfg),
        "goal_pixels_eih": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = _episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = np.sort(
        valid_indices[g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)]
    )

    print(random_episode_indices)

    row_data = dataset.get_row_data(random_episode_indices)
    eval_episodes = row_data[col_name]
    eval_start_idx = row_data["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = run_fresh_eval(world, dataset, cfg, eval_episodes, eval_start_idx, results_path)
    end_time = time.time()

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
