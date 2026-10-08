"""Minimal LeRobot v2.1 episode reader for complete measured-future windows.

Only pyarrow and PyAV are optional real-data dependencies. Video frames are
decoded on demand into bounded, episode-local caches; no acquisition or
normalizer logic belongs here.
"""

from collections import OrderedDict
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F

from .action_codec import ActionCodec
from .schema import ActionLabelSource, DataConfig, TrainingSample


def pose6d_rotvec_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Convert [..., xyz, rotation-vector] to SE(3) using Rodrigues."""
    value = np.asarray(pose, dtype=np.float64)
    if value.shape[-1:] != (6,) or not np.isfinite(value).all():
        raise ValueError("TCP pose must be finite [...,6] xyz + rotation vector")
    vector = value[..., 3:6]
    theta2 = np.sum(vector * vector, axis=-1)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    # The series avoids cancellation and division by zero near the identity.
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    a[small] = 1 - theta2[small] / 6 + theta2[small] ** 2 / 120
    b[small] = 0.5 - theta2[small] / 24 + theta2[small] ** 2 / 720
    np.divide(np.sin(theta), theta, out=a, where=~small)
    np.divide(1 - np.cos(theta), theta2, out=b, where=~small)
    x, y, z = (vector[..., i] for i in range(3))
    skew = np.zeros(value.shape[:-1] + (3, 3), dtype=np.float64)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    rotation = np.eye(3) + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)
    result = np.broadcast_to(np.eye(4), value.shape[:-1] + (4, 4)).copy()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = value[..., :3]
    if not np.allclose(rotation @ np.swapaxes(rotation, -1, -2), np.eye(3), atol=1e-8) or not np.allclose(
        np.linalg.det(rotation), 1, atol=1e-8
    ):
        raise ValueError("TCP rotation vector did not produce SO(3)")
    return result


def assemble_agent_pos(joint_position: np.ndarray, state: np.ndarray) -> np.ndarray:
    joints = np.asarray(joint_position, dtype=np.float32)
    measured = np.asarray(state, dtype=np.float32)
    if joints.shape[-1:] != (6,) or measured.shape[-1:] != (13,) or joints.shape[:-1] != measured.shape[:-1]:
        raise ValueError("Expected aligned joint_position [...,6] and observation.state [...,13]")
    if not np.isfinite(joints).all() or not np.isfinite(measured).all():
        raise ValueError("Robot state must be finite")
    gripper = measured[..., 6:7]
    if np.any((gripper < -1e-4) | (gripper > 1 + 1e-4)):
        raise ValueError("Measured normalized gripper is outside [0,1]")
    return np.concatenate((joints, gripper), axis=-1)


def reconstruct_tactile_depth(rgb: np.ndarray) -> torch.Tensor:
    """H.264 decoded RGB depth visualization -> policy-normalized depth."""
    value = np.asarray(rgb)
    if value.ndim != 4 or value.shape[1:] != (288, 384, 3) or value.dtype != np.uint8:
        raise ValueError("Tactile RGB must be uint8 [T,288,384,3]")
    depth = (value[..., 0].astype(np.float32) - value[..., 2].astype(np.float32)) / 255.0
    if not np.isfinite(depth).all() or np.any((depth < -1) | (depth > 1)):
        raise ValueError("Reconstructed tactile depth is outside [-1,1]")
    return torch.from_numpy(depth[:, None].copy())


def resize_rgb_frames(rgb: np.ndarray) -> torch.Tensor:
    """RGB HWC uint8 -> RGB CHW uint8 at 240x320; bilinear antialiased."""
    value = np.asarray(rgb)
    if value.ndim != 4 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("RGB frames must be uint8 [T,H,W,3]")
    frames = torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float()
    resized = F.interpolate(frames, size=(240, 320), mode="bilinear", align_corners=False, antialias=True)
    return resized.round().clamp_(0, 255).to(torch.uint8)


def complete_anchor_indices(frame_count: int, config: DataConfig) -> range:
    if frame_count < config.observation_history + config.action_horizon:
        raise ValueError("Episode needs at least 19 frames for a complete window")
    first = config.observation_history - 1
    stop = frame_count - config.action_horizon
    return range(first, stop)


def build_training_sample(
    *, episode_id: str, anchor_index: int, timestamps: np.ndarray, state: np.ndarray,
    joint_position: np.ndarray, external_rgb: np.ndarray, wrist_rgb: np.ndarray,
    tactile_rgb_0: np.ndarray, tactile_rgb_1: np.ndarray, data_config: DataConfig,
) -> TrainingSample:
    """Assemble one complete window from aligned numeric data and three video frames."""
    if data_config.tactile_representation_source != "legacy_video_reconstructed":
        raise ValueError("RGB depth-video reconstruction requires legacy_video_reconstructed provenance")
    if anchor_index not in complete_anchor_indices(len(timestamps), data_config):
        raise IndexError(f"anchor_index {anchor_index} does not have a complete observation/future window")
    if len(state) != len(timestamps) or len(joint_position) != len(timestamps):
        raise ValueError("Numeric columns must have identical episode lengths")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Aligned timestamps must be finite and increasing")
    obs = slice(anchor_index - 2, anchor_index + 1)
    future = slice(anchor_index + 1, anchor_index + 17)
    agent_pos = assemble_agent_pos(joint_position[obs], state[obs])
    future_gripper = assemble_agent_pos(joint_position[future], state[future])[:, 6:7]
    anchor_pose = pose6d_rotvec_to_matrix(state[anchor_index, :6])
    future_poses = pose6d_rotvec_to_matrix(state[future, :6])
    target = ActionCodec().encode_chunk(anchor_pose, future_poses, future_gripper)
    return TrainingSample(
        episode_id=episode_id,
        observation_timestamps=torch.as_tensor(timestamps[obs].copy(), dtype=torch.float32),
        external_rgb=resize_rgb_frames(external_rgb),
        wrist_rgb=resize_rgb_frames(wrist_rgb),
        tactile_depth_0=reconstruct_tactile_depth(tactile_rgb_0),
        tactile_depth_1=reconstruct_tactile_depth(tactile_rgb_1),
        agent_pos=torch.from_numpy(agent_pos.copy()),
        anchor_tcp_pose=torch.from_numpy(anchor_pose.copy()),
        target_action=torch.from_numpy(target.copy()),
        action_label_source=ActionLabelSource.MEASURED_FUTURE,
        action_history=None,
    )


class _VideoReader:
    def __init__(self, path: Path, *, frame_count: int, fps: int, size: tuple[int, int], cache_size: int) -> None:
        try:
            import av
        except ImportError as exc:
            raise ImportError("LeRobot video reading requires the realdata extra: pip install .[realdata]") from exc
        self.path = path
        self.container = av.open(str(path))
        if not self.container.streams.video:
            self.close()
            raise ValueError(f"No video stream: {path}")
        self.stream = self.container.streams.video[0]
        rate = self.stream.average_rate or self.stream.base_rate
        if rate is None or abs(float(rate) - fps) > 1e-6:
            self.close()
            raise ValueError(f"Video fps mismatch: {path}")
        if (self.stream.width, self.stream.height) != size:
            self.close()
            raise ValueError(f"Video size mismatch: {path}")
        if self.stream.frames and self.stream.frames != frame_count:
            self.close()
            raise ValueError(f"Video frame count mismatch: {path}")
        self.fps = fps
        self.frame_count = frame_count
        self.cache_size = cache_size
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def _remember(self, index: int, rgb: np.ndarray) -> None:
        self.cache[index] = rgb
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def read_frames(self, indices: tuple[int, ...]) -> np.ndarray:
        if not indices or min(indices) < 0 or max(indices) >= self.frame_count:
            raise IndexError("Video frame index outside episode")
        missing = [i for i in indices if i not in self.cache]
        if missing:
            first, last = min(missing), max(missing)
            seek_at = int((first / self.fps) / float(self.stream.time_base))
            self.container.seek(seek_at, stream=self.stream, backward=True)
            found: set[int] = set()
            for frame in self.container.decode(self.stream):
                if frame.pts is None:
                    raise ValueError(f"Video frame has no PTS: {self.path}")
                index = round(float(frame.pts * frame.time_base) * self.fps)
                if index < first:
                    continue
                if index > last:
                    break
                if index in found:
                    raise ValueError(f"Duplicate video frame index {index}: {self.path}")
                found.add(index)
                self._remember(index, frame.to_ndarray(format="rgb24"))
                if len(found) == last - first + 1:
                    break
            if any(i not in self.cache for i in missing):
                raise ValueError(f"Missing video frames {missing}: {self.path}")
        for index in indices:
            self.cache.move_to_end(index)
        return np.stack([self.cache[i] for i in indices])

    def close(self) -> None:
        self.container.close()


class LeRobotV21EpisodeAdapter:
    """Read one specified converted episode; get_window expects an anchor frame index."""

    def __init__(
        self, dataset_root: str | Path, episode_index: int, data_config: DataConfig, *, frame_cache_size: int = 64,
    ) -> None:
        if data_config.tactile_representation_source != "legacy_video_reconstructed":
            raise ValueError("This adapter requires legacy_video_reconstructed tactile provenance")
        if frame_cache_size < 3:
            raise ValueError("frame_cache_size must be at least 3 frames per stream")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset root does not exist: {self.dataset_root}")
        self.episode_index = int(episode_index)
        self.data_config = data_config
        info = self._json("meta/info.json")
        if info.get("codebase_version") != "v2.1":
            raise ValueError("Expected LeRobot v2.1 metadata")
        source_fps = float(info["fps"])
        if source_fps != data_config.control_frequency_hz:
            raise ValueError(f"Dataset fps {source_fps} != configured {data_config.control_frequency_hz}")
        self.fps = data_config.control_frequency_hz
        episodes = self._jsonl("meta/episodes.jsonl")
        matching = [item for item in episodes if item.get("episode_index") == self.episode_index]
        if len(matching) != 1:
            raise ValueError(f"Episode index {self.episode_index} is absent or duplicated")
        episode = matching[0]
        self.episode_length = int(episode["length"])
        self.valid_anchor_indices = complete_anchor_indices(self.episode_length, data_config)
        manifest_path = self.dataset_root / "meta/conversion_manifest.jsonl"
        if manifest_path.exists():
            matching_manifest = [item for item in self._jsonl("meta/conversion_manifest.jsonl")
                                 if item.get("episode_index") == self.episode_index]
            if len(matching_manifest) != 1 or matching_manifest[0].get("verified") is not True:
                raise ValueError(f"Episode {self.episode_index} has no verified conversion manifest entry")
        chunk = self.episode_index // int(info["chunks_size"])
        data_path = self.dataset_root / info["data_path"].format(episode_chunk=chunk, episode_index=self.episode_index)
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing episode Parquet: {data_path}")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("LeRobot Parquet reading requires the realdata extra: pip install .[realdata]") from exc
        columns = ["observation.joint_position", "observation.state", "timestamp", "frame_index", "action"]
        schema = pq.ParquetFile(data_path).schema_arrow
        if "source.ros_timestamp_ns" in schema.names:
            columns.append("source.ros_timestamp_ns")
        missing = set(columns) - set(schema.names)
        if missing:
            raise ValueError(f"Missing Parquet columns: {sorted(missing)}")
        table = pq.read_table(data_path, columns=columns)
        if table.num_rows != self.episode_length:
            raise ValueError("Parquet row count differs from episode metadata")
        self.state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        self.joint_position = np.asarray(table["observation.joint_position"].to_pylist(), dtype=np.float32)
        self.timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32).reshape(-1)
        self.lerobot_action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        self.ros_timestamp_ns = (
            np.asarray(table["source.ros_timestamp_ns"].to_pylist(), dtype=np.int64).reshape(-1)
            if "source.ros_timestamp_ns" in columns else None
        )
        frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64).reshape(-1)
        if not np.array_equal(frame_index, np.arange(self.episode_length)):
            raise ValueError("Parquet frame_index is not 0..episode_length-1")
        if self.state.shape != (self.episode_length, 13) or self.joint_position.shape != (self.episode_length, 6):
            raise ValueError("Parquet robot state has the wrong shape")
        if self.lerobot_action.shape != (self.episode_length, 7) or not np.isfinite(self.lerobot_action).all():
            raise ValueError("Parquet LeRobot action has the wrong shape or non-finite values")
        assemble_agent_pos(self.joint_position, self.state)
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Parquet timestamps must be finite and increasing")
        serial_mapping, tactile_paths = self._tactile_paths(episode)
        video_template = info["video_path"]
        paths = {
            "external_rgb": self.dataset_root / video_template.format(
                episode_chunk=chunk, episode_index=self.episode_index, video_key="observation.images.external_rgb"),
            "wrist_rgb": self.dataset_root / video_template.format(
                episode_chunk=chunk, episode_index=self.episode_index, video_key="observation.images.wrist_rgb"),
            **tactile_paths,
        }
        features = info["features"]
        external_shape = features["observation.images.external_rgb"]["shape"]
        wrist_shape = features["observation.images.wrist_rgb"]["shape"]
        sizes = {
            "external_rgb": (int(external_shape[2]), int(external_shape[1])),
            "wrist_rgb": (int(wrist_shape[2]), int(wrist_shape[1])),
            "tactile_depth_0": (384, 288),
            "tactile_depth_1": (384, 288),
        }
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError(f"Missing video: {[str(path) for path in paths.values() if not path.is_file()]}")
        self._readers: dict[str, _VideoReader] = {}
        try:
            for name, path in paths.items():
                self._readers[name] = _VideoReader(
                    path, frame_count=self.episode_length, fps=self.fps, size=sizes[name], cache_size=frame_cache_size
                )
        except Exception:
            self.close()
            raise
        self.episode_id = str(episode.get("episode_uuid", f"episode_{self.episode_index:06d}"))
        self.metadata: Mapping[str, object] = MappingProxyType({
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "episode_count": len(episodes),
            "episode_length": self.episode_length,
            "fps": self.fps,
            "valid_window_count": len(self.valid_anchor_indices),
            "external_rgb_size": sizes["external_rgb"],
            "wrist_rgb_size": sizes["wrist_rgb"],
            "tactile_size": sizes["tactile_depth_0"],
            "tactile_serial_mapping": MappingProxyType(serial_mapping),
            "tactile_representation_source": data_config.tactile_representation_source,
            "action_label_source": data_config.action_label_source.value,
        })

    def _json(self, relative: str) -> dict:
        path = self.dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _jsonl(self, relative: str) -> list[dict]:
        path = self.dataset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing LeRobot metadata: {path}")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _tactile_paths(self, episode: dict) -> tuple[dict[str, str], dict[str, Path]]:
        mapping = episode.get("tactile_auxiliary_video_mapping", {})
        serials = episode.get("serials", {})
        if all(key in mapping and "depth" in mapping[key] and key in serials for key in ("tactile_0", "tactile_1")):
            ordered = [(key, str(serials[key]), self.dataset_root / mapping[key]["depth"])
                       for key in ("tactile_0", "tactile_1")]
            if any(path.parent.name != serial for _, serial, path in ordered):
                raise ValueError("Tactile depth video path disagrees with metadata serial")
        else:
            depth_root = self.dataset_root / "tactile_videos/depth"
            found = sorted(path for path in depth_root.iterdir() if path.is_dir()) if depth_root.is_dir() else []
            if len(found) != 2:
                raise ValueError("Expected exactly two tactile depth serial directories")
            ordered = [(f"tactile_{i}", path.name, path / f"episode_{self.episode_index:06d}.mp4")
                       for i, path in enumerate(found)]
        if len({serial for _, serial, _ in ordered}) != 2:
            raise ValueError("Tactile sensor serials must be distinct")
        return (
            {f"tactile_depth_{i}": serial for i, (_, serial, _) in enumerate(ordered)},
            {f"tactile_depth_{i}": path for i, (_, _, path) in enumerate(ordered)},
        )

    def __len__(self) -> int:
        return len(self.valid_anchor_indices)

    def get_window(self, anchor_index: int) -> TrainingSample:
        if anchor_index not in self.valid_anchor_indices:
            raise IndexError(f"anchor_index {anchor_index} is outside complete windows {self.valid_anchor_indices}")
        indices = tuple(range(anchor_index - 2, anchor_index + 1))
        frames = {name: reader.read_frames(indices) for name, reader in self._readers.items()}
        return build_training_sample(
            episode_id=self.episode_id, anchor_index=anchor_index, timestamps=self.timestamps,
            state=self.state, joint_position=self.joint_position,
            external_rgb=frames["external_rgb"], wrist_rgb=frames["wrist_rgb"],
            tactile_rgb_0=frames["tactile_depth_0"], tactile_rgb_1=frames["tactile_depth_1"],
            data_config=self.data_config,
        )

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "LeRobotV21EpisodeAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
