from pathlib import Path

import pytest
import torch
import yaml

from visuotactile_flow.data.dataset import load_data_config
from visuotactile_flow.data.schema import ActionLabelSource, ActionSpec, DataConfig, Episode, TrainingSample


ROOT = Path(__file__).resolve().parents[1]


def test_action_and_observation_dimensions():
    spec = ActionSpec()
    config = load_data_config(ROOT / "configs/data/default.yaml")
    assert (spec.horizon, spec.action_dim) == (16, 10)
    assert (config.observation_history, config.action_horizon, config.action_dim) == (3, 16, 10)


def test_action_label_source_enum_and_config():
    assert DataConfig(action_label_source="commanded_target").action_label_source is ActionLabelSource.COMMANDED_TARGET
    with pytest.raises(ValueError):
        DataConfig(action_label_source="ambiguous")


def test_robot_start_index_unset_and_model_target_labeled():
    robot = yaml.safe_load((ROOT / "configs/robot/ur5e.yaml").read_text())
    assert robot["action"]["start_index"] is None
    model_path = ROOT / "configs/model/flow_300m.yaml"
    assert "UNVALIDATED PARAMETER TARGET" in model_path.read_text()
    model = yaml.safe_load(model_path.read_text())
    assert (model["action_horizon"], model["action_dim"]) == (16, 10)


def test_episode_preserves_both_target_channels_and_provenance():
    n = 2
    episode = Episode(
        episode_id="e1", task_id="usb", timestamps=torch.tensor([0.0, 1.0]),
        external_rgb=torch.zeros(n, 3, 240, 320), wrist_rgb=torch.zeros(n, 3, 240, 320),
        tactile_depth_0=torch.zeros(n, 1, 288, 384), tactile_depth_1=torch.zeros(n, 1, 288, 384),
        agent_pos=torch.zeros(n, 7), tcp_pose=torch.eye(4).repeat(n, 1, 1),
        gripper_state=torch.zeros(n, 1), commanded_tcp_target=torch.eye(4).repeat(n, 1, 1),
        commanded_gripper_target=torch.ones(n, 1), source_timestamp={"robot": torch.zeros(n)},
        source_age={"robot": torch.zeros(n)}, frame_reused={"robot": torch.zeros(n, dtype=torch.bool)},
        action_source=("human", "policy"),
    )
    assert episode.commanded_tcp_target is not None
    assert episode.action_source[0].value == "human"


def test_training_sample_has_reserved_action_history():
    sample = TrainingSample(
        episode_id="e1", observation_timestamps=torch.arange(3),
        external_rgb=torch.zeros(3, 3, 2, 2), wrist_rgb=torch.zeros(3, 3, 2, 2),
        tactile_depth_0=torch.zeros(3, 1, 2, 2), tactile_depth_1=torch.zeros(3, 1, 2, 2),
        agent_pos=torch.zeros(3, 7), anchor_tcp_pose=torch.eye(4),
        target_action=torch.zeros(16, 10), action_label_source="measured_future",
        action_history=torch.zeros(5, 10),
    )
    assert sample.action_history.shape == (5, 10)
