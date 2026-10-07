# Visuotactile Flow Policy

Independent repository for the data and action contracts of a future UR5e
visuotactile flow policy. This first stage contains no model, training loop,
sampler, encoder wrapper, or robot deployment implementation.

## Confirmed robot contract

The earlier USB tactile Diffusion Policy consumes three observations at about
30 Hz: external RGB, wrist RGB, two tactile depth images, and a seven-value
`agent_pos` vector. The vector contains six actual UR5e joint angles in radians
followed by measured gripper position divided by 255. External and wrist RGB
come from separate ResNet18 encoders; tactile depth 0 and 1 likewise have
separate single-channel ResNet18 encoders. Each branch yields 512 values per
frame. This repository does not yet load those weights.

RGB acquisition is external 848×480 and wrist 640×480. Online processing
converts BGR to RGB and resizes both to 320×240. The policy converts uint8 to
float in [0,1], resizes to 224×224, takes a 216×216 center crop at inference,
then applies ImageNet mean/std normalization. Each tactile worker emits
`clip(SDK_depth / 0.7, -1, 1)` at `[1,288,384]`; the policy resizes depth to
224×224 without the RGB crop or ImageNet normalization.

These details were checked against the existing repository at
`/home/pine/openpi/ros2_teleop_dataset/vision_RL/offline_RL/usb_insertion/`
and the sibling `tactile_RL` implementation. They document compatibility;
this package does not import or copy the old implementation or checkpoints.

## Action geometry

`ActionSpec` fixes `H=16`, `D=10`. Each step stores three metres of translation,
six rotation values, and one absolute normalized gripper target. For one
latest-observation anchor `T_a` and each future target `T_i`:

```text
T_delta_i = inverse(T_a) @ T_i
delta_p_i = R_a.T @ (p_i - p_a)
delta_R_i = R_a.T @ R_i
rotation6d_i = delta_R_i[:2, :].reshape(6)  # first TWO ROWS, row-major
action_i = [delta_p_i, rotation6d_i, gripper_target_i]
```

`ActionCodec` accepts homogeneous `[4,4]` poses, or matching batches. It uses
the same anchor for every step. Decoding normalizes rotation row 1, removes its
projection from row 2, normalizes row 2, and takes their cross product as row
3. This row-wise Gram-Schmidt procedure matches the old `from_delta` geometry
and projects finite imperfect network outputs back onto SO(3). Degenerate
rows are rejected. Gripper values are returned unchanged. Neither translation
nor rotation deltas accumulate across a chunk.

The old DP uses a 16-step outward action chunk. Its deployment configuration
currently skips indices 0–3, but the reason was not established. The new
robot configuration therefore leaves `action.start_index: null` pending a
latency and closed-loop experiment.

## Data and normalization

The `Episode` schema keeps measured TCP/gripper state and optional commanded
TCP/gripper targets separately. `DataConfig.action_label_source` accepts
`measured_future` or `commanded_target`. The default `measured_future` is a
candidate only: new acquisition should save both, and a formal experiment
must choose the training label. `TrainingSample` specifies three observation
frames and a `[16,10]` encoded target. It reserves optional `action_history`
for later work, without using it now.

**IMPORTANT: Normalization statistics MUST be fitted using TRAIN SPLIT ONLY.
Never fit on validation/test data.** Old DP checkpoint statistics must not be
used for the newly collected dataset. `MinMaxNormalizer` fits the last channel
over `[N,H,D]` or `[N,T,D]`, maps ordinary training limits to `[-1,1]`, handles
constant channels, and stores JSON-friendly statistics. `FeatureNormalizer`
leaves room for later quantile or mean/std implementations. The choice of
min/max is provisional.

`FlowSource.sample(target_action, condition=None, history=None,
generator=None)` defines an action-shaped source. Only `GaussianSource` exists
today; it follows the target's shape, dtype, and device and accepts a seeded
`torch.Generator`. No A2A implementation is present.

## Local checks

Use the existing Python environment; package installation is unnecessary:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /home/pine/openpi/ros2_teleop_dataset/vision_RL/offline_RL/.venv/bin/python -m pytest -q
```

The model settings under `configs/model/` are an **UNVALIDATED PARAMETER
TARGET**, not a verified 300M parameter count. Parameter size must be counted
after the model is implemented.
