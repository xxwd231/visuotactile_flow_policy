# Visuotactile Flow Policy

Independent repository for the data, action, encoder, and condition contracts
of a future UR5e visuotactile flow policy. It contains no Flow Expert,
training loop, inference sampler, or robot deployment implementation.

## Confirmed robot contract

The earlier USB tactile Diffusion Policy consumes three observations at about
30 Hz: external RGB, wrist RGB, two tactile depth images, and a seven-value
`agent_pos` vector. The vector contains six actual UR5e joint angles in radians
followed by measured gripper position divided by 255. External and wrist RGB
come from separate ResNet18 encoders; tactile depth 0 and 1 likewise have
separate single-channel ResNet18 encoders. Each branch yields 512 values per
frame. This repository loads exported standalone weights from the tactile DP.

RGB acquisition is external 848×480 and wrist 640×480. Online processing
converts BGR to RGB and resizes both to 320×240. The policy converts uint8 to
float in [0,1], resizes to 224×224, takes a 216×216 center crop at inference,
then applies ImageNet mean/std normalization. Each tactile worker emits
`clip(SDK_depth / 0.7, -1, 1)` at `[1,288,384]`; the policy resizes depth to
224×224 without the RGB crop or ImageNet normalization.

These details were checked against the existing repository at
`/home/pine/openpi/ros2_teleop_dataset/vision_RL/offline_RL/usb_insertion/`
and the sibling `tactile_RL` implementation. They document compatibility;
the runtime package does not import the old implementation or checkpoints.

## Reused encoders

The audited backbone is `torchvision.models.resnet18(weights=None)` with
`fc=Identity`. All `BatchNorm2d` modules are replaced at the same module names
with `GroupNorm(num_groups=num_channels//16, num_channels=num_channels)`;
PyTorch defaults give `eps=1e-5` and `affine=True`. Thus the 64/128/256/512
channel layers use 4/8/16/32 groups. A tactile branch replaces the RGB conv1
with a single-channel convolution initialized from the **mean** of the RGB
conv1 weights; loading the checkpoint then replaces that initialization with
the trained weights. There are no other backbone architecture changes in the
audited encoder path. Each RGB branch has 11,176,512 parameters; each tactile
branch has 11,170,240.

`VisionEncoder` and `TactileEncoder` each own two independent branches and
return a dictionary of separate `[B,T,512]` features. They accept any positive
history length `T`. They default to frozen branches. `freeze()` disables
gradients and keeps the branch in eval mode even when the parent is switched
to train mode; `unfreeze()` restores gradients. The new project must load
exported weights before using a frozen branch for training or inference.

RGB preprocessing assumes the acquisition layer has already converted BGR to
RGB and resized to 320×240. `RGBPreprocessor` requires RGB `uint8` in
`[B,T,3,H,W]` and applies `/255`, resize to 224×224, inference center crop to
216×216, and ImageNet mean/std normalization. `TactilePreprocessor` requires
`[B,T,1,288,384]` and an explicit `depth_encoding`: `raw_sdk_depth` applies
`clip(depth/0.7,-1,1)` once; `policy_normalized` requires input in `[-1,1]`.
Both modes then resize to 224×224 without RGB cropping or color normalization.
The encoding declaration belongs in dataset metadata/configuration. A value
mistakenly declared as raw cannot always be recognized as already normalized
from its numeric range alone, so collection must preserve this provenance.

## Condition tokens

The first condition representation uses 3 observation frames × (2 RGB + 2
tactile + 1 state) = **15 condition tokens**, each with **1024 dimensions**.
The public order is time-major: at each time step, external RGB, wrist RGB,
tactile depth 0, tactile depth 1, then agent position. Time 0 is the oldest
frame. The four encoder features remain separate before `ConditionAdapter`;
each has its own `LayerNorm(512) → Linear(512,1024)` projection. The already
normalized 7-D state uses `Linear(7,512) → SiLU → Linear(512,1024)`.
Learned modality and temporal embeddings are added before a final LayerNorm.
An optional boolean validity mask travels in the same token order, with
`True` meaning valid; repeated frames are not automatically marked invalid.
`ConditionAdapter` does not perform attention-based multimodal fusion. The
Flow Action Expert will later attend to these tokens.

Export portable weights from a trusted local old checkpoint without importing
the old repository into this package:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/pine/openpi/ros2_teleop_dataset/vision_RL/offline_RL/.venv/bin/python tools/export_legacy_encoders.py \
  --checkpoint /home/pine/openpi/ros2_teleop_dataset/tactile_RL/offline_RL/usb_insertion/models/epoch800/DP.ckpt
```

This writes four `.pt` files and `metadata.json` to
`artifacts/legacy_encoders/`, which Git ignores. Each `.pt` is a versioned
state dictionary and can be loaded independently with
`load_encoder_weights(branch, path, strict=True)`; the old `DP.ckpt` is not
needed on another machine. The export records source checkpoint size, upstream
commit, architecture, preprocessing, key prefixes, branch counts, and file
sizes. It deliberately does not export old action or state normalizer values.

Local parity verification is restricted to `tools/verify_legacy_encoder_parity.py`:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/pine/openpi/ros2_teleop_dataset/vision_RL/offline_RL/.venv/bin/python tools/verify_legacy_encoder_parity.py \
  --checkpoint /home/pine/openpi/ros2_teleop_dataset/tactile_RL/offline_RL/usb_insertion/models/epoch800/DP.ckpt
```

The check compares all four branches from policy-facing RGB uint8 at 320×240
and SDK-like depth converted once. It exercises both old and new preprocessing
and reports feature shape and errors at CPU float32 with `atol=1e-5`,
`rtol=1e-4`. The current four branches each have shape `[1,512]`, zero max
and mean absolute error, and pass `allclose`. This verifies the encoder path,
not camera acquisition, sensor calibration, GPU kernel parity, or the DP head.

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
