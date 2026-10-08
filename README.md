# Visuotactile Flow Policy

Independent repository for the data, action, encoder, condition, Flow Action
Expert, and Stage-1 policy contracts of a future UR5e visuotactile policy.
It contains a CFM objective and Euler inference sampler, but no training loop
or robot deployment implementation.

## Flow core contract

The Stage-1 standard CFM contract is defined in
`configs/flow/standard_cfm.yaml`: a Gaussian source, linear conditional
path, `NOISE_AT_ZERO`, and uniform model time in [0,1]. Source noise lives at
t=0 and the clean action target at t=1. The alternative
`configs/flow/openpi_style.yaml` uses `NOISE_AT_ONE` and scaled Beta time
sampling as a time/direction recipe; this model is not OpenPI. The two paths
are equivalent under time reversal, but a trained checkpoint's convention
must not change mid-run.

`FlowConvention` owns source/target endpoints and integration direction;
`LinearConditionalFlowPath` owns interpolation and target velocity.
`FlowTensorSpec` lets `GaussianSource` draw noise without a clean target
during inference. Uniform sampling accepts a seeded `torch.Generator`.
`BetaTimeSampler` currently uses the global PyTorch RNG and rejects a
generator argument; optional time complementation must be explicit.
Timestep features use float32 normalized-time sin/cos with periods from
0.004 to 4.0 before the learned MLP.

The Standard Conditional Flow Matching core now has an objective and an
Euler solver. The objective accepts **already normalized action** `a`
from a future `ActionCodec → StructuredActionNormalizer` training chain.
It does not encode, normalize, or denormalize actions. With `NOISE_AT_ZERO`:

```text
z ~ N(0,I), t ~ Uniform(0,1)
x_t = (1-t)z + ta
u_t = a-z
loss = mean((v_theta(x_t,t,C)-u_t)^2)
```

An optional boolean `action_valid_mask[B,H]` excludes invalid steps from
the loss. Its denominator is `valid_step_count * D`; it does not alter the
path sample. Inference starts with Gaussian noise and uses explicit Euler
from t=0 to t=1:

```text
x_(k+1) = x_k + (t_(k+1)-t_k) * v_theta(x_k,t_k,C)
```

The Euler result remains a **normalized action**. The Stage-1 policy core
denormalizes it to fixed-anchor 10D encoded action. A future deployment
boundary must decode it with `ActionCodec` and form robot commands.
`NOISE_AT_ONE` uses the same objective and solver with reversed endpoints
and velocity sign. Both YAML files declare an initial 10-step Euler recipe;
10 expert forwards are not a 30 Hz real-time guarantee. Timing must later
include the full encoders, condition adapter, expert × NFE, IPC, and robot
command path. The old deployment's skipped first four actions have no
confirmed latency explanation.

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
Flow Action Expert attends to these tokens.

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
TCP/gripper targets separately. **TrainingDataContract v1** fixes the training
label to `measured_future`; no commanded-target reader exists. Observations
are aligned frames `t-2,t-1,t` at 30 Hz. The anchor is measured TCP at `t`;
the target is measured future `t+1...t+16`, encoded as one fixed-anchor
`[16,10]` chunk. State is six measured joint angles (radians) plus measured
normalized gripper position `[0,1]`, giving seven dimensions.
`TrainingSample` reserves optional `action_history` for later work.

The tactile representation risk is recorded in the model contract:

```yaml
legacy_tactile_train_representation: h264_decoded_rgb_difference_over_255
legacy_tactile_runtime_representation: raw_sdk_depth_clip_over_0p7
equivalence: approximate_unquantified
```

PNG channel quantization is bounded, but H.264
and calibration differences have not been measured with paired frames.
`DataConfig.tactile_representation_source` records either
`raw_sdk_normalized` or `legacy_video_reconstructed`, both declared as
`policy_normalized` at the encoder boundary. Before new training starts,
choose a canonical tactile representation. Prefer SDK depth normalized once
when lossless raw depth is available. If only legacy visualization video is
available, use `(R-B)/255` for training and assess deployment representation
matching. This stage does not change the robot runtime.

**IMPORTANT: Normalization statistics MUST be fitted using TRAIN SPLIT ONLY.
Never fit on validation/test data.** Old DP checkpoint statistics must not be
used for the newly collected dataset. The legacy `MinMaxNormalizer` remains
version 1. The new `normalization` package supplies identity, mean/std,
min/max v2, quantile, and physical fixed-range methods. Structured action
normalization splits translation 3 / rotation 6 / gripper 1; structured state
normalization splits joints 6 / gripper 1. Both serialize train-only statistics,
floor flags, modes, model contract, and tactile provenance. The three action
and two state YAML files are experimental candidates, not final choices.
`legacy_limits` exactly follows the old Diffusion Policy `limits` formula:
for span below `range_eps`, scale is 1 and offset is `-input_min` when the
output is `[-1,1]`. Small variation is preserved. `minmax` means the newer
V2 safe midpoint-collapse method: it sends near-constant channels to the
output midpoint and restores their training center on inverse transformation.
These modes have different serialized types and must not be interchanged.
Mean/std and quantile use configurable floors.
The `fit_split=train` metadata field is a caller contract, not proof that
`fit()` received training frames. Enforcing the split remains the future
training/data pipeline's responsibility.
`analyze_action_distribution` reports channel tails, horizon profiles, and
Gaussian-source flow velocity scale. The older `MinMaxNormalizer` fits the
last channel over `[N,H,D]` or `[N,T,D]`, maps ordinary training limits to
`[-1,1]`, handles constant channels, and stores JSON-friendly version-1
statistics. The new methods do not reinterpret that format.

`FlowTensorSpec(shape, device, dtype)` defines action-shaped allocation without
a clean target. `GaussianSource.sample(spec, condition=None, history=None,
generator=None)` draws from it and accepts a seeded `torch.Generator`.
Training may construct a spec from a target tensor; inference constructs it
from batch size, horizon, dimension, device, and dtype. No A2A implementation
is present.

## Local checks

Use the existing Python environment; package installation is unnecessary:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 /home/pine/openpi/ros2_teleop_dataset/vision_RL/offline_RL/.venv/bin/python -m pytest -q
```

## Flow Action Expert

The first `FlowActionExpert` is a standard Transformer / DiT-style action
backbone: 14 blocks, width 1024, 16 heads of width 64, FFN ratio 4, and
bidirectional self-attention over 16 action steps of dimension 10. Blocks
2, 4, 6, 8, 10, 12, and 14 additionally cross-attend to condition tokens.
Flow time enters self-attention and FFN through zero-initialized AdaLN-Zero
modulation; vision, tactile, and state information enters through unmodulated
cross-attention. The FFN uses `GELU(approximate="tanh")`. The action output
head is zero-initialized, so a fresh model returns exactly zero velocity.
Condition length `N` is variable; each sample needs at least one valid token.

The verified Action Expert has **297,309,194 parameters**. The 14-layer
setting replaces the earlier unverified 20-layer estimate, which omitted the
per-block AdaLN-Zero modulation from its rough count. With frozen encoders,
the parameter inventory is:

| Component | Parameters | Trainable |
| --- | ---: | ---: |
| Vision encoders (2 × 11,176,512) | 22,353,024 | 0 |
| Tactile encoders (2 × 11,170,240) | 22,340,480 | 0 |
| Condition Adapter | 2,650,112 | 2,650,112 |
| Flow Action Expert | 297,309,194 | 297,309,194 |
| Total policy | 344,652,810 | 299,959,306 |

The DiT-style architecture remains only `v_theta(x_t, t, C)`. The CFM
objective and Euler solver are separate modules; there is no training loop
or robot deployment. Run
`python tools/inspect_flow_expert.py` for the live parameter breakdown;
`--smoke` additionally checks full-model CUDA bf16 inference where supported.

## Stage-1 policy and checkpoint

`PolicyObservationBatch` takes three policy-facing frames: uint8 RGB
`[B,3,3,240,320]` for each camera, normalized tactile depth
`[B,3,1,288,384]` for each sensor, and measured `agent_pos[B,3,7]`.
Every modality shares the device. Optional boolean masks `[B,3]` mark
attention-valid tokens. **A false mask is not NaN sanitization**: invalid
frames must still contain finite placeholder or reused values, because
encoders run before attention masking. Tactile depth is already in roughly
`[-1,1]`; SDK depth conversion belongs at the data or deployment boundary.

`VisuotactileFlowPolicy.encode_condition()` sends both RGB streams to
`VisionEncoder`, both depth streams to `TactileEncoder`, and normalizes
`agent_pos` with `StructuredStateNormalizer` before `ConditionAdapter`.
The Adapter keeps its time-major 15-token layout. For training,
`compute_training_loss()` accepts an already encoded, fixed-anchor 10D
action chunk, normalizes it with `StructuredActionNormalizer`, then calls
the CFM objective and Expert. Neither the policy nor CFM calls
`ActionCodec`. Both normalizers must be fitted or loaded beforehand.

For inference, call `policy.eval()` and `sample_encoded_action()`. Gaussian
noise and Euler accumulation use float32 state and time. The Expert may
perform matmuls under an outer mixed-precision autocast context; its
velocity is cast back to float32 before the Euler update. The policy
denormalizes the final action and returns a fixed-anchor encoded 10D
chunk. It does not decode absolute robot poses or issue commands. This
float32 ODE contract is an initial numerical choice, not a performance claim.

Checkpoint format v1 saves the complete policy model state (all four
encoders, Adapter, and Expert), separate fitted action/state normalizer
states, the model/flow/data/normalization config bundle, and optional
dataset fingerprint, encoder provenance, source commit, and notes.
Loading into a compatible preconstructed policy checks version and config,
then restores model and normalizer states. The resulting checkpoint does
not depend on legacy `DP.ckpt` or standalone encoder weights. Those weights
are only an optional initialization source. Inspect the default module
inventory with `python tools/inspect_policy_core.py`; this does not run
inference or require encoder export files.

## Synthetic learning gate

`python tools/overfit_synthetic.py --device cuda` exercises the complete
Stage-1 learning path with a **tiny test policy**. Fixed pixel projections
replace the four ResNet branches, while the real ConditionAdapter, CFM
objective, FlowActionExpert, normalizers, and Euler solver remain in use.
Only the 3,648-parameter Adapter and 318,794-parameter tiny Expert are
optimized (322,442 parameters total). The generated eight-example training
set requires all five modality groups and horizon position to determine the
target. Outputs go to the Git-ignored `outputs/synthetic_overfit/` directory.

With seed 20261008 on CUDA, Phase A ran 600 steps: fixed CFM loss fell from
1.715405 to 0.001021, and fixed-source Euler endpoint RMSE was 0.023418.
Fresh-policy Phase B ran 1500 steps: fixed CFM loss fell from 1.937774 to
0.024335, while seeded Euler RMSE fell from 1.384683 to 0.067808.
Correct-condition RMSE was 0.065474 versus 1.480839 with cyclically
shuffled observations and identical Gaussian noise. Both gates passed;
the trained tiny-policy checkpoint reproduced its seeded sample after
load. These results verify CFM learning plumbing, Euler endpoint
improvement, and use of conditioning. They do not validate convergence of
the 297M model, real robot performance, or the real data distribution.
