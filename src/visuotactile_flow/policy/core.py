"""Stage-1 model-side wiring for normalized CFM training and Euler sampling."""

import torch
from torch import nn

from visuotactile_flow.condition import ConditionAdapter
from visuotactile_flow.condition.schema import ConditionOutput
from visuotactile_flow.data.schema import DataConfig
from visuotactile_flow.encoders import TactileEncoder
from visuotactile_flow.flow import (
    ConditionalFlowMatchingObjective, EulerSolver, FlowTensorSpec, GaussianSource,
    LinearConditionalFlowPath,
)
from visuotactile_flow.normalization import (
    StructuredActionNormalizer, StructuredStateNormalizer,
)

from .schema import PolicyObservationBatch, PolicySampleOutput, PolicyTrainingOutput


class VisuotactileFlowPolicy(nn.Module):
    """Dependency-injected policy core; ActionCodec and robot commands live outside."""

    def __init__(
        self, *, vision_encoder: nn.Module, tactile_encoder: nn.Module,
        condition_adapter: ConditionAdapter, flow_expert: nn.Module,
        action_normalizer: StructuredActionNormalizer,
        state_normalizer: StructuredStateNormalizer,
        cfm_objective: ConditionalFlowMatchingObjective,
        euler_solver: EulerSolver, data_config: DataConfig,
    ) -> None:
        super().__init__()
        if not isinstance(data_config, DataConfig):
            raise ValueError("data_config must be DataConfig")
        if (data_config.observation_history, data_config.action_horizon,
                data_config.action_dim, data_config.state_dim) != (3, 16, 10, 7):
            raise ValueError("Stage-1 requires Tobs=3, H=16, D=10 and state_dim=7")
        for name, module in (
            ("vision_encoder", vision_encoder), ("tactile_encoder", tactile_encoder),
            ("condition_adapter", condition_adapter), ("flow_expert", flow_expert),
        ):
            if not isinstance(module, nn.Module):
                raise ValueError(f"{name} must be an nn.Module")
        if (condition_adapter.state_dim != 7
                or condition_adapter.model_dim != flow_expert.hidden_dim):
            raise ValueError("ConditionAdapter state/model dimensions must match policy and Expert")
        if (flow_expert.action_horizon, flow_expert.action_dim) != (16, 10):
            raise ValueError("FlowActionExpert must output H=16, D=10")
        if not isinstance(action_normalizer, StructuredActionNormalizer):
            raise ValueError("action_normalizer must be StructuredActionNormalizer")
        if not isinstance(state_normalizer, StructuredStateNormalizer):
            raise ValueError("state_normalizer must be StructuredStateNormalizer")
        if (action_normalizer.data_config != data_config
                or state_normalizer.data_config != data_config):
            raise ValueError("Normalizer DataConfig must match policy DataConfig")
        if not isinstance(cfm_objective, ConditionalFlowMatchingObjective):
            raise ValueError("cfm_objective must be ConditionalFlowMatchingObjective")
        if not isinstance(euler_solver, EulerSolver):
            raise ValueError("euler_solver must be EulerSolver")
        if cfm_objective.convention != euler_solver.convention:
            raise ValueError("Objective and Euler solver conventions must match")
        if not isinstance(cfm_objective.source, GaussianSource):
            raise ValueError("Stage-1 policy requires a Gaussian source")
        if not isinstance(cfm_objective.path, LinearConditionalFlowPath):
            raise ValueError("Stage-1 policy requires a linear conditional flow path")
        if isinstance(tactile_encoder, TactileEncoder):
            if tactile_encoder.preprocess.depth_encoding.value != data_config.tactile_encoding:
                raise ValueError("TactileEncoder encoding must match DataConfig")
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.condition_adapter = condition_adapter
        self.flow_expert = flow_expert
        self.action_normalizer = action_normalizer
        self.state_normalizer = state_normalizer
        self.cfm_objective = cfm_objective
        self.euler_solver = euler_solver
        self.data_config = data_config

    def _require_normalizers(self) -> None:
        if not self.action_normalizer.fitted or not self.state_normalizer.fitted:
            raise RuntimeError("action and state normalizer must be fitted or loaded")

    def encode_condition(self, observations: PolicyObservationBatch) -> ConditionOutput:
        self._require_normalizers()
        if not isinstance(observations, PolicyObservationBatch):
            raise ValueError("observations must be PolicyObservationBatch")
        vision = self.vision_encoder(
            external_rgb=observations.external_rgb, wrist_rgb=observations.wrist_rgb,
        )
        tactile = self.tactile_encoder(
            tactile_depth_0=observations.tactile_depth_0,
            tactile_depth_1=observations.tactile_depth_1,
        )
        features = {**vision, **tactile}
        reference = features["external_rgb"]
        if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
            raise ValueError("external_rgb encoder feature must be floating")
        for name in ("external_rgb", "wrist_rgb", "tactile_depth_0", "tactile_depth_1"):
            feature = features[name]
            if (not isinstance(feature, torch.Tensor) or not feature.is_floating_point()
                    or feature.device != reference.device
                    or feature.device != observations.agent_pos.device
                    or not bool(torch.isfinite(feature).all())):
                raise ValueError(f"{name} encoder feature must be finite and on observation device")
            features[name] = feature.to(dtype=reference.dtype)
        normalized_state = self.state_normalizer.normalize(observations.agent_pos)
        if normalized_state.device != reference.device:
            raise ValueError("Normalized state must remain on observation device")
        normalized_state = normalized_state.to(dtype=reference.dtype)
        return self.condition_adapter(
            **features, agent_pos=normalized_state,
            external_rgb_valid=observations.external_rgb_valid,
            wrist_rgb_valid=observations.wrist_rgb_valid,
            tactile_depth_0_valid=observations.tactile_depth_0_valid,
            tactile_depth_1_valid=observations.tactile_depth_1_valid,
            agent_pos_valid=observations.agent_pos_valid,
        )

    def compute_training_loss(
        self, observations: PolicyObservationBatch,
        encoded_target_action: torch.Tensor, *,
        action_valid_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        source_override: torch.Tensor | None = None,
        time_override: torch.Tensor | None = None,
    ) -> PolicyTrainingOutput:
        self._require_normalizers()
        condition = self.encode_condition(observations)
        if (not isinstance(encoded_target_action, torch.Tensor)
                or encoded_target_action.shape != (
                    observations.agent_pos.shape[0], self.data_config.action_horizon,
                    self.data_config.action_dim,
                )
                or encoded_target_action.device != observations.agent_pos.device
                or not encoded_target_action.is_floating_point()
                or not bool(torch.isfinite(encoded_target_action).all())):
            raise ValueError("encoded_target_action must be finite floating [B,16,10] on observation device")
        normalized_target_action = self.action_normalizer.normalize(encoded_target_action)
        batch = self.cfm_objective.prepare_training_batch(
            normalized_target_action, generator=generator,
            source_override=source_override, time_override=time_override,
        )
        result = self.cfm_objective.compute_loss(
            self.flow_expert, batch, condition.tokens, condition.valid_mask,
            action_valid_mask=action_valid_mask,
        )
        return PolicyTrainingOutput(
            loss=result.loss, prediction=result.prediction,
            velocity_target=result.velocity_target, x_t=result.x_t,
            time=result.time, normalized_target_action=normalized_target_action,
            condition_tokens=condition.tokens, condition_valid_mask=condition.valid_mask,
        )

    def sample_encoded_action(
        self, observations: PolicyObservationBatch, *,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
    ) -> PolicySampleOutput:
        if self.training:
            raise RuntimeError("call policy.eval() before sampling")
        self._require_normalizers()
        with torch.inference_mode():
            condition = self.encode_condition(observations)
            batch_size = observations.agent_pos.shape[0]
            initial_source = self.cfm_objective.source.sample(
                FlowTensorSpec(
                    (batch_size, self.data_config.action_horizon, self.data_config.action_dim),
                    condition.tokens.device, torch.float32,
                ),
                generator=generator,
            )

            def velocity_fn(x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
                velocity = self.flow_expert(
                    x_t, time, condition.tokens, condition.valid_mask,
                )
                # Model matmuls may autocast; the ODE state and update stay float32.
                return velocity.to(dtype=x_t.dtype)

            result = self.euler_solver.solve(
                initial_source, velocity_fn, return_intermediates=return_intermediates,
            )
            encoded_action = self.action_normalizer.denormalize(result.sample)
            return PolicySampleOutput(
                normalized_action=result.sample, encoded_action=encoded_action,
                num_function_evaluations=result.num_function_evaluations,
                time_grid=result.time_grid, trajectory=result.trajectory,
            )
