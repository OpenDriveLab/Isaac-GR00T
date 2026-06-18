# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
)


logger = logging.getLogger(__name__)


def _nan_guard_enabled() -> bool:
    return os.environ.get("GROOT_NAN_GUARD", "0").lower() in {"1", "true", "yes", "on"}


def _nan_guard_log_forwards() -> int:
    return int(os.environ.get("GROOT_NAN_GUARD_LOG_FORWARDS", "5"))


def _nan_guard_expected_action_mask_sum() -> float | None:
    value = os.environ.get("GROOT_NAN_GUARD_EXPECT_ACTION_MASK_SUM")
    if value is None or value == "":
        return None
    return float(value)


def _nan_guard_is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _nan_guard_tensor_summary(name: str, tensor: torch.Tensor) -> str:
    shape = tuple(tensor.shape)
    if tensor.numel() == 0:
        return f"{name}: shape={shape} dtype={tensor.dtype} empty"
    if not torch.is_floating_point(tensor):
        return f"{name}: shape={shape} dtype={tensor.dtype}"

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    bad_count = detached.numel() - finite_count
    if finite_count > 0:
        finite_values = detached[finite].float()
        min_value = float(finite_values.min().item())
        max_value = float(finite_values.max().item())
        mean_value = float(finite_values.mean().item())
        std_value = float(finite_values.std(unbiased=False).item())
    else:
        min_value = max_value = mean_value = std_value = float("nan")
    total_sum = float(detached.float().sum().item()) if bad_count == 0 else float("nan")
    return (
        f"{name}: shape={shape} dtype={tensor.dtype} finite={finite_count}/{detached.numel()} "
        f"bad={bad_count} sum={total_sum:.6g} min={min_value:.6g} max={max_value:.6g} "
        f"mean={mean_value:.6g} std={std_value:.6g}"
    )


def _nan_guard_check_tensor(name: str, tensor: torch.Tensor | None, phase: str, call_idx: int) -> None:
    if tensor is None:
        return
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return
    if _nan_guard_is_main_process() and call_idx < _nan_guard_log_forwards():
        logger.info(
            "[GROOT_NAN_GUARD:%s forward=%d] %s",
            phase,
            call_idx,
            _nan_guard_tensor_summary(name, tensor),
        )
    if not torch.isfinite(tensor.detach()).all():
        raise RuntimeError(
            f"[GROOT_NAN_GUARD:{phase}] non-finite tensor: "
            f"{_nan_guard_tensor_summary(name, tensor)}"
        )


class FutureTactileDenoisingEncoder(nn.Module):
    """Encode noisy future tactile latents with explicit diffusion timestep conditioning."""

    def __init__(self, tactile_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.tactile_dim = tactile_dim
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(tactile_dim, hidden_size)
        self.time_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, tactile: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if tactile.ndim != 3:
            raise ValueError(
                "Future tactile encoder expects [B, T, D] input, "
                f"got shape {tuple(tactile.shape)}"
            )
        if tactile.shape[-1] != self.tactile_dim:
            raise ValueError(
                f"Future tactile dim {tactile.shape[-1]} != configured {self.tactile_dim}"
            )

        batch_size, horizon, _ = tactile.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == batch_size:
            timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
        else:
            raise ValueError("Expected `timesteps` to have shape [B] for future tactile encoding.")

        tactile_emb = self.input_proj(tactile)
        time_emb = self.pos_encoding(timesteps).to(dtype=tactile_emb.dtype)
        hidden = torch.cat([tactile_emb, time_emb], dim=-1)
        hidden = F.silu(self.time_proj(hidden))
        return self.output_proj(hidden)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.use_tactile_token = bool(getattr(config, "use_tactile_token", False))
        self.tactile_dropout_prob = float(getattr(config, "tactile_dropout_prob", 0.0))
        self.tactile_latent_dim = int(getattr(config, "tactile_latent_dim", 128))
        if self.use_tactile_token:
            self.tactile_projector = nn.Sequential(
                nn.Linear(self.tactile_latent_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.input_embedding_dim),
            )
            self.tactile_type_embedding = nn.Parameter(torch.zeros(1, 1, self.input_embedding_dim))
            self.null_tactile_token = nn.Parameter(torch.zeros(1, 1, self.input_embedding_dim))
            self.reset_tactile_token_parameters()

        self.use_future_tactile_aux = bool(getattr(config, "use_future_tactile_aux", False))
        self.future_tactile_loss_weight = float(
            getattr(config, "future_tactile_loss_weight", 0.5)
        )
        self.future_tactile_dim = int(getattr(config, "future_tactile_dim", 128))
        self.future_tactile_horizon = int(getattr(config, "future_tactile_horizon", 16))
        if self.use_future_tactile_aux:
            self.future_tactile_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.future_tactile_dim,
            )

        self.use_joint_tactile_denoising = bool(
            getattr(config, "use_joint_tactile_denoising", False)
        )
        if self.use_future_tactile_aux and self.use_joint_tactile_denoising:
            raise ValueError(
                "use_future_tactile_aux and use_joint_tactile_denoising are mutually exclusive"
            )
        if self.use_joint_tactile_denoising and not self.use_tactile_token:
            raise ValueError("A5 joint tactile denoising requires use_tactile_token=True")
        self.joint_tactile_loss_weight = float(
            getattr(config, "joint_tactile_loss_weight", 0.5)
        )
        self.joint_tactile_dim = int(getattr(config, "joint_tactile_dim", 128))
        self.joint_tactile_horizon = int(getattr(config, "joint_tactile_horizon", 16))
        if self.use_joint_tactile_denoising:
            self.future_tactile_encoder = FutureTactileDenoisingEncoder(
                tactile_dim=self.joint_tactile_dim,
                hidden_size=self.input_embedding_dim,
                output_dim=self.input_embedding_dim,
            )
            self.joint_tactile_velocity_decoder = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.joint_tactile_dim),
            )
            self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, self.input_embedding_dim))
            self.joint_future_tactile_type_embedding = nn.Parameter(
                torch.zeros(1, 1, self.input_embedding_dim)
            )
            self.reset_joint_tactile_denoising_parameters()

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    @staticmethod
    def _init_mlp(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_tactile_token_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.tactile_type_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.null_tactile_token, mean=0.0, std=0.02)
            self._init_mlp(self.tactile_projector)

    def reset_future_tactile_parameters(self) -> None:
        with torch.no_grad():
            for m in self.future_tactile_decoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif hasattr(m, "W") and isinstance(m.W, nn.Parameter):
                    nn.init.normal_(m.W, mean=0.0, std=0.02)
                    if hasattr(m, "b") and isinstance(m.b, nn.Parameter):
                        nn.init.zeros_(m.b)

    def reset_joint_tactile_denoising_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.action_type_embedding, mean=0.0, std=0.02)
            nn.init.normal_(self.joint_future_tactile_type_embedding, mean=0.0, std=0.02)
            self._init_mlp(self.future_tactile_encoder)
            self._init_mlp(self.joint_tactile_velocity_decoder)

    def assert_finite_action_head(self) -> None:
        for name, p in self.named_parameters():
            if not torch.isfinite(p.data).all():
                bad = int((~torch.isfinite(p.data)).sum().item())
                total = p.data.numel()
                raise RuntimeError(
                    f"Non-finite values in action_head.{name}: "
                    f"{bad}/{total} elements are NaN/Inf"
                )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.use_tactile_token:
                self.tactile_projector.requires_grad_(False)
                self.tactile_type_embedding.requires_grad_(False)
                self.null_tactile_token.requires_grad_(False)
            if self.use_future_tactile_aux:
                self.future_tactile_decoder.requires_grad_(False)
            if self.use_joint_tactile_denoising:
                self.future_tactile_encoder.requires_grad_(False)
                self.joint_tactile_velocity_decoder.requires_grad_(False)
                self.action_type_embedding.requires_grad_(False)
                self.joint_future_tactile_type_embedding.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.use_tactile_token:
                    self.tactile_projector.eval()
                if self.use_future_tactile_aux:
                    self.future_tactile_decoder.eval()
                if self.use_joint_tactile_denoising:
                    self.future_tactile_encoder.eval()
                    self.joint_tactile_velocity_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _encode_tactile_features(
        self, action_input: BatchFeature, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor | None:
        if not self.use_tactile_token:
            return None

        if "tactile" in action_input and action_input.tactile is not None:
            tactile = action_input.tactile.to(device=device, dtype=dtype)
            if tactile.ndim == 3:
                tactile = tactile.reshape(tactile.shape[0], -1)
            elif tactile.ndim == 1:
                tactile = tactile.unsqueeze(0)
            assert tactile.shape[-1] == self.tactile_latent_dim, (
                f"Tactile latent dim {tactile.shape[-1]} != configured {self.tactile_latent_dim}"
            )
            tactile_features = self.tactile_projector(tactile).unsqueeze(1)
            tactile_features = tactile_features + self.tactile_type_embedding.to(
                device=device, dtype=tactile_features.dtype
            )
        else:
            tactile_features = self.null_tactile_token.to(device=device, dtype=dtype).expand(
                batch_size, -1, -1
            )

        if self.training and self.tactile_dropout_prob > 0:
            do_dropout = (
                torch.rand(batch_size, device=device) < self.tactile_dropout_prob
            )[:, None, None]
            null_token = self.null_tactile_token.to(
                device=device, dtype=tactile_features.dtype
            ).expand_as(tactile_features)
            tactile_features = torch.where(do_dropout, null_token, tactile_features)

        return tactile_features

    def _add_sequence_position_embedding(
        self, features: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not self.config.add_pos_embed:
            return features
        pos_ids = torch.arange(features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        return features + pos_embs

    def _build_dit_sequence(
        self,
        state_features: torch.Tensor,
        tactile_features: torch.Tensor | None,
        action_features: torch.Tensor,
        future_tactile_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, slice, slice | None]:
        parts = [state_features]
        pos = state_features.shape[1]

        if tactile_features is not None:
            parts.append(tactile_features)
            pos += tactile_features.shape[1]

        action_slice = slice(pos, pos + action_features.shape[1])
        parts.append(action_features)
        pos += action_features.shape[1]

        future_tactile_slice = None
        if future_tactile_features is not None:
            future_tactile_slice = slice(pos, pos + future_tactile_features.shape[1])
            parts.append(future_tactile_features)

        return torch.cat(parts, dim=1), action_slice, future_tactile_slice

    def _encode_noisy_future_tactile(
        self,
        future_tactile_target: torch.Tensor | None,
        t: torch.Tensor,
        t_discretized: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, int]:
        if not self.use_joint_tactile_denoising:
            return None, None, None, 0
        if future_tactile_target is None:
            raise RuntimeError(
                "use_joint_tactile_denoising=True but the batch does not contain "
                "future_tactile. Check the A5 modality config "
                "state.metadata['future_tactile_keys'] and processor output."
            )

        future_tactile_target = future_tactile_target.to(device=device, dtype=dtype)
        h = min(future_tactile_target.shape[1], self.joint_tactile_horizon)
        future_tactile_target = future_tactile_target[:, :h, :]
        if future_tactile_target.shape[-1] != self.joint_tactile_dim:
            raise ValueError(
                f"Future tactile dim {future_tactile_target.shape[-1]} != configured "
                f"{self.joint_tactile_dim}"
            )

        noise = torch.randn_like(future_tactile_target)
        noisy_future_tactile = (1 - t) * noise + t * future_tactile_target
        future_tactile_velocity = future_tactile_target - noise
        future_tactile_features = self.future_tactile_encoder(
            noisy_future_tactile, t_discretized
        )
        future_tactile_features = future_tactile_features + self.joint_future_tactile_type_embedding.to(
            device=device, dtype=future_tactile_features.dtype
        )
        future_tactile_features = self._add_sequence_position_embedding(
            future_tactile_features, device
        )
        return future_tactile_features, future_tactile_velocity, noisy_future_tactile, h

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)
        tactile_features = self._encode_tactile_features(
            action_input, state_features.shape[0], device, state_features.dtype
        )

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if self.use_joint_tactile_denoising:
            action_features = action_features + self.action_type_embedding.to(
                device=device, dtype=action_features.dtype
            )
        action_features = self._add_sequence_position_embedding(action_features, device)

        future_tactile_features, future_tactile_velocity, noisy_future_tactile, future_tactile_h = (
            self._encode_noisy_future_tactile(
                getattr(action_input, "future_tactile", None),
                t,
                t_discretized,
                device,
                action_features.dtype,
            )
        )

        # Join vision, language, state, optional tactile, action, and optional future tactile.
        sa_embs, action_slice, future_tactile_slice = self._build_dit_sequence(
            state_features, tactile_features, action_features, future_tactile_features
        )
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        action_hidden = model_output[:, action_slice, :]
        pred_actions = self.action_decoder(action_hidden, embodiment_id)
        pred_future_tactile_velocity = None
        if self.use_joint_tactile_denoising:
            if future_tactile_slice is None:
                raise RuntimeError("Internal error: missing future tactile slice for A5")
            future_tactile_hidden = model_output[:, future_tactile_slice, :]
            pred_future_tactile_velocity = self.joint_tactile_velocity_decoder(
                future_tactile_hidden
            )

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        if _nan_guard_enabled():
            call_idx = getattr(self, "_nan_guard_forward_calls", 0)
            self._nan_guard_forward_calls = call_idx + 1
            if pred_actions.shape != velocity.shape or pred_actions.shape != action_mask.shape:
                raise RuntimeError(
                    "[GROOT_NAN_GUARD:forward] action tensor shape mismatch: "
                    f"pred_actions={tuple(pred_actions.shape)} velocity={tuple(velocity.shape)} "
                    f"action_mask={tuple(action_mask.shape)}"
                )
            expected_per_sample = _nan_guard_expected_action_mask_sum()
            if expected_per_sample is not None:
                expected = float(action_mask.shape[0]) * expected_per_sample
                actual = float(action_mask.detach().float().sum().item())
                if abs(actual - expected) > 1e-3:
                    raise RuntimeError(
                        "[GROOT_NAN_GUARD:forward] unexpected action_mask sum: "
                        f"actual={actual:.6g} expected={expected:.6g} "
                        f"batch={action_mask.shape[0]} per_sample={expected_per_sample:.6g}"
                    )
            for name, tensor in (
                ("actions", actions),
                ("noise", noise),
                ("velocity", velocity),
                ("noisy_trajectory", noisy_trajectory),
                ("state_features", state_features),
                ("tactile_features", tactile_features),
                ("action_features", action_features),
                ("noisy_future_tactile", noisy_future_tactile),
                ("future_tactile_velocity", future_tactile_velocity),
                ("future_tactile_features", future_tactile_features),
                ("sa_embs", sa_embs),
                ("model_output", model_output),
                ("pred_actions", pred_actions),
                ("pred_future_tactile_velocity", pred_future_tactile_velocity),
                ("action_mask", action_mask),
            ):
                _nan_guard_check_tensor(name, tensor, "forward", call_idx)

        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        if _nan_guard_enabled():
            _nan_guard_check_tensor("action_loss", action_loss, "loss", call_idx)
            _nan_guard_check_tensor("loss", loss, "loss", call_idx)
            if float(action_mask.detach().float().sum().item()) > 0 and float(loss.detach().float().item()) == 0.0:
                logger.warning(
                    "[GROOT_NAN_GUARD:loss forward=%d] loss is exactly 0.0 with nonzero action_mask; %s; %s; %s",
                    call_idx,
                    _nan_guard_tensor_summary("pred_actions", pred_actions),
                    _nan_guard_tensor_summary("velocity", velocity),
                    _nan_guard_tensor_summary("action_mask", action_mask),
                )

        future_tactile_loss = None
        future_tactile_pred = None
        future_tactile_target = getattr(action_input, "future_tactile", None)
        if self.use_future_tactile_aux and future_tactile_target is None:
            raise RuntimeError(
                "use_future_tactile_aux=True but the batch does not contain "
                "future_tactile. Check the A3/A4 modality config "
                "state.metadata['future_tactile_keys'], the processor output, "
                "and the collated training batch; otherwise the auxiliary head "
                "would silently train as plain A2/A1."
            )
        if self.use_future_tactile_aux:
            future_tactile_target = future_tactile_target.to(
                device=model_output.device, dtype=model_output.dtype
            )
            h = min(future_tactile_target.shape[1], self.future_tactile_horizon)
            future_tactile_target = future_tactile_target[:, :h, :]
            assert future_tactile_target.shape[-1] == self.future_tactile_dim, (
                f"Future tactile dim {future_tactile_target.shape[-1]} != configured "
                f"{self.future_tactile_dim}"
            )
            future_tactile_pred = self.future_tactile_decoder(action_hidden, embodiment_id)[:, :h, :]
            temporal_mask = action_mask[:, :h, :1].to(
                device=future_tactile_pred.device, dtype=future_tactile_pred.dtype
            )
            temporal_mask = temporal_mask.expand_as(future_tactile_pred)
            future_tactile_loss = (
                F.mse_loss(future_tactile_pred, future_tactile_target, reduction="none")
                * temporal_mask
            ).sum() / (temporal_mask.sum() + 1e-6)
            loss = loss + self.future_tactile_loss_weight * future_tactile_loss

        joint_tactile_loss = None
        if self.use_joint_tactile_denoising:
            temporal_mask = action_mask[:, :future_tactile_h, :1].to(
                device=pred_future_tactile_velocity.device,
                dtype=pred_future_tactile_velocity.dtype,
            )
            temporal_mask = temporal_mask.expand_as(pred_future_tactile_velocity)
            joint_tactile_loss = (
                F.mse_loss(
                    pred_future_tactile_velocity,
                    future_tactile_velocity,
                    reduction="none",
                )
                * temporal_mask
            ).sum() / (temporal_mask.sum() + 1e-6)
            loss = loss + self.joint_tactile_loss_weight * joint_tactile_loss
            if _nan_guard_enabled():
                _nan_guard_check_tensor("joint_tactile_loss", joint_tactile_loss, "loss", call_idx)

        output = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "tactile_features": tactile_features,
        }
        if future_tactile_loss is not None:
            output["future_tactile_loss"] = future_tactile_loss
            output["future_tactile_pred"] = future_tactile_pred
        if joint_tactile_loss is not None:
            output["joint_tactile_loss"] = joint_tactile_loss
            output["pred_future_tactile_velocity"] = pred_future_tactile_velocity
        return output

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state and optional tactile token.
        state_features = self.state_encoder(state, embodiment_id)
        tactile_features = self._encode_tactile_features(
            action_input, state_features.shape[0], vl_embeds.device, state_features.dtype
        )

        return BatchFeature(
            data={
                "backbone_features": vl_embeds,
                "state_features": state_features,
                "tactile_features": tactile_features,
            }
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        tactile_features: torch.Tensor | None,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features
        options = options or {}
        return_future_tactile = bool(options.get("return_future_tactile", False))

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)
        future_tactile_pred = None
        if self.use_joint_tactile_denoising:
            future_tactile_pred = torch.randn(
                size=(batch_size, self.joint_tactile_horizon, self.joint_tactile_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        final_action_hidden = None
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.use_joint_tactile_denoising:
                action_features = action_features + self.action_type_embedding.to(
                    device=device, dtype=action_features.dtype
                )
            action_features = self._add_sequence_position_embedding(action_features, device)

            future_tactile_features = None
            if self.use_joint_tactile_denoising:
                future_tactile_features = self.future_tactile_encoder(
                    future_tactile_pred, timesteps_tensor
                )
                future_tactile_features = (
                    future_tactile_features
                    + self.joint_future_tactile_type_embedding.to(
                        device=device, dtype=future_tactile_features.dtype
                    )
                )
                future_tactile_features = self._add_sequence_position_embedding(
                    future_tactile_features, device
                )

            # Join vision, language, state, optional tactile, action, and optional future tactile.
            sa_embs, action_slice, future_tactile_slice = self._build_dit_sequence(
                state_features,
                tactile_features,
                action_features,
                future_tactile_features,
            )

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            action_hidden = model_output[:, action_slice, :]
            final_action_hidden = action_hidden
            pred_velocity = self.action_decoder(action_hidden, embodiment_id)
            pred_future_tactile_velocity = None
            if self.use_joint_tactile_denoising:
                if future_tactile_slice is None:
                    raise RuntimeError("Internal error: missing future tactile slice for A5")
                future_tactile_hidden = model_output[:, future_tactile_slice, :]
                pred_future_tactile_velocity = self.joint_tactile_velocity_decoder(
                    future_tactile_hidden
                )

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength
            if self.use_joint_tactile_denoising:
                future_tactile_pred = future_tactile_pred + dt * pred_future_tactile_velocity

        output = {
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "tactile_features": tactile_features,
        }

        future_tactile_target = getattr(action_input, "future_tactile", None)
        future_tactile_h = None
        if future_tactile_target is not None:
            future_tactile_h = min(
                int(future_tactile_target.shape[1]),
                int(getattr(self, "future_tactile_horizon", future_tactile_target.shape[1])),
            )
        if future_tactile_h is None and self.use_joint_tactile_denoising:
            future_tactile_h = self.joint_tactile_horizon

        if return_future_tactile and self.use_future_tactile_aux:
            if final_action_hidden is None:
                raise RuntimeError("Internal error: no action hidden state for future tactile eval")
            h = future_tactile_h if future_tactile_h is not None else self.future_tactile_horizon
            future_tactile_pred = self.future_tactile_decoder(final_action_hidden, embodiment_id)[
                :, :h, :
            ]

        if return_future_tactile and future_tactile_pred is not None:
            output["future_tactile_pred"] = future_tactile_pred
        if return_future_tactile and future_tactile_target is not None:
            h = future_tactile_h if future_tactile_h is not None else future_tactile_target.shape[1]
            future_tactile_target = future_tactile_target[:, :h, :].to(
                device=actions.device, dtype=actions.dtype
            )
            output["future_tactile_target"] = future_tactile_target
            output["future_tactile_delta_indices"] = list(range(h))
            if h > 0:
                output["current_tactile"] = future_tactile_target[:, 0, :]
        return BatchFeature(data=output)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            tactile_features=features.tactile_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
