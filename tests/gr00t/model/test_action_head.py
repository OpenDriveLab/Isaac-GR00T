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

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])

    def test_future_tactile_aux_requires_target(self):
        config = _small_config(
            use_future_tactile_aux=True,
            future_tactile_dim=5,
            future_tactile_horizon=3,
        )
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        with pytest.raises(RuntimeError, match="use_future_tactile_aux=True"):
            head.forward(_make_backbone_output(config), action_input)

    def test_future_tactile_aux_outputs_loss_and_prediction(self):
        config = _small_config(
            use_future_tactile_aux=True,
            future_tactile_dim=5,
            future_tactile_horizon=3,
        )
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        action_input["future_tactile"] = torch.randn(2, config.future_tactile_horizon, 5)
        out = head.forward(_make_backbone_output(config), action_input)
        assert "future_tactile_loss" in out
        assert "future_tactile_pred" in out
        assert out["future_tactile_pred"].shape == (2, config.future_tactile_horizon, 5)
        assert torch.isfinite(out["future_tactile_loss"])
        assert torch.isfinite(out["future_tactile_pred"]).all()

    def test_joint_tactile_denoising_requires_target(self):
        config = _small_config(
            use_tactile_token=True,
            tactile_latent_dim=5,
            use_joint_tactile_denoising=True,
            joint_tactile_dim=5,
            joint_tactile_horizon=3,
        )
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        with pytest.raises(RuntimeError, match="use_joint_tactile_denoising=True"):
            head.forward(_make_backbone_output(config), action_input)

    def test_joint_tactile_denoising_outputs_loss_and_velocity(self):
        config = _small_config(
            use_tactile_token=True,
            tactile_latent_dim=5,
            use_joint_tactile_denoising=True,
            joint_tactile_dim=5,
            joint_tactile_horizon=3,
        )
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        action_input["tactile"] = torch.randn(2, 1, 5)
        action_input["future_tactile"] = torch.randn(2, config.joint_tactile_horizon, 5)
        out = head.forward(_make_backbone_output(config), action_input)
        assert "joint_tactile_loss" in out
        assert "pred_future_tactile_velocity" in out
        assert out["pred_future_tactile_velocity"].shape == (2, config.joint_tactile_horizon, 5)
        assert torch.isfinite(out["loss"])
        assert torch.isfinite(out["joint_tactile_loss"])
        assert torch.isfinite(out["pred_future_tactile_velocity"]).all()

    def test_future_aux_and_joint_denoising_are_mutually_exclusive(self):
        config = _small_config(
            use_tactile_token=True,
            use_future_tactile_aux=True,
            use_joint_tactile_denoising=True,
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            Gr00tN1d7ActionHead(config)


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1

    def test_get_action_joint_tactile_denoising_keeps_action_shape(self):
        config = _small_config(
            use_tactile_token=True,
            tactile_latent_dim=5,
            use_joint_tactile_denoising=True,
            joint_tactile_dim=5,
            joint_tactile_horizon=3,
        )
        head = Gr00tN1d7ActionHead(config)
        action_input = _make_action_input(config)
        action_input["tactile"] = torch.randn(2, 1, 5)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        assert out["future_tactile_pred"].shape == (2, config.joint_tactile_horizon, 5)
        assert torch.isfinite(out["action_pred"]).all()
        assert torch.isfinite(out["future_tactile_pred"]).all()


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad
