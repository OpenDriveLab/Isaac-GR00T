import torch
from torch import nn
from transformers import TrainingArguments

from gr00t.experiment.trainer import Gr00tTrainer


class _TinyAuxLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_values):
        loss = self.weight * input_values.float().mean()
        return {
            "loss": loss,
            "future_tactile_loss": loss.detach() + 2.0,
        }


class _TinyJointLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_values):
        loss = self.weight * input_values.float().mean()
        return {
            "loss": loss,
            "joint_tactile_loss": loss.detach() + 4.0,
        }


def _make_trainer(model, tmp_path):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        logging_steps=1,
        report_to="none",
    )
    trainer = Gr00tTrainer(model=model, args=args)
    trainer.state.global_step = 1
    logged = []
    trainer.log = lambda logs, start_time=None: logged.append(logs)
    return trainer, logged


def test_compute_loss_logs_future_tactile_loss(tmp_path):
    model = _TinyAuxLossModel()
    trainer, logged = _make_trainer(model, tmp_path)

    loss = trainer.compute_loss(model, {"input_values": torch.tensor([1.0])})

    assert torch.isfinite(loss)
    assert any("future_tactile_loss" in entry for entry in logged)
    logged_aux = [entry["future_tactile_loss"] for entry in logged if "future_tactile_loss" in entry]
    assert logged_aux == [3.0]


def test_compute_loss_logs_joint_tactile_loss(tmp_path):
    model = _TinyJointLossModel()
    trainer, logged = _make_trainer(model, tmp_path)

    loss = trainer.compute_loss(model, {"input_values": torch.tensor([1.0])})

    assert torch.isfinite(loss)
    assert any("joint_tactile_loss" in entry for entry in logged)
    logged_aux = [entry["joint_tactile_loss"] for entry in logged if "joint_tactile_loss" in entry]
    assert logged_aux == [5.0]
