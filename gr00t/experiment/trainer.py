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

"""Custom Trainer with simple profiling utilities.

This subclass of HuggingFace's ``Trainer`` measures:
1. Data loading latency (time between the end of the previous ``training_step`` and
   the start of the current ``training_step``).
2. Forward-pass latency (time spent inside the base ``training_step`` implementation,
   which essentially wraps the model's forward / loss computation).

The statistics are logged via ``self.log`` every ``profile_log_interval`` steps and
also sent to the standard ``logging`` logger.  This is *not* meant to be a fully
fledged profiler – it is a quick, lightweight way to confirm whether the training
pipeline is bottlenecked by data loading or by the model's computation.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _nan_guard_enabled() -> bool:
    return _env_flag("GROOT_NAN_GUARD")


def _nan_guard_log_steps() -> int:
    return int(os.environ.get("GROOT_NAN_GUARD_LOG_STEPS", "5"))


def _nan_guard_expected_action_mask_sum() -> float | None:
    value = os.environ.get("GROOT_NAN_GUARD_EXPECT_ACTION_MASK_SUM")
    if value is None or value == "":
        return None
    return float(value)


def _nan_guard_param_targets() -> tuple[str, ...]:
    value = os.environ.get("GROOT_NAN_GUARD_PARAM_TARGETS", "action_head")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
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


def _raise_if_nonfinite(name: str, tensor: torch.Tensor, phase: str) -> None:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return
    if not torch.isfinite(tensor.detach()).all():
        raise RuntimeError(
            f"[GROOT_NAN_GUARD:{phase}] non-finite tensor: {_tensor_summary(name, tensor)}"
        )


def _log_tensor(name: str, tensor: torch.Tensor, phase: str, step: int) -> None:
    if _is_main_process() and step < _nan_guard_log_steps():
        logging.info("[GROOT_NAN_GUARD:%s step=%d] %s", phase, step, _tensor_summary(name, tensor))


def _scan_named_tensors(
    named_tensors,
    phase: str,
    step: int,
    *,
    log_matches: bool = False,
) -> None:
    targets = _nan_guard_param_targets()
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if targets and not any(target in name for target in targets):
            continue
        _raise_if_nonfinite(name, tensor, phase)
        if log_matches:
            _log_tensor(name, tensor, phase, step)


class NanGuardCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if not _nan_guard_enabled():
            return
        model = kwargs.get("model")
        if model is None:
            return
        _scan_named_tensors(
            ((name, param) for name, param in model.named_parameters() if param.requires_grad),
            "post-optimizer-params",
            state.global_step,
        )


class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()


class _BatchIterator:
    """Lightweight iterator that yields pre-collated batches."""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:  # noqa: D401
    """Compute token-level accuracy, ignoring ``-100`` label positions.

    Args:
        preds: Predicted token ids of shape ``(batch, seq_len)``.
        labels: Ground-truth label ids with the same shape as ``preds``.

    Returns:
        Scalar tensor with the fraction of correctly predicted labels in the
        current batch.
    """
    # casual prediction
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# Global variables for batched evaluation metrics
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0


def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: Optional[int] = None
):
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


class Gr00tTrainer(Trainer):
    """Trainer that bypasses torch dataloader and makes data collator async."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # noqa: D401 – simple description above
        """Initialize the trainer.

        Args:
            *args: Positional arguments forwarded to ``Trainer``.
        """
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        super().__init__(
            *args,
            **kwargs,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )
        if _nan_guard_enabled():
            self.add_callback(NanGuardCallback())
            if _is_main_process():
                logging.info(
                    "GROOT_NAN_GUARD enabled: expected_action_mask_sum=%s param_targets=%s",
                    _nan_guard_expected_action_mask_sum(),
                    _nan_guard_param_targets(),
                )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # Hide epoch from logged metrics as it's misleading for Iterable datasets.
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):  # noqa: D401
        """Return a iterable dataloader without skipping the data during resume, but reseed the dataset instead."""

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            # ``new_seed`` MUST be the same on every rank: ``ShardedMixtureDataset``
            # builds its shard schedule from this seed and partitions disjointly
            # by index, so a per-rank delta here would cause sample duplication
            # / loss across ranks. Both inputs are rank-symmetric (the dataset's
            # own seed was set rank-symmetrically at __init__, and global_step
            # is read from TrainerState which is broadcast via rendezvous).
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        persistent_workers = self.args.dataloader_num_workers > 0 and _env_flag(
            "GROOT_DATALOADER_PERSISTENT_WORKERS", default=True
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context can only be used with num_workers > 0
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy computation override
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """Compute loss *and* log token-level accuracy every training step.

        We delegate the heavy-lifting (including label smoothing, custom loss
        functions, etc.) to the parent ``Trainer.compute_loss`` implementation
        by calling it with ``return_outputs=True``.  After obtaining the loss
        *and* model outputs, we calculate accuracy and push it to the logger.
        """

        if _nan_guard_enabled():
            self._nan_guard_check_batch(inputs)

        # Use parent implementation to preserve built-in functionality.
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # Record last loss for testing purposes.
        self.loss = loss

        if self.state.global_step % self.args.logging_steps == 0 and model.training and outputs is not None:
            for loss_name in ("future_tactile_loss", "joint_tactile_loss"):
                if loss_name not in outputs:
                    continue
                aux_loss = outputs[loss_name]
                if torch.is_tensor(aux_loss):
                    aux_tensor = aux_loss.detach().to(device=loss.device).reshape(1)
                else:
                    aux_tensor = torch.tensor([float(aux_loss)], device=loss.device)
                aux_mean = self._nested_gather(aux_tensor).mean().item()
                if self.args.local_rank in (-1, 0):
                    self.log({loss_name: aux_mean})

        # --------------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

                # Log a sample of ground-truth vs predicted action tokens from
                # the first batch element so users can verify the model is
                # learning the right behaviors.
                shifted_labels = inputs["labels"][:1, 1:].cpu()
                shifted_preds = preds[:1, :-1]
                mask_0 = shifted_labels[0] != -100
                gt_tokens = shifted_labels[0][mask_0][:20]
                if self.action_offset is not None:
                    gt_tokens = gt_tokens - self.action_offset
                gt_sample = gt_tokens.tolist()
                pred_sample = shifted_preds[0][mask_0[: shifted_preds.shape[1]]][:20].tolist()
                logging.info(
                    "Step %d — GT vs Pred (first 20 action tokens, batch[0]):\n"
                    "  GT:   %s\n  Pred: %s",
                    self.state.global_step,
                    gt_sample,
                    pred_sample,
                )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        if _nan_guard_enabled():
            _scan_named_tensors(
                ((name, param) for name, param in model.named_parameters() if param.requires_grad),
                "pre-forward-params",
                self.state.global_step,
            )

        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        if _nan_guard_enabled():
            _scan_named_tensors(
                (
                    (f"{name}.grad", param.grad)
                    for name, param in model.named_parameters()
                    if param.requires_grad
                ),
                "post-backward-grads",
                self.state.global_step,
            )
            _scan_named_tensors(
                ((name, param) for name, param in model.named_parameters() if param.requires_grad),
                "post-backward-params",
                self.state.global_step,
            )
        return loss

    def _nan_guard_check_batch(self, inputs: dict[str, Any]) -> None:
        batch = inputs.get("inputs", inputs)
        if not isinstance(batch, dict):
            return

        step = self.state.global_step
        for key in ("state", "action", "tactile", "future_tactile", "action_mask"):
            tensor = batch.get(key)
            if torch.is_tensor(tensor):
                _raise_if_nonfinite(f"batch.{key}", tensor, "batch")
                _log_tensor(f"batch.{key}", tensor, "batch", step)

        action = batch.get("action")
        action_mask = batch.get("action_mask")
        if not (torch.is_tensor(action) and torch.is_tensor(action_mask)):
            raise RuntimeError("[GROOT_NAN_GUARD:batch] missing action or action_mask in training batch")
        if action.shape != action_mask.shape:
            raise RuntimeError(
                "[GROOT_NAN_GUARD:batch] action/action_mask shape mismatch: "
                f"action={tuple(action.shape)} action_mask={tuple(action_mask.shape)}"
            )

        expected_per_sample = _nan_guard_expected_action_mask_sum()
        if expected_per_sample is None:
            return
        expected = float(action.shape[0]) * expected_per_sample
        actual = float(action_mask.detach().float().sum().item())
        if abs(actual - expected) > 1e-3:
            raise RuntimeError(
                "[GROOT_NAN_GUARD:batch] unexpected action_mask sum: "
                f"actual={actual:.6g} expected={expected:.6g} batch={action.shape[0]} "
                f"per_sample={expected_per_sample:.6g}"
            )
