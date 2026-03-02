"""
Standalone disentanglement distillation runner with checkpointing and early stopping.
"""

import argparse
import os
import random
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import torch
from torch.nn import functional as F

from Memory_meta import Memory_Meta, Memory_module
from utils import (
    Args,
    Device,
    MLogger,
    Model_config,
    Model_config_attention8,
    Model_config_multi_head_attention5_conv3,
    Model_config_attention5_conv3,
    Model_config_attention5_conv3_large,
    _split_parameters,
    load_config,
)


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {key: _move_to_device(val, device) for key, val in obj.items()}
    return obj


def _ensure_optimizer_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def _load_distillation_artifacts(save_path: str, device: torch.device) -> Tuple[Any, Any, Any]:
    prev_loss = joblib.load(os.path.join(save_path, "prev_loss.pkl"))
    prev_data = joblib.load(os.path.join(save_path, "prev_data.pkl"))
    prev_models = joblib.load(os.path.join(save_path, "models.pkl"))
    prev_loss = _move_to_device(prev_loss, device)
    prev_data = _move_to_device(prev_data, device)
    prev_models = _move_to_device(prev_models, device)
    return prev_loss, prev_data, prev_models


def _save_epoch_outputs(
    save_path: str,
    checkpoint_dir: str,
    epoch_idx: int,
    model: Memory_Meta,
    avg_loss: float,
    best_loss: float,
    bad_epochs: int,
    max_epochs: int,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_payload = {
        "epoch": epoch_idx,
        "memory_module_state": model.Memory_module.state_dict(),
        "optimizer_state": model.Memory_module.optim.state_dict(),
        "avg_loss": avg_loss,
        "best_loss": best_loss,
        "bad_epochs": bad_epochs,
        "max_epochs": max_epochs,
    }
    torch.save(checkpoint_payload, os.path.join(checkpoint_dir, "distill_last.pt"))
    torch.save(checkpoint_payload, os.path.join(save_path, "distill_last.pt"))

    joblib.dump(
        model.Memory_module.memory.content_memory,
        os.path.join(save_path, "Content_memory.pkl"),
    )
    joblib.dump(
        list(model.Memory_module.memory.parameters()),
        os.path.join(save_path, "Query.pkl"),
    )


def _load_checkpoint(
    checkpoint_path: str,
    model: Memory_Meta,
    device: torch.device,
) -> Tuple[int, float, int, Optional[int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.Memory_module.load_state_dict(checkpoint["memory_module_state"])
    model.Memory_module.optim.load_state_dict(checkpoint["optimizer_state"])
    _ensure_optimizer_device(model.Memory_module.optim, device)
    start_epoch = checkpoint.get("epoch", -1) + 1
    best_loss = checkpoint.get("best_loss", float("inf"))
    bad_epochs = checkpoint.get("bad_epochs", 0)
    max_epochs = checkpoint.get("max_epochs")
    return start_epoch, best_loss, bad_epochs, max_epochs


MODEL_CONFIG_MAP = {
    "default": Model_config,
    "attention8": Model_config_attention8,
    "multi_head_attention5_conv3": Model_config_multi_head_attention5_conv3,
    "attention5_conv3": Model_config_attention5_conv3,
    "attention5_conv3_large":Model_config_attention5_conv3_large,
}


def run_distillation(
    save_path: str,
    args: Args,
    model_config: list,
    logger: Optional[MLogger] = None,
    device: Optional[str] = None,
    resume: bool = False,
    checkpoint_path: Optional[str] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = 20,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    """Run disentanglement distillation with checkpointing and early stopping."""
    device = torch.device(device or Device)
    model = Memory_Meta(args, model_config).to(device)

    prev_loss, prev_data, prev_models = _load_distillation_artifacts(save_path, device)
    if hasattr(prev_models, "shape") and prev_models.shape:
        args.task_num = int(prev_models.shape[0])

    memory_module = Memory_module(args, model.meta_Parameter_nums)
    model.Memory_module = memory_module.to(device)

    # Load Memory_module initial state for reproducible distillation
    memory_init_path = os.path.join(save_path, "memory_module_init.pt")
    if os.path.exists(memory_init_path):
        model.Memory_module.load_state_dict(torch.load(memory_init_path, map_location=device))
        if logger:
            logger.info(f"Loaded Memory_module initial state from {memory_init_path}")
    else:
        if logger:
            logger.warning(f"memory_module_init.pt not found, using fresh initialization")

    model.Memory_module.prev_loss = prev_loss
    model.Memory_module.prev_data = prev_data
    model.Memory_module.models = prev_models

    checkpoint_dir = os.path.join(save_path, "distillation_checkpoints")
    if resume:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(checkpoint_dir, "distill_last.pt")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(save_path, "distill_last.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        start_epoch, best_loss, bad_epochs, ckpt_max_epochs = _load_checkpoint(
            checkpoint_path, model, device
        )
        if max_epochs is None and ckpt_max_epochs is not None:
            max_epochs = ckpt_max_epochs
    else:
        start_epoch = 0
        best_loss = float("inf")
        bad_epochs = 0

    if max_epochs is None:
        max_epochs = args.distillation_epoch

    if logger:
        logger.info("Starting disentanglement distillation phase")
        logger.info(
            f"Distillation config: max_epochs={max_epochs}, patience={patience}, min_delta={min_delta}"
        )
        if resume:
            logger.info(f"Resuming distillation from epoch {start_epoch + 1}")

    for d_epoch in range(start_epoch, max_epochs):
        if logger:
            logger.info(f"Distillation Epoch: [{d_epoch + 1}/{max_epochs}]")
        else:
            print(f"Distillation Epoch: {d_epoch + 1}/{max_epochs}")

        model.Memory_module.writehead(model.Memory_module.models)
        total_loss = 0.0
        task_count = len(model.Memory_module.prev_data)

        for task_idx, (index_prev, x_prev, y_prev) in enumerate(model.Memory_module.prev_data):
            similarity_weights = model.Memory_module(index_prev)[0]

            logits = []
            for content in model.Memory_module.memory.content_memory:
                weights_memory = _split_parameters(
                    content.unsqueeze(0),
                    model.net.parameters(),
                )
                logits.append(model.net(x_prev, weights_memory, bn_training=True))

            weighted_softmax = sum(
                similarity_weights[k] * F.softmax(logit)
                for k, logit in enumerate(logits)
            )
            task_loss = torch.sum(
                torch.log(weighted_softmax) * model.Memory_module.prev_loss[task_idx] * -1
            )
            total_loss += task_loss

        avg_loss = total_loss / max(task_count, 1)

        if logger:
            logger.info(
                f"Distillation Epoch: [{d_epoch + 1}/{max_epochs}]\tLoss: {avg_loss.item():.5f}"
            )

        model.Memory_module.optim.zero_grad()
        avg_loss.backward()
        model.Memory_module.optim.step()
        model.Memory_module.content_memory = model.Memory_module.memory.content_memory.detach()

        avg_loss_value = float(avg_loss.item())
        if avg_loss_value + min_delta < best_loss:
            best_loss = avg_loss_value
            bad_epochs = 0
        else:
            bad_epochs += 1

        _save_epoch_outputs(
            save_path=save_path,
            checkpoint_dir=checkpoint_dir,
            epoch_idx=d_epoch,
            model=model,
            avg_loss=avg_loss_value,
            best_loss=best_loss,
            bad_epochs=bad_epochs,
            max_epochs=max_epochs,
        )

        if patience is not None and bad_epochs >= patience:
            if logger:
                logger.info(
                    f"Early stopping triggered at epoch {d_epoch + 1} "
                    f"(best_loss={best_loss:.5f})"
                )
            break

    if logger:
        logger.info("Distillation training completed")

    return {
        "best_loss": best_loss,
        "bad_epochs": bad_epochs,
        "checkpoint_dir": checkpoint_dir,
    }


def _build_args_from_config(config: Dict[str, Any], dimension_override: Optional[int] = None) -> Args:
    """Build Args from config, optionally overriding dimension (C and R).

    Args:
        config: Configuration dictionary
        dimension_override: If provided, override both num_of_index (C) and len_of_index (R)
    """
    C = dimension_override if dimension_override else config["Train"]["Meta_learning"]["Model_parameter"]["num_of_index"]
    R = dimension_override if dimension_override else config["Train"]["Meta_learning"]["Model_parameter"]["len_of_index"]

    return Args(
        C=C,
        L=config["Train"]["Meta_learning"]["Model_parameter"]["len_of_embedding"],
        R=R,
        meta_lr=config["Train"]["Meta_learning"]["Model_parameter"]["meta_lr"],
        update_lr=config["Train"]["Meta_learning"]["Model_parameter"]["inner_loop_lr"],
        update_step=config["Train"]["Meta_learning"]["Model_parameter"]["inner_update_step"],
        update_step_test=config["Train"]["Meta_learning"]["Model_parameter"]["inner_fine_tuning"],
        regular=config["Train"]["Meta_learning"]["Model_parameter"]["regular_coefficient"],
        epoch=config["Train"]["Meta_learning"]["Trainer_parameter"]["epoch"],
        distillation_epoch=config["Train"]["Disentanglement_distillation"]["Trainer_parameter"]["epoch"],
        num_of_tasks=0,
    )


def main() -> None:
    default_config_path = os.path.join(os.path.dirname(__file__), "Configs", "TrainingConfig.yaml")
    parser = argparse.ArgumentParser(description="Run disentanglement distillation only.")
    parser.add_argument("--save-path", required=True, help="Path containing prev_loss.pkl, prev_data.pkl, models.pkl")
    parser.add_argument("--config", default=default_config_path, help="Path to TrainingConfig.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default="attention5_conv3_large",
        choices=list(MODEL_CONFIG_MAP.keys()),
        help="Model configuration to use",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint-path", default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override distillation epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stop patience")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Early stop min delta")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", default=None, help="Override device (e.g., cuda or cpu)")
    parser.add_argument("--dimension", type=int, default=None, help="Override dimension (C and R) directly")
    args_distillation = parser.parse_args()

    if args_distillation.seed is not None:
        _set_seed(args_distillation.seed)

    config = load_config(args_distillation.config)
    logger = MLogger(os.path.join(args_distillation.save_path, "distillation.log"))

    # Check for dimension_info.pkl (created by multi_dimension_train.py)
    dimension_override = args_distillation.dimension
    dim_info_path = os.path.join(args_distillation.save_path, "dimension_info.pkl")
    if dimension_override is None and os.path.exists(dim_info_path):
        dim_info = joblib.load(dim_info_path)
        dimension_override = dim_info.get("dimension")
        logger.info(f"Loaded dimension from dimension_info.pkl: {dimension_override}")

    args = _build_args_from_config(config, dimension_override=dimension_override)
    logger.info(f"Using distillation parameters: C={args.C}, R={args.R}")

    device = args_distillation.device or config["Train"]["Meta_learning"]["Model_parameter"]["device"]
    run_distillation(
        save_path=args_distillation.save_path,
        args=args,
        model_config=MODEL_CONFIG_MAP[args_distillation.model],
        logger=logger,
        device=device,
        resume=args_distillation.resume,
        checkpoint_path=args_distillation.checkpoint_path,
        max_epochs=args_distillation.max_epochs,
        patience=args_distillation.patience,
        min_delta=args_distillation.min_delta,
    )


if __name__ == "__main__":
    main()
