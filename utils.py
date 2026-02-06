"""Utility module for configuration management and training setup."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.transforms import Compose

from transforms import transform_dict


class Config:
    """Global configuration class with nested configuration sections."""
    
    cfg_file: str = ''

    class Dataset:
        """Dataset configuration section."""
        path: Optional[str] = None
        dataname: Optional[str] = None
        node_attr_dim: Optional[int] = None
        edge_attr_dim: Optional[int] = None
        full_pre_transform: Optional[dict] = None  # Transformed data saved for next loading
        inmemory_transform: Optional[dict] = None  # Transform before training
        onthefly_transform: Optional[dict] = None  # Transform during training
        poly_method: Optional[str] = None  # Name of polynomials for graph structural encoding
        poly_dim: Optional[int] = None  # Dimension of graph structural encoding
        follow_batch: Optional[List[str]] = None  # Creates assignment batch vectors

    class Model:
        """Model architecture configuration section."""
        # Basic configuration
        name: str = ""
        node_encoder: Optional[str] = None
        edge_encoder: Optional[str] = None
        pe_encoder: Optional[str] = None

        hidden_dim: int = 0
        layer_type: Optional[str] = None
        num_layers: int = 0
        mlp_depth: int = 0
        pooling: str = ""
        drop_prob: float = 0.0
        output_drop_prob: float = 0.0
        jk_mode: str = ""
        task_type: str = ""
        num_tasks: int = 0
        norm_type: str = ""
        act_type: str = "relu"
        agg: str = ""
        ffn: bool = True
        residual: bool = True

        # DenseInputEncoder configuration
        max_num_nodes: Optional[int] = None

        # Graph Transformer configuration
        attn_heads: int = 0
        attn_drop_prob: float = 0.20
        clamp: float = 5.0
        weight_fn: str = "softmax"
        degree_scaler: str = "log(1+deg)"

    class Train:
        """Training configuration section."""
        runs: int = 10
        save_dir: str = "results"
        num_workers: int = 0
        batch_size: int = 64
        lr: float = 1e-3  # Learning rate
        min_lr: float = 1e-6  # Minimal learning rate
        weight_decay: float = 1e-5
        num_warmup_epochs: int = 50
        num_epochs: int = 2000
        test_eval_interval: int = 10
        optimizer: str = "AdamW"
        scheduler: str = "cosine_with_warmup"
        accumulate_grad_batches: int = 1
        log_every_n_steps: int = 50
        ckpt_period: Optional[int] = None

    class Log:
        """Logging configuration section."""
        local_log: bool = False
        monitor: str = 'val/mae'
        monitor_mode: str = 'min'

    dataset = Dataset()
    model = Model()
    train = Train()
    log = Log()


# Global config instance
cfg = Config()


def _convert_value(val: Any) -> Union[int, float, Any]:
    """Convert value to appropriate type (int, float, or original).
    
    Args:
        val: Value to convert
        
    Returns:
        Converted value as int, float, or original type
    """
    try:
        float_val = float(val)
        return int(float_val) if float_val.is_integer() else float_val
    except (ValueError, TypeError):
        return val


def _build_transforms(transform_config: Dict[str, Dict], custom_args: Dict) -> Union[Compose, Any]:
    """Build transformation pipeline from configuration.
    
    Args:
        transform_config: Dictionary of transform names and parameters
        custom_args: Additional arguments to inject into transforms
        
    Returns:
        Single transform or Compose of multiple transforms
    """
    transforms = []
    for transform_name, params in transform_config.items():
        params.update(custom_args)
        transforms.append(transform_dict[transform_name](**params))
    
    return Compose(transforms) if len(transforms) > 1 else transforms[0] if transforms else None


def load_cfg(args: argparse.Namespace) -> Config:
    """Load configuration from YAML file and command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Populated Config object
    """
    with open(args.cfg_file, "r") as file:
        cfg_dict = yaml.safe_load(file)

    cfg.log.local_log = args.local_log
    args_dict = vars(args)
    
    # Store cfg_file before processing other attributes
    if 'cfg_file' in args_dict and args_dict['cfg_file'] is not None:
        cfg.cfg_file = args_dict['cfg_file']
    
    # Get all class attributes (including nested config objects)
    # Use dir() to get all attributes, filter out private/magic methods
    cfg_attrs = {attr: getattr(cfg, attr) for attr in dir(cfg) 
                 if not attr.startswith('_') and not callable(getattr(cfg, attr))}
    
    
    # Update configuration from YAML and command line arguments
    for key1, val1 in cfg_attrs.items():
        # Skip cfg_file as it's already processed
        if key1 == 'cfg_file':
            continue

        # Handle simple string/scalar attributes at root level
        if isinstance(val1, (str, int, float, bool)) or val1 is None:
            # First apply config from YAML file
            if key1 in cfg_dict:
                setattr(cfg, key1, _convert_value(cfg_dict[key1]))
            # Then override with command line arguments if provided
            if key1 in args_dict and args_dict[key1] is not None:
                setattr(cfg, key1, _convert_value(args_dict[key1]))

        # Handle nested configuration objects (Dataset, Model, Train, Log)
        elif hasattr(val1, "__class__"):
            # Get all attributes of the nested config object, including class variables
            nested_attrs = {k: v for k, v in vars(val1.__class__).items() 
                          if not k.startswith('_') and not callable(v)}
            
            for key2 in nested_attrs.keys():
                # First apply config from YAML file
                if key1 in cfg_dict and isinstance(cfg_dict[key1], dict) and key2 in cfg_dict[key1]:
                    setattr(getattr(cfg, key1), key2, _convert_value(cfg_dict[key1][key2]))
                # Then override with command line arguments if provided
                if key2 in args_dict and args_dict[key2] is not None:
                    setattr(getattr(cfg, key1), key2, _convert_value(args_dict[key2]))

    # Build custom arguments for transforms
    custom_args = {'poly_dim': cfg.dataset.poly_dim}
    if cfg.dataset.poly_method is not None:
        custom_args['poly_method'] = cfg.dataset.poly_method

    # Build transformation pipelines
    if cfg.dataset.full_pre_transform is not None:
        cfg.dataset.full_pre_transform = _build_transforms(cfg.dataset.full_pre_transform, custom_args)

    if cfg.dataset.onthefly_transform is not None:
        cfg.dataset.onthefly_transform = _build_transforms(cfg.dataset.onthefly_transform, custom_args)

    if cfg.dataset.inmemory_transform is not None:
        cfg.dataset.inmemory_transform = _build_transforms(cfg.dataset.inmemory_transform, custom_args)

    # Print configuration summary
    print("\n" + "="*80)
    print("Configuration Summary:")
    print("="*80)
    print(f"Config File: {cfg.cfg_file}")
    print(f"\nDataset:")
    print(f"  - Polynomial Method: {cfg.dataset.poly_method}")
    print(f"  - Polynomial Dimension: {cfg.dataset.poly_dim}")
    print(f"\nModel:")
    print(f"  - Name: {cfg.model.name}")
    print(f"  - Hidden Dimension: {cfg.model.hidden_dim}")
    print(f"  - Layer Type: {cfg.model.layer_type}")
    print(f"  - Number of Layers: {cfg.model.num_layers}")
    print(f"  - Task Type: {cfg.model.task_type}")
    print(f"  - Number of Tasks: {cfg.model.num_tasks}")
    print(f"\nTraining:")
    print(f"  - Runs: {cfg.train.runs}")
    print(f"  - Epochs: {cfg.train.num_epochs}")
    print(f"  - Batch Size: {cfg.train.batch_size}")
    print(f"  - Learning Rate: {cfg.train.lr}")
    print(f"  - Weight Decay: {cfg.train.weight_decay}")
    print(f"  - Optimizer: {cfg.train.optimizer}")
    print(f"  - Scheduler: {cfg.train.scheduler}")
    print("="*80 + "\n")
    return cfg


def sanitize_path(path: Union[str, Path]) -> str:
    """Ensure path is safe and normalized.
    
    Args:
        path: Input path
        
    Returns:
        Normalized absolute path string
    """
    return str(Path(path).resolve())


def _build_config_label() -> str:
    """Build configuration label string from current config.
    
    Returns:
        Configuration label string
    """
    cfg_label = f"{Path(cfg.cfg_file).stem}"
    
    if cfg.dataset.poly_method is not None:
        cfg_label += f".{cfg.dataset.poly_method}"
    if cfg.dataset.poly_dim is not None:
        cfg_label += f".{cfg.dataset.poly_dim}"
    if cfg.model.num_layers is not None:
        cfg_label += f".L{cfg.model.num_layers}"
    if cfg.model.hidden_dim is not None:
        cfg_label += f".H{cfg.model.hidden_dim}"
    
    return cfg_label


def create_trainer(
    timestamp: str,
    run_label: str,
    timer: Any,
    enable_progress_bar: bool = False
) -> Trainer:
    """Create PyTorch Lightning trainer with configured callbacks.
    
    Args:
        timestamp: Timestamp string for save directory
        run_label: Label for this training run
        timer: Timer callback instance
        enable_progress_bar: Whether to enable progress bar
        
    Returns:
        Configured Trainer instance
    """
    machine = os.getenv("MACHINE", "")
    if not machine:
        print("Warning: Environment variable 'MACHINE' is not set.")

    cfg_label = _build_config_label()
    save_subdir = sanitize_path(f"{cfg.train.save_dir}/{timestamp}")
    os.makedirs(save_subdir, exist_ok=True)

    logger = CSVLogger(
        save_dir=save_subdir,
        name=f"{run_label}-{cfg_label}",
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.num_epochs,
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        logger=logger,
        gradient_clip_algorithm="norm",
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(
                monitor=cfg.log.monitor,
                mode=cfg.log.monitor_mode,
                every_n_epochs=cfg.train.ckpt_period
            ),
            timer
        ],
        log_every_n_steps=cfg.train.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
    )
    
    return trainer


def log_final_results(model: Any, results_allruns: Dict[str, List[float]], total_runs: int) -> None:
    """Log final results with statistics across multiple runs.
    
    Args:
        model: Trained model instance
        results_allruns: Dictionary of metric names to lists of values
        total_runs: Total number of runs
    """
    print('\n' * 2)
    print(model)
    
    max_memory_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    print(f"torch.cuda.max_memory_reserved: {max_memory_gb:.1f}GB")

    max_key_len = max(len(key) for key in results_allruns.keys())
    
    print('\n' * 2)
    for key, vals in results_allruns.items():
        padding = ' ' * (max_key_len + 2 - len(key))
        print(f"{key}:{padding}{vals[-1]:.5f}")

    current_runs = len(results_allruns['final/avg_train_time_epoch'])
    if total_runs > 1 and current_runs == total_runs:
        print('\n' * 2)
        print(f"Total {total_runs} runs:")
        for key, vals in results_allruns.items():
            padding = ' ' * (max_key_len + 2 - len(key))
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            print(f"{key}:{padding}{mean_val:.5f} +- {std_val:.5f}")
