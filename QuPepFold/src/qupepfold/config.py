"""Configuration management for QuPepFold."""

from dataclasses import fields, asdict
from typing import Optional
import yaml

from .types import FoldConfig


def get_default_config(**overrides) -> FoldConfig:
    """Get default FoldConfig with optional overrides.
    
    Args:
        **overrides: Keyword arguments to override defaults.
        
    Returns:
        FoldConfig with applied overrides.
        
    Example:
        config = get_default_config(shots=4000, backend="runtime")
    """
    return FoldConfig(**overrides)


def config_from_yaml(path: str) -> FoldConfig:
    """Load FoldConfig from a YAML file.
    
    Args:
        path: Path to YAML file.
        
    Returns:
        FoldConfig with values from file.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    # Filter to only known FoldConfig fields
    valid_keys = {f.name for f in fields(FoldConfig)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return FoldConfig(**filtered)


def config_to_yaml(config: FoldConfig, path: str) -> None:
    """Save FoldConfig to a YAML file.
    
    Args:
        config: FoldConfig to save.
        path: Output YAML path.
    """
    data = asdict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def config_from_dict(d: dict) -> FoldConfig:
    """Construct FoldConfig from a dictionary.
    
    Silently ignores unknown keys.
    
    Args:
        d: Dictionary of config values.
        
    Returns:
        FoldConfig.
    """
    valid_keys = {f.name for f in fields(FoldConfig)}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return FoldConfig(**filtered)


def validate_config(config: FoldConfig) -> None:
    """Validate a FoldConfig for consistency.
    
    Raises:
        ValueError: If configuration is invalid.
    """
    if config.fragment_length < 3:
        raise ValueError("fragment_length must be >= 3")
    
    if config.overlap_turns < 0:
        raise ValueError("overlap_turns must be >= 0")
    
    if config.overlap_turns >= config.fragment_length - 1:
        raise ValueError("overlap_turns must be < fragment_length - 1")
    
    if config.shots < 1:
        raise ValueError("shots must be >= 1")
    
    if config.backend == "runtime" and not config.ibm_backend:
        raise ValueError("ibm_backend must be specified when backend='runtime'")
    
    if config.warmstart_mode not in ("basis", "biased_ry"):
        raise ValueError("warmstart_mode must be 'basis' or 'biased_ry'")
    
    if not 0 < config.warmstart_bias < 1:
        raise ValueError("warmstart_bias must be in (0, 1)")
    
    # CVaR validation
    if not 0 < config.cvar_alpha <= 1:
        raise ValueError("cvar_alpha must be in (0, 1]")
    
    if config.optimization_tries < 1:
        raise ValueError("optimization_tries must be >= 1")
    
    if config.sa_restarts < 1:
        raise ValueError("sa_restarts must be >= 1")
    
    if not 0 < config.export_prob_threshold < 1:
        raise ValueError("export_prob_threshold must be in (0, 1)")
    
    # Encoding validation
    valid_encodings = {"turn", "hp", "constrained_local"}
    if config.encoding not in valid_encodings:
        raise ValueError(f"encoding must be one of {valid_encodings}, got '{config.encoding}'")
    
    # Optimizer validation
    valid_optimizers = {"spsa", "cobyla", "gradient", "nelder_mead"}
    if config.optimizer not in valid_optimizers:
        raise ValueError(f"optimizer must be one of {valid_optimizers}, got '{config.optimizer}'")
    
    # Stopping criteria
    if config.energy_tolerance < 0:
        raise ValueError("energy_tolerance must be >= 0")
    
    if config.early_stop_patience < 1:
        raise ValueError("early_stop_patience must be >= 1")
    
    if config.max_wall_clock_seconds is not None and config.max_wall_clock_seconds <= 0:
        raise ValueError("max_wall_clock_seconds must be > 0")
    
    # Fragment strategy
    valid_strategies = {"fixed_window", "disorder", "domain", "user_defined"}
    if config.fragment_strategy not in valid_strategies:
        raise ValueError(f"fragment_strategy must be one of {valid_strategies}")
