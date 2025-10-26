"""
Pytest configuration and shared fixtures for model tests.
"""
import pytest
import torch
import numpy as np


@pytest.fixture
def sample_observation_2d():
    """Generate sample 2D observation (batch_size, channels, length)."""
    batch_size = 4
    channels = 3
    length = 100
    return torch.randn(batch_size, channels, length)


@pytest.fixture
def sample_observation_flattened():
    """Generate sample flattened observation."""
    batch_size = 4
    features = 50
    return torch.randn(batch_size, features)


@pytest.fixture
def sample_features():
    """Generate sample feature vectors from backbone."""
    batch_size = 4
    feature_dim = 64
    return torch.randn(batch_size, feature_dim)


@pytest.fixture
def device():
    """Get available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Validation helper functions
def validate_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, allow_batch_dim: bool = True):
    """
    Validate tensor shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (excluding batch dimension if allow_batch_dim=True)
        allow_batch_dim: If True, only check dimensions after batch dimension
    """
    assert isinstance(tensor, torch.Tensor), "Output must be a torch.Tensor"
    
    if allow_batch_dim:
        # Check all dimensions except batch
        assert tensor.shape[1:] == expected_shape, \
            f"Shape mismatch: got {tensor.shape[1:]}, expected {expected_shape}"
    else:
        assert tensor.shape == expected_shape, \
            f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"


def validate_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype = torch.float32):
    """Validate tensor data type."""
    assert tensor.dtype == expected_dtype, \
        f"Data type mismatch: got {tensor.dtype}, expected {expected_dtype}"


def validate_tensor_finite(tensor: torch.Tensor):
    """Check that tensor contains only finite values (no NaN or inf)."""
    assert torch.isfinite(tensor).all(), "Tensor contains non-finite values (NaN or inf)"


def validate_tensor_range(tensor: torch.Tensor, min_val: float = None, max_val: float = None):
    """
    Validate that tensor values are within specified range.
    
    Args:
        tensor: Tensor to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
    """
    if min_val is not None:
        assert tensor.min() >= min_val, \
            f"Tensor contains values below minimum: {tensor.min().item()} < {min_val}"
    
    if max_val is not None:
        assert tensor.max() <= max_val, \
            f"Tensor contains values above maximum: {tensor.max().item()} > {max_val}"


def validate_probabilities(probs: torch.Tensor, dim: int = -1):
    """
    Validate that tensor represents valid probabilities.
    
    Args:
        probs: Probability tensor
        dim: Dimension along which probabilities should sum to 1
    """
    # Check range [0, 1]
    validate_tensor_range(probs, min_val=0.0, max_val=1.0)
    
    # Check sum to 1 along specified dimension
    prob_sums = probs.sum(dim=dim)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
        f"Probabilities don't sum to 1: {prob_sums}"


def validate_logits(logits: torch.Tensor):
    """Validate logits (unbounded real values)."""
    # Logits should be finite but can be any real number
    validate_tensor_finite(logits)


def validate_actions(actions: torch.Tensor, n_actions: int):
    """
    Validate action indices.
    
    Args:
        actions: Action tensor (integer indices)
        n_actions: Number of possible actions
    """
    assert actions.dtype in [torch.long, torch.int, torch.int32, torch.int64], \
        f"Actions should be integer type, got {actions.dtype}"
    
    validate_tensor_range(actions, min_val=0, max_val=n_actions - 1)


def validate_values(values: torch.Tensor):
    """Validate value predictions (unbounded real values)."""
    # Values should be finite real numbers
    validate_tensor_finite(values)

