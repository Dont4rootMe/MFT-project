"""
Tests for model architectures and output validation.

Tests cover:
- Model initialization
- Forward pass output shapes
- Output value ranges and types
- Gradient flow
- Model predictions validity
"""
import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from pipelines.rl_agent_policy.models.cnn import CNNBackbone
from pipelines.rl_agent_policy.models.mlp import MLPBackbone
from pipelines.rl_agent_policy.models.heads import ActorHead, CriticHead
from pipelines.rl_agent_policy.models.tests.conftest import (
    validate_tensor_shape,
    validate_tensor_dtype,
    validate_tensor_finite,
    validate_tensor_range,
    validate_probabilities,
    validate_logits,
    validate_actions,
    validate_values
)


class TestCNNBackbone:
    """Test CNN backbone architecture."""
    
    def test_initialization(self):
        """Test CNN backbone can be initialized."""
        input_shape = (3, 100)
        model = CNNBackbone(input_shape)
        
        assert model is not None
        assert hasattr(model, 'output_dim')
        assert model.output_dim > 0
    
    def test_forward_pass_shape(self, sample_observation_2d):
        """Test forward pass produces correct output shape."""
        batch_size = sample_observation_2d.shape[0]
        input_shape = sample_observation_2d.shape[1:]
        
        model = CNNBackbone(input_shape)
        output = model(sample_observation_2d)
        
        # Should output (batch_size, output_dim)
        validate_tensor_shape(output, (model.output_dim,))
        assert output.shape[0] == batch_size
    
    def test_forward_pass_finite(self, sample_observation_2d):
        """Test forward pass produces finite values."""
        input_shape = sample_observation_2d.shape[1:]
        model = CNNBackbone(input_shape)
        
        output = model(sample_observation_2d)
        validate_tensor_finite(output)
    
    def test_forward_pass_dtype(self, sample_observation_2d):
        """Test forward pass produces float32 output."""
        input_shape = sample_observation_2d.shape[1:]
        model = CNNBackbone(input_shape)
        
        output = model(sample_observation_2d)
        validate_tensor_dtype(output, torch.float32)
    
    def test_gradient_flow(self, sample_observation_2d):
        """Test gradients flow through the network."""
        input_shape = sample_observation_2d.shape[1:]
        model = CNNBackbone(input_shape)
        
        sample_observation_2d.requires_grad_(True)
        output = model(sample_observation_2d)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert sample_observation_2d.grad is not None
        assert not torch.isnan(sample_observation_2d.grad).any()
    
    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        input_shape = (3, 100)
        model = CNNBackbone(input_shape)
        
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, *input_shape)
            output = model(x)
            assert output.shape == (batch_size, model.output_dim)
    
    def test_output_dim_consistency(self):
        """Test output_dim attribute matches actual output."""
        input_shape = (3, 100)
        model = CNNBackbone(input_shape)
        
        x = torch.randn(2, *input_shape)
        output = model(x)
        
        assert output.shape[1] == model.output_dim


class TestMLPBackbone:
    """Test MLP backbone architecture."""
    
    def test_initialization(self):
        """Test MLP backbone can be initialized."""
        input_shape = (5, 20)
        model = MLPBackbone(input_shape)
        
        assert model is not None
        assert hasattr(model, 'output_dim')
        assert model.output_dim == 64
    
    def test_forward_pass_shape(self, sample_observation_2d):
        """Test forward pass produces correct output shape."""
        batch_size = sample_observation_2d.shape[0]
        input_shape = sample_observation_2d.shape[1:]
        
        model = MLPBackbone(input_shape)
        output = model(sample_observation_2d)
        
        # Should output (batch_size, 64)
        validate_tensor_shape(output, (64,))
        assert output.shape[0] == batch_size
    
    def test_forward_pass_finite(self, sample_observation_2d):
        """Test forward pass produces finite values."""
        input_shape = sample_observation_2d.shape[1:]
        model = MLPBackbone(input_shape)
        
        output = model(sample_observation_2d)
        validate_tensor_finite(output)
    
    def test_forward_pass_non_negative(self, sample_observation_2d):
        """Test ReLU activations produce non-negative values."""
        input_shape = sample_observation_2d.shape[1:]
        model = MLPBackbone(input_shape)
        
        output = model(sample_observation_2d)
        # MLP uses ReLU, so output should be non-negative
        validate_tensor_range(output, min_val=0.0)
    
    def test_gradient_flow(self, sample_observation_2d):
        """Test gradients flow through the network."""
        input_shape = sample_observation_2d.shape[1:]
        model = MLPBackbone(input_shape)
        
        sample_observation_2d.requires_grad_(True)
        output = model(sample_observation_2d)
        loss = output.sum()
        loss.backward()
        
        assert sample_observation_2d.grad is not None
        assert not torch.isnan(sample_observation_2d.grad).any()


class TestActorHead:
    """Test Actor head for action prediction."""
    
    def test_initialization(self):
        """Test Actor head can be initialized."""
        in_dim = 64
        n_actions = 5
        model = ActorHead(in_dim, n_actions)
        
        assert model is not None
    
    def test_forward_pass_shape(self, sample_features):
        """Test forward pass produces correct output shape."""
        batch_size = sample_features.shape[0]
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        logits = model(sample_features)
        
        # Should output (batch_size, n_actions)
        validate_tensor_shape(logits, (n_actions,))
        assert logits.shape[0] == batch_size
    
    def test_logits_are_valid(self, sample_features):
        """Test that output logits are valid (finite, unbounded)."""
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        logits = model(sample_features)
        
        validate_logits(logits)
    
    def test_logits_to_probabilities(self, sample_features):
        """Test conversion from logits to probabilities."""
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        logits = model(sample_features)
        
        # Convert to probabilities using softmax
        probs = F.softmax(logits, dim=-1)
        
        validate_probabilities(probs, dim=-1)
    
    def test_probabilities_to_actions(self, sample_features):
        """Test conversion from probabilities to action indices."""
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        logits = model(sample_features)
        probs = F.softmax(logits, dim=-1)
        
        # Deterministic action selection (argmax)
        actions_deterministic = probs.argmax(dim=-1)
        validate_actions(actions_deterministic, n_actions)
        
        # Stochastic action selection (sampling)
        actions_stochastic = torch.multinomial(probs, num_samples=1).squeeze(-1)
        validate_actions(actions_stochastic, n_actions)
    
    def test_action_probabilities_sum_to_one(self, sample_features):
        """Test that action probabilities sum to 1."""
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        logits = model(sample_features)
        probs = F.softmax(logits, dim=-1)
        
        # Sum across actions dimension
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            "Action probabilities don't sum to 1"
    
    def test_log_probabilities(self, sample_features):
        """Test log probability computation."""
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        logits = model(sample_features)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Log probs should be <= 0
        validate_tensor_range(log_probs, max_val=0.0)
        validate_tensor_finite(log_probs)
    
    def test_gradient_flow(self, sample_features):
        """Test gradients flow through Actor head."""
        in_dim = sample_features.shape[1]
        n_actions = 5
        
        model = ActorHead(in_dim, n_actions)
        sample_features.requires_grad_(True)
        
        logits = model(sample_features)
        loss = logits.sum()
        loss.backward()
        
        assert sample_features.grad is not None


class TestCriticHead:
    """Test Critic head for value prediction."""
    
    def test_initialization(self):
        """Test Critic head can be initialized."""
        in_dim = 64
        model = CriticHead(in_dim)
        
        assert model is not None
    
    def test_forward_pass_shape(self, sample_features):
        """Test forward pass produces correct output shape."""
        batch_size = sample_features.shape[0]
        in_dim = sample_features.shape[1]
        
        model = CriticHead(in_dim)
        values = model(sample_features)
        
        # Should output (batch_size, 1)
        validate_tensor_shape(values, (1,))
        assert values.shape[0] == batch_size
    
    def test_values_are_valid(self, sample_features):
        """Test that predicted values are valid (finite, unbounded)."""
        in_dim = sample_features.shape[1]
        
        model = CriticHead(in_dim)
        values = model(sample_features)
        
        validate_values(values)
    
    def test_values_are_scalar_per_batch(self, sample_features):
        """Test that each batch element gets a single scalar value."""
        batch_size = sample_features.shape[0]
        in_dim = sample_features.shape[1]
        
        model = CriticHead(in_dim)
        values = model(sample_features)
        
        # Should be (batch_size, 1)
        assert values.shape == (batch_size, 1)
        
        # Can be squeezed to (batch_size,)
        values_squeezed = values.squeeze(-1)
        assert values_squeezed.shape == (batch_size,)
    
    def test_gradient_flow(self, sample_features):
        """Test gradients flow through Critic head."""
        in_dim = sample_features.shape[1]
        
        model = CriticHead(in_dim)
        sample_features.requires_grad_(True)
        
        values = model(sample_features)
        loss = values.sum()
        loss.backward()
        
        assert sample_features.grad is not None


class TestEndToEndPipeline:
    """Test complete pipeline from observation to action."""
    
    def test_cnn_to_actor_pipeline(self, sample_observation_2d):
        """Test CNN backbone to Actor head pipeline."""
        input_shape = sample_observation_2d.shape[1:]
        n_actions = 5
        
        # Create pipeline
        backbone = CNNBackbone(input_shape)
        actor = ActorHead(backbone.output_dim, n_actions)
        
        # Forward pass
        features = backbone(sample_observation_2d)
        logits = actor(features)
        probs = F.softmax(logits, dim=-1)
        actions = probs.argmax(dim=-1)
        
        # Validate
        validate_probabilities(probs, dim=-1)
        validate_actions(actions, n_actions)
    
    def test_mlp_to_actor_pipeline(self, sample_observation_2d):
        """Test MLP backbone to Actor head pipeline."""
        input_shape = sample_observation_2d.shape[1:]
        n_actions = 5
        
        # Create pipeline
        backbone = MLPBackbone(input_shape)
        actor = ActorHead(backbone.output_dim, n_actions)
        
        # Forward pass
        features = backbone(sample_observation_2d)
        logits = actor(features)
        probs = F.softmax(logits, dim=-1)
        actions = probs.argmax(dim=-1)
        
        # Validate
        validate_probabilities(probs, dim=-1)
        validate_actions(actions, n_actions)
    
    def test_cnn_to_critic_pipeline(self, sample_observation_2d):
        """Test CNN backbone to Critic head pipeline."""
        input_shape = sample_observation_2d.shape[1:]
        
        # Create pipeline
        backbone = CNNBackbone(input_shape)
        critic = CriticHead(backbone.output_dim)
        
        # Forward pass
        features = backbone(sample_observation_2d)
        values = critic(features)
        
        # Validate
        validate_values(values)
        assert values.shape == (sample_observation_2d.shape[0], 1)
    
    def test_actor_critic_together(self, sample_observation_2d):
        """Test Actor and Critic working together (A2C style)."""
        input_shape = sample_observation_2d.shape[1:]
        n_actions = 5
        
        # Create networks
        backbone = CNNBackbone(input_shape)
        actor = ActorHead(backbone.output_dim, n_actions)
        critic = CriticHead(backbone.output_dim)
        
        # Forward pass
        features = backbone(sample_observation_2d)
        logits = actor(features)
        values = critic(features)
        
        # Validate
        validate_logits(logits)
        validate_values(values)
        
        # Convert to probabilities and actions
        probs = F.softmax(logits, dim=-1)
        actions = probs.argmax(dim=-1)
        
        validate_probabilities(probs, dim=-1)
        validate_actions(actions, n_actions)


class TestOutputTransformations:
    """Test various output transformations for API usage."""
    
    def test_logits_to_probs_stable(self):
        """Test numerically stable logits to probabilities conversion."""
        # Create extreme logits to test numerical stability
        logits = torch.tensor([
            [1000.0, 0.0, -1000.0],  # Extreme values
            [0.0, 0.0, 0.0],         # Uniform
            [-10.0, 0.0, 10.0]       # Normal range
        ])
        
        probs = F.softmax(logits, dim=-1)
        
        validate_probabilities(probs, dim=-1)
        validate_tensor_finite(probs)
    
    def test_probs_to_action_deterministic(self):
        """Test deterministic action selection from probabilities."""
        probs = torch.tensor([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.3, 0.4]
        ])
        
        actions = probs.argmax(dim=-1)
        
        expected_actions = torch.tensor([0, 1, 2])
        assert torch.equal(actions, expected_actions)
    
    def test_probs_to_action_stochastic(self):
        """Test stochastic action sampling from probabilities."""
        torch.manual_seed(42)
        
        probs = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ])
        
        # Sample multiple times
        n_samples = 100
        actions_list = []
        for _ in range(n_samples):
            actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
            actions_list.append(actions)
        
        actions_all = torch.stack(actions_list)
        
        # All sampled actions should be valid
        for actions in actions_list:
            validate_actions(actions, n_actions=3)
    
    def test_action_to_one_hot(self):
        """Test conversion of action indices to one-hot encoding."""
        actions = torch.tensor([0, 2, 1])
        n_actions = 3
        
        one_hot = F.one_hot(actions, num_classes=n_actions).float()
        
        expected = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0]
        ])
        
        assert torch.equal(one_hot, expected)
    
    def test_value_normalization(self):
        """Test value prediction normalization for API output."""
        values = torch.tensor([[100.5], [-50.2], [0.0], [1000.0]])
        
        # Normalize to [0, 1] range
        v_min = values.min()
        v_max = values.max()
        values_normalized = (values - v_min) / (v_max - v_min + 1e-8)
        
        validate_tensor_range(values_normalized, min_val=0.0, max_val=1.0)
    
    def test_confidence_score(self):
        """Test extracting confidence score from action probabilities."""
        probs = torch.tensor([
            [0.8, 0.1, 0.1],  # High confidence
            [0.4, 0.3, 0.3],  # Low confidence
        ])
        
        # Confidence = max probability
        confidence = probs.max(dim=-1)[0]
        
        assert confidence[0] > 0.7  # High confidence
        assert confidence[1] < 0.5  # Low confidence
        validate_tensor_range(confidence, min_val=0.0, max_val=1.0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_sample_batch(self):
        """Test models work with batch size of 1."""
        input_shape = (3, 100)
        n_actions = 5
        
        backbone = CNNBackbone(input_shape)
        actor = ActorHead(backbone.output_dim, n_actions)
        
        x = torch.randn(1, *input_shape)
        features = backbone(x)
        logits = actor(features)
        
        assert logits.shape == (1, n_actions)
    
    def test_large_batch(self):
        """Test models work with large batch sizes."""
        input_shape = (3, 100)
        n_actions = 5
        batch_size = 128
        
        backbone = CNNBackbone(input_shape)
        actor = ActorHead(backbone.output_dim, n_actions)
        
        x = torch.randn(batch_size, *input_shape)
        features = backbone(x)
        logits = actor(features)
        
        assert logits.shape == (batch_size, n_actions)
    
    def test_zero_input(self):
        """Test models handle zero input."""
        input_shape = (3, 100)
        
        backbone = CNNBackbone(input_shape)
        x = torch.zeros(2, *input_shape)
        
        output = backbone(x)
        validate_tensor_finite(output)
    
    def test_extreme_input_values(self):
        """Test models handle extreme input values."""
        input_shape = (3, 100)
        
        backbone = CNNBackbone(input_shape)
        
        # Very large values
        x_large = torch.ones(2, *input_shape) * 1000
        output_large = backbone(x_large)
        validate_tensor_finite(output_large)
        
        # Very small values
        x_small = torch.ones(2, *input_shape) * 0.001
        output_small = backbone(x_small)
        validate_tensor_finite(output_small)

