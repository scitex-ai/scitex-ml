"""Tests for scitex_ml.loss._L1L2Losses module.

NOTE: The current implementation requires CUDA as it hardcodes .cuda() calls.
All tests will skip if CUDA is not available.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np
import torch.nn as nn

from scitex_ml.loss import elastic, l1, l2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for current implementation (hardcoded .cuda() in source)",
)


class SimpleModel(nn.Module):
    """Simple model for testing regularization."""

    def __init__(self, input_dim=10, hidden_dim=5, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class BNModel(nn.Module):
    """Model with BatchNorm for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return self.fc2(x)


# ── fixtures ───────────────────────────────────────────────────

@pytest.fixture
def model():
    """Create a simple CUDA model for testing."""
    torch.manual_seed(42)
    return SimpleModel().cuda()


# ── L1 loss tests ──────────────────────────────────────────────

def test_l1_basic_returns_tensor(model):
    # Arrange
    # Act
    loss = l1(model)
    # Assert
    assert isinstance(loss, torch.Tensor)


def test_l1_loss_is_scalar(model):
    # Arrange
    # Act
    loss = l1(model)
    # Assert
    assert loss.dim() == 0


def test_l1_loss_is_positive_on_random_model(model):
    # Arrange
    # Act
    loss = l1(model)
    # Assert
    assert loss.item() > 0


def test_l1_loss_resides_on_cuda_device(model):
    # Arrange
    # Act
    loss = l1(model)
    # Assert
    assert loss.is_cuda


def test_l1_lambda_parameter_does_not_affect_loss_value(model):
    # Arrange
    # Act
    loss1 = l1(model, lambda_l1=0.01)
    loss2 = l1(model, lambda_l1=0.1)
    # Assert
    assert torch.allclose(loss1, loss2)


def test_l1_loss_requires_grad(model):
    # Arrange
    # Act
    loss = l1(model)
    # Assert
    assert loss.requires_grad


def test_l1_backward_populates_grad_on_all_parameters(model):
    # Arrange
    loss = l1(model)
    # Act
    loss.backward()
    # Assert
    for param in model.parameters():
        assert param.grad is not None


def test_l1_zero_parameters_return_zero_loss():
    # Arrange
    model = SimpleModel().cuda()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    # Act
    loss = l1(model)
    # Assert
    assert loss.item() == 0.0


def test_l1_larger_model_has_greater_loss():
    # Arrange
    torch.manual_seed(42)
    small_model = SimpleModel(5, 3, 1).cuda()
    torch.manual_seed(42)
    large_model = SimpleModel(100, 50, 10).cuda()
    # Act
    small_loss = l1(small_model)
    large_loss = l1(large_model)
    # Assert
    assert large_loss.item() > small_loss.item()


def test_l1_numerically_stable_with_tiny_parameters(model):
    # Arrange
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1e-8)
    # Act
    loss = l1(model)
    # Assert
    assert torch.isfinite(loss)


def test_l1_numerically_stable_no_nans_with_tiny_parameters(model):
    # Arrange
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(1e-8)
    # Act
    loss = l1(model)
    # Assert
    assert not torch.isnan(loss)


def test_l1_computed_matches_manual_sum_of_abs(model):
    # Arrange
    expected_l1 = sum(torch.abs(param).sum().item() for param in model.parameters())
    # Act
    computed_l1 = l1(model).item()
    # Assert
    assert abs(expected_l1 - computed_l1) < 1e-5


def test_l1_regularization_reduces_loss_over_training_steps(model):
    # Arrange
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 2).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    initial_l1 = l1(model).item()
    # Act
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y) + 0.1 * l1(model)
        loss.backward()
        optimizer.step()
    final_l1 = l1(model).item()
    # Assert
    assert final_l1 < initial_l1


# ── L2 loss tests ──────────────────────────────────────────────

def test_l2_basic_returns_tensor(model):
    # Arrange
    # Act
    loss = l2(model)
    # Assert
    assert isinstance(loss, torch.Tensor)


def test_l2_loss_is_scalar(model):
    # Arrange
    # Act
    loss = l2(model)
    # Assert
    assert loss.dim() == 0


def test_l2_loss_is_positive_on_random_model(model):
    # Arrange
    # Act
    loss = l2(model)
    # Assert
    assert loss.item() > 0


def test_l2_lambda_parameter_does_not_affect_loss_value(model):
    # Arrange
    # Act
    loss1 = l2(model, lambda_l2=0.01)
    loss2 = l2(model, lambda_l2=0.1)
    # Assert
    assert torch.allclose(loss1, loss2)


def test_l2_loss_requires_grad(model):
    # Arrange
    # Act
    loss = l2(model)
    # Assert
    assert loss.requires_grad


def test_l2_backward_populates_grad_on_all_parameters(model):
    # Arrange
    loss = l2(model)
    # Act
    loss.backward()
    # Assert
    for param in model.parameters():
        assert param.grad is not None


def test_l2_computed_matches_manual_sum_of_norms(model):
    # Arrange
    expected_l2 = sum(
        torch.norm(param).sum().item() for param in model.parameters()
    )
    # Act
    computed_l2 = l2(model).item()
    # Assert
    assert abs(expected_l2 - computed_l2) < 1e-5


def test_l2_orthogonal_init_returns_positive_loss():
    # Arrange
    model = SimpleModel().cuda()
    for param in model.parameters():
        if param.dim() >= 2:
            nn.init.orthogonal_(param)
    # Act
    loss = l2(model)
    # Assert
    assert loss.item() > 0


def test_l2_zero_parameters_return_zero_loss():
    # Arrange
    model = SimpleModel().cuda()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    # Act
    loss = l2(model)
    # Assert
    assert loss.item() == 0.0


# ── elastic net tests ──────────────────────────────────────────

def test_elastic_basic_returns_tensor(model):
    # Arrange
    # Act
    loss = elastic(model)
    # Assert
    assert isinstance(loss, torch.Tensor)


def test_elastic_loss_is_scalar(model):
    # Arrange
    # Act
    loss = elastic(model)
    # Assert
    assert loss.dim() == 0


def test_elastic_basic_is_positive(model):
    # Arrange
    # Act
    loss = elastic(model)
    # Assert
    assert loss.item() > 0


def test_elastic_pure_l1_equals_l1_loss(model):
    # Arrange
    # Act
    elastic_loss = elastic(model, alpha=1.0, l1_ratio=1.0)
    l1_loss = l1(model)
    # Assert
    assert torch.allclose(elastic_loss, l1_loss, rtol=1e-5)


def test_elastic_pure_l2_equals_l2_loss(model):
    # Arrange
    # Act
    elastic_loss = elastic(model, alpha=1.0, l1_ratio=0.0)
    l2_loss = l2(model)
    # Assert
    assert torch.allclose(elastic_loss, l2_loss, rtol=1e-5)


def test_elastic_balanced_mix_equals_expected_formula(model):
    # Arrange
    l1_loss = l1(model)
    l2_loss = l2(model)
    expected = 0.5 * l1_loss + 0.5 * l2_loss
    # Act
    elastic_loss = elastic(model, alpha=1.0, l1_ratio=0.5)
    # Assert
    assert torch.allclose(elastic_loss, expected, rtol=1e-5)


def test_elastic_alpha_scaling_doubles_the_loss(model):
    # Arrange
    # Act
    loss1 = elastic(model, alpha=1.0, l1_ratio=0.5)
    loss2 = elastic(model, alpha=2.0, l1_ratio=0.5)
    # Assert
    assert torch.allclose(loss2, 2.0 * loss1, rtol=1e-5)


def test_elastic_negative_l1_ratio_raises_assertion_error(model):
    # Arrange
    # Act
    ctx = pytest.raises(AssertionError)
    # Assert
    with ctx:
        elastic(model, alpha=1.0, l1_ratio=-0.1)


def test_elastic_l1_ratio_above_one_raises_assertion_error(model):
    # Arrange
    # Act
    ctx = pytest.raises(AssertionError)
    # Assert
    with ctx:
        elastic(model, alpha=1.0, l1_ratio=1.1)


def test_elastic_gradient_backward_populates_grad(model):
    # Arrange
    loss = elastic(model, alpha=1.0, l1_ratio=0.7)
    # Act
    loss.backward()
    # Assert
    for param in model.parameters():
        assert param.grad is not None


def test_elastic_different_l1_ratios_produce_distinct_losses(model):
    # Arrange
    ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    # Act
    losses = [elastic(model, alpha=1.0, l1_ratio=ratio).item() for ratio in ratios]
    # Assert
    assert len(set(losses)) == len(losses)


def test_elastic_regularization_reduces_loss_over_training_steps(model):
    # Arrange
    x = torch.randn(10, 10).cuda()
    y = torch.randn(10, 2).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial_loss = elastic(model, alpha=1.0, l1_ratio=0.5).item()
    # Act
    for _ in range(20):
        optimizer.zero_grad()
        output = model(x)
        reg_loss = elastic(model, alpha=0.01, l1_ratio=0.5)
        total_loss = criterion(output, y) + reg_loss
        total_loss.backward()
        optimizer.step()
    final_loss = elastic(model, alpha=1.0, l1_ratio=0.5).item()
    # Assert
    assert final_loss < initial_loss


# ── integration tests ──────────────────────────────────────────

def test_elastic_consistent_with_l1_and_l2_formula():
    # Arrange
    torch.manual_seed(42)
    model = SimpleModel().cuda()
    l1_loss = l1(model)
    l2_loss = l2(model)
    expected = 0.5 * l1_loss + 0.5 * l2_loss
    # Act
    elastic_loss = elastic(model, alpha=1.0, l1_ratio=0.5)
    # Assert
    assert torch.allclose(elastic_loss, expected, rtol=1e-5)


def test_regularizers_are_finite_on_batchnorm_model():
    # Arrange
    model = BNModel().cuda()
    # Act
    l1_loss = l1(model)
    l2_loss = l2(model)
    elastic_loss = elastic(model)
    # Assert
    assert all(torch.isfinite(loss) for loss in [l1_loss, l2_loss, elastic_loss])


def test_l1_empty_model_returns_zero():
    # Arrange
    class EmptyModel(nn.Module):
        def forward(self, x):
            return x
    model = EmptyModel().cuda()
    # Act
    loss = l1(model)
    # Assert
    assert loss.item() == 0.0


def test_l2_empty_model_returns_zero():
    # Arrange
    class EmptyModel(nn.Module):
        def forward(self, x):
            return x
    model = EmptyModel().cuda()
    # Act
    loss = l2(model)
    # Assert
    assert loss.item() == 0.0


def test_elastic_empty_model_returns_zero():
    # Arrange
    class EmptyModel(nn.Module):
        def forward(self, x):
            return x
    model = EmptyModel().cuda()
    # Act
    loss = elastic(model)
    # Assert
    assert loss.item() == 0.0


def test_l1_dropout_model_eval_mode_returns_positive_loss():
    # Arrange
    class DropoutModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(20, 5)
        def forward(self, x):
            x = self.fc1(x)
            x = self.dropout(x)
            return self.fc2(x)
    model = DropoutModel().cuda()
    model.eval()
    # Act
    l1_loss = l1(model)
    # Assert
    assert l1_loss.item() > 0


def test_l1_regularizer_returns_cuda_tensor():
    # Arrange
    model = SimpleModel().cuda()
    # Act
    loss = l1(model)
    # Assert
    assert loss.is_cuda


def test_l2_regularizer_returns_cuda_tensor():
    # Arrange
    model = SimpleModel().cuda()
    # Act
    loss = l2(model)
    # Assert
    assert loss.is_cuda


def test_elastic_regularizer_returns_cuda_tensor():
    # Arrange
    model = SimpleModel().cuda()
    # Act
    loss = elastic(model)
    # Assert
    assert loss.is_cuda
