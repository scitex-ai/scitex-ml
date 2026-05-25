#!/usr/bin/env python3
# Time-stamp: "2025-06-01 15:15:00 (ywatanabe)"
# File: ./tests/scitex/ai/feature_extraction/test_vit.py

"""Tests for scitex_ml.feature_extraction.vit module."""

import pytest

torch = pytest.importorskip("torch")
try:
    import torchvision  # noqa: F401
    import pytorch_pretrained_vit  # noqa: F401
except (ImportError, RuntimeError) as exc:
    pytest.skip(
        f"vit test dependencies unavailable: {exc}", allow_module_level=True
    )
import contextlib
import os

import numpy as np

from scitex_ml.feature_extraction import vit as vit_module
from scitex_ml.feature_extraction.vit import VitFeatureExtractor, _setup_device


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


@contextlib.contextmanager
def _clear_env():
    """Temporarily clear os.environ; restore on exit."""
    saved = dict(os.environ)
    os.environ.clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


class _FakeViT:
    """Hand-rolled fake replacing the ViT model.

    Tracks call counts for eval/to and is callable.
    """

    def __init__(self, *args, **kwargs):
        self.image_size = 224
        self.eval_calls = 0
        self.to_calls = 0
        # Default forward output; tests override via `return_value`.
        self.return_value = torch.randn(1, 1000)

    def eval(self):
        self.eval_calls += 1
        return self

    def to(self, device):
        self.to_calls += 1
        return self

    def __call__(self, x):
        return self.return_value


class _CpuTrackingTensor:
    """Wraps a tensor and tracks .cpu() calls."""

    def __init__(self, tensor):
        self._tensor = tensor
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        return self._tensor

    def reshape(self, *args, **kwargs):
        return self._tensor.reshape(*args, **kwargs)


class TestSetupDevice:
    """Test suite for _setup_device function."""

    def test_setup_device_none_cuda_available(self):
        """Test device setup when None is passed and CUDA is available."""
        with _swap_attr(torch.cuda, "is_available", lambda: True):
            device = _setup_device(None)
            assert device == "cuda"

    def test_setup_device_none_cuda_not_available(self):
        """Test device setup when None is passed and CUDA is not available."""
        with _swap_attr(torch.cuda, "is_available", lambda: False):
            device = _setup_device(None)
            assert device == "cpu"

    def test_setup_device_explicit_cpu(self):
        """Test device setup with explicit CPU."""
        device = _setup_device("cpu")
        assert device == "cpu"

    def test_setup_device_explicit_cuda(self):
        """Test device setup with explicit CUDA."""
        device = _setup_device("cuda")
        assert device == "cuda"


class TestVitFeatureExtractor:
    """Test suite for VitFeatureExtractor class."""

    @pytest.fixture
    def fake_vit_model(self):
        """Create a fake ViT model instance (single shared instance per test)."""
        return _FakeViT()

    @pytest.fixture
    def mock_environment(self, tmp_path, fake_vit_model):
        """Set up fake environment for testing.

        Patches the ViT constructor to return our fake instance and stubs
        os.path.exists so the model directory check always passes.
        """
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        def _vit_factory(*args, **kwargs):
            return fake_vit_model

        with _swap_attr(vit_module, "ViT", _vit_factory):
            with _swap_attr(vit_module._os.path, "exists", lambda *a, **kw: True):
                yield model_dir

    def test_initialization_default_params(self, mock_environment):
        """Test initialization with default parameters."""
        extractor = VitFeatureExtractor()
        assert extractor.model_name == "B_16"
        assert extractor.torch_home == "./models"
        assert extractor.device in ["cpu", "cuda"]

    def test_initialization_custom_params(self, mock_environment):
        """Test initialization with custom parameters."""
        extractor = VitFeatureExtractor(
            model_name="L_32", torch_home=str(mock_environment), device="cpu"
        )
        assert extractor.model_name == "L_32"
        assert extractor.torch_home == str(mock_environment)
        assert extractor.device == "cpu"

    def test_initialization_sets_torch_home(self, mock_environment):
        """Test that initialization sets TORCH_HOME environment variable."""
        torch_home = str(mock_environment)
        with _clear_env():
            _ = VitFeatureExtractor(torch_home=torch_home)
            assert os.environ["TORCH_HOME"] == torch_home

    def test_valid_model_names(self, mock_environment):
        """Test all valid model names."""
        valid_models = [
            "B_16",
            "B_32",
            "L_16",
            "L_32",
            "B_16_imagenet1k",
            "B_32_imagenet1k",
            "L_16_imagenet1k",
            "L_32_imagenet1k",
        ]

        for model_name in valid_models:
            extractor = VitFeatureExtractor(model_name=model_name)
            assert extractor.model_name == model_name

    def test_invalid_model_name_raises_error(self, mock_environment):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model name"):
            VitFeatureExtractor(model_name="invalid_model")

    def test_nonexistent_model_directory_raises_error(self):
        """Test that non-existent model directory raises FileNotFoundError."""
        with _swap_attr(vit_module._os.path, "exists", lambda *a, **kw: False):
            with pytest.raises(FileNotFoundError, match="Model directory not found"):
                VitFeatureExtractor(torch_home="/nonexistent/path")

    def test_transform_pipeline_setup(self, mock_environment):
        """Test that transform pipeline is properly set up."""
        extractor = VitFeatureExtractor()

        # Check that transform is a Compose object
        from torchvision.transforms import Compose

        assert isinstance(extractor.transform, Compose)

        # Check transform steps
        transforms = extractor.transform.transforms
        assert len(transforms) == 4  # ToPILImage, Resize, ToTensor, Normalize

    def test_preprocess_array_2d_input(self, mock_environment):
        """Test preprocessing of 2D array."""
        extractor = VitFeatureExtractor()

        # Create 2D test tensor (H, W)
        arr = torch.randn(32, 32)
        processed, batch_shape = extractor._preprocess_array(
            arr, dim=(-2, -1), channel_dim=None
        )

        assert processed.shape == (1, 3, 224, 224)  # 1 image, 3 channels, resized
        assert batch_shape == ()

    def test_preprocess_array_3d_input(self, mock_environment):
        """Test preprocessing of 3D array with batch dimension."""
        extractor = VitFeatureExtractor()

        # Create 3D test tensor (B, H, W)
        batch_size = 4
        arr = torch.randn(batch_size, 32, 32)
        processed, batch_shape = extractor._preprocess_array(
            arr, dim=(-2, -1), channel_dim=None
        )

        assert processed.shape == (batch_size, 3, 224, 224)
        assert batch_shape == (batch_size,)

    def test_preprocess_array_high_dimensional(self, mock_environment):
        """Test preprocessing of high-dimensional array."""
        extractor = VitFeatureExtractor()

        # Create 6D test tensor
        arr = torch.randn(2, 3, 4, 5, 32, 32)
        processed, batch_shape = extractor._preprocess_array(
            arr, dim=(-2, -1), channel_dim=None
        )

        # Should flatten all but spatial dims
        assert processed.shape == (2 * 3 * 4 * 5, 3, 224, 224)
        assert batch_shape == (2, 3, 4, 5)

    def test_extract_features_simple(self, mock_environment, fake_vit_model):
        """Test feature extraction with simple input."""
        extractor = VitFeatureExtractor()

        # Create test input
        arr = torch.randn(32, 32)

        # Fake model returns features
        fake_vit_model.return_value = torch.randn(1, 1000)

        features = extractor.extract_features(arr, axis=(-2, -1))

        assert features.shape == (1000,)
        assert isinstance(features, torch.Tensor)

    def test_extract_features_batch(self, mock_environment, fake_vit_model):
        """Test feature extraction with batch input."""
        extractor = VitFeatureExtractor()

        # Create batch test input
        batch_size = 4
        arr = torch.randn(batch_size, 32, 32)

        # Fake model returns batch features
        fake_vit_model.return_value = torch.randn(batch_size, 1000)

        features = extractor.extract_features(arr, axis=(-2, -1))

        assert features.shape == (batch_size, 1000)

    def test_extract_features_high_dimensional(self, mock_environment, fake_vit_model):
        """Test feature extraction preserves non-spatial dimensions."""
        extractor = VitFeatureExtractor()

        # Create high-dimensional input
        arr = torch.randn(2, 3, 4, 32, 32)

        # Fake model returns appropriate number of features
        fake_vit_model.return_value = torch.randn(2 * 3 * 4, 1000)

        features = extractor.extract_features(arr, axis=(-2, -1))

        assert features.shape == (2, 3, 4, 1000)

    def test_extract_features_no_grad(self, mock_environment, fake_vit_model):
        """Test that feature extraction runs in no_grad mode."""
        extractor = VitFeatureExtractor()

        arr = torch.randn(32, 32, requires_grad=True)

        # Track if no_grad was used
        no_grad_called = False
        original_no_grad = torch.no_grad

        def tracking_no_grad():
            nonlocal no_grad_called
            no_grad_called = True
            return original_no_grad()

        with _swap_attr(torch, "no_grad", tracking_no_grad):
            _ = extractor.extract_features(arr, axis=(-2, -1))

        assert no_grad_called

    def test_extract_features_device_handling(self, mock_environment, fake_vit_model):
        """Test proper device handling during feature extraction."""
        extractor = VitFeatureExtractor(device="cpu")

        # Just verify the extractor was initialized with correct device
        assert extractor.device == "cpu"

        # The model should have been moved to a device
        assert fake_vit_model.to_calls >= 1

    def test_extract_features_cpu_output(self, mock_environment, fake_vit_model):
        """Test that output is always on CPU."""
        extractor = VitFeatureExtractor(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        arr = torch.randn(32, 32)

        # Wrap model output so we can track .cpu() calls.
        wrapped_output = _CpuTrackingTensor(torch.randn(1, 1000))
        fake_vit_model.return_value = wrapped_output

        features = extractor.extract_features(arr, axis=(-2, -1))

        # Check that .cpu() was called exactly once
        assert wrapped_output.cpu_calls == 1
        # Sanity-check the resulting features.
        assert features.shape == (1000,)

    def test_model_eval_mode(self, mock_environment, fake_vit_model):
        """Test that model is set to eval mode."""
        _ = VitFeatureExtractor()

        # Check that eval was called during initialization
        assert fake_vit_model.eval_calls == 1

    @pytest.mark.parametrize(
        "axis",
        [
            (-2, -1),  # Last two dimensions - standard case
        ],
    )
    def test_different_axis_specifications(
        self, mock_environment, fake_vit_model, axis
    ):
        """Test feature extraction with different axis specifications."""
        extractor = VitFeatureExtractor()

        # Create tensor with spatial dims at end
        arr = torch.randn(3, 4, 32, 32)

        fake_vit_model.return_value = torch.randn(12, 1000)  # 3*4 = 12

        features = extractor.extract_features(arr, axis=axis)

        # Features should have correct shape
        assert features.ndim == 3  # Original 4D - 2 spatial + 1 feature
        assert features.shape[-1] == 1000

    def test_negative_axis_handling(self, mock_environment):
        """Test that negative axis indices are handled correctly."""
        extractor = VitFeatureExtractor()

        arr = torch.randn(2, 3, 32, 32)

        # Test internal preprocessing
        processed, batch_shape = extractor._preprocess_array(
            arr, dim=(-2, -1), channel_dim=None
        )

        # Should correctly identify spatial dimensions
        assert processed.shape[0] == 2 * 3  # Batch flattened
        assert processed.shape[2:] == (224, 224)  # Resized spatial

    def test_transform_grayscale_to_rgb(self, mock_environment):
        """Test that grayscale images are converted to RGB."""
        extractor = VitFeatureExtractor()

        # Single channel input
        arr = torch.randn(1, 32, 32)
        processed, _ = extractor._preprocess_array(arr, dim=(-2, -1), channel_dim=None)

        # Should have 3 channels after processing
        assert processed.shape[1] == 3


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/feature_extraction/vit.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-27 21:36:51 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/ai/feature_extraction/vit.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/ai/feature_extraction/vit.py"
#
# """
# Functionality:
#     Extracts features from images using Vision Transformer (ViT) models
# Input:
#     Image arrays of arbitrary dimensions
# Output:
#     Feature vectors (1000-dimensional embeddings)
# Prerequisites:
#     torch, PIL, torchvision
# """
#
# import os as _os
# from typing import Tuple, Union
#
# import torch
# import torch as _torch
# import numpy as np
# from pytorch_pretrained_vit import ViT
# from torchvision import transforms as _transforms
#
# # from scitex.decorators import batch_torch_fn
#
#
# def _setup_device(device: Union[str, None]) -> str:
#     if device is None:
#         device = "cuda" if _torch.cuda.is_available() else "cpu"
#     return device
#
#
# class VitFeatureExtractor:
#     def __init__(
#         self,
#         model_name="B_16",
#         torch_home="./models",
#         device=None,
#     ):
#         self.valid_models = [
#             "B_16",
#             "B_32",
#             "L_16",
#             "L_32",
#             "B_16_imagenet1k",
#             "B_32_imagenet1k",
#             "L_16_imagenet1k",
#             "L_32_imagenet1k",
#         ]
#         self.model_name = model_name
#         self.torch_home = torch_home
#         self.device = _setup_device(device)
#
#         _os.environ["TORCH_HOME"] = torch_home
#         self._validate_inputs()
#         self.model = ViT(model_name, pretrained=True).to(self.device).eval()
#         self.transform = _transforms.Compose(
#             [
#                 _transforms.ToPILImage(),
#                 _transforms.Resize(self.model.image_size),
#                 _transforms.ToTensor(),
#                 _transforms.Normalize(0.5, 0.5),
#             ]
#         )
#
#     def _validate_inputs(self):
#         if self.model_name not in self.valid_models:
#             raise ValueError(f"Invalid model name. Choose from: {self.valid_models}")
#         if not _os.path.exists(self.torch_home):
#             raise FileNotFoundError(f"Model directory not found: {self.torch_home}")
#
#     def _preprocess_array(
#         self,
#         arr: _torch.Tensor,
#         dim: Tuple[int, int],
#         channel_dim: Union[int, None],
#     ) -> _torch.Tensor:
#         # print(f"Input array shape: {arr.shape}")
#
#         orig_shape = arr.shape
#         dim = tuple(d if d >= 0 else len(orig_shape) + d for d in dim)
#
#         perm = list(range(len(orig_shape)))
#         for d in sorted(dim):
#             perm.remove(d)
#             perm.append(d)
#         arr = arr.permute(perm)
#
#         # Flatten all dimensions except the last two (spatial dimensions)
#         batch_shape = arr.shape[:-2]
#         spatial_shape = arr.shape[-2:]
#         arr = arr.reshape(-1, *spatial_shape)
#
#         # Process each image
#         transformed = []
#         for img in arr:
#             img = img.unsqueeze(0)
#             img = img.repeat(3, 1, 1)
#             transformed.append(self.transform(img))
#         result = _torch.stack(transformed)
#         return result, batch_shape
#
#     # @batch_method
#     # @torch_method
#     # @batch_torch_fn
#     def extract_features(
#         self,
#         arr,
#         axis=(-2, -1),
#         dim=None,
#         channel_dim=None,
#         batch_size=None,
#         device="cuda",
#     ):
#         processed_arr, batch_shape = self._preprocess_array(
#             arr,
#             axis,
#             channel_dim,
#         )
#         # print(f"Processed shape: {processed_arr.shape}")
#
#         processed_arr = processed_arr.to(self.device)
#         with _torch.no_grad():
#             features = self.model(processed_arr).cpu()
#
#         return features.reshape(*batch_shape, -1)
#
#
# if __name__ == "__main__":
#     import scitex
#
#     extractor = scitex_ml.feature_extraction.VitFeatureExtractor(
#         model_name="B_16_imagenet1k"
#     )
#     tensor = torch.randn(3, 2, 4, 5, 32, 32)
#     processed = extractor.extract_features(tensor, (-2, -1), None)
#     print(processed.shape)
#
#     arr = np.random.rand(3, 2, 4, 5, 32, 32)
#     processed = extractor.extract_features(arr, (-2, -1), None)
#     print(processed.shape)
#     # torch.Size([3, 2, 4, 5, 32, 32])
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/ai/feature_extraction/vit.py
# --------------------------------------------------------------------------------
