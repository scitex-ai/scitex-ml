#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_ml/embedding/_ecapa.py
# ----------------------------------------
from __future__ import annotations

"""ECAPA-TDNN speaker embedding for SciTeX Voice V1.

Turns a mono 16 kHz waveform into a fixed-dim speaker embedding. This is
the *inference* half of the verification seam; comparison lives in
``scitex_ml.similarity``, and the admit/discard decision + tunable
threshold live in scitex-audio (never here).

Backend: SpeechBrain's ``spkrec-ecapa-voxceleb`` (192-dim). Weights are
cached locally under ``$SCITEX_ML_ECAPA_DIR`` (default
``~/.cache/scitex-ml/ecapa``) and never leave the WireGuard mesh — the
no-external-API privacy guarantee (non-negotiable #1) applies to models
too.

torch + speechbrain are imported lazily so this module and
``import scitex_ml`` stay usable with neither installed; the helpful
install hint fires at ``load()`` time. Pin note: on Pascal (compute-03's
GTX 1070) use a ``torch<=2.7`` cu12x wheel — 2.8+ dropped sm_61 wheels.
See ``scitex_ml.inference._device``.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..inference import resolve_device

__all__ = ["ECAPAEmbedder", "EMBEDDING_DIM"]

# spkrec-ecapa-voxceleb produces 192-dim embeddings.
EMBEDDING_DIM = 192

_DEFAULT_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_ENV_CACHE = "SCITEX_ML_ECAPA_DIR"
_ENV_SOURCE = "SCITEX_ML_ECAPA_SOURCE"

_INSTALL_HINT = (
    "ECAPA embedding needs torch + speechbrain. Install the voice extra:\n"
    "    pip install 'scitex-ml[voice]'\n"
    "On a Pascal GPU (GTX 1070, compute-03) pin a cu12x wheel: "
    "torch<=2.7 (2.8+ dropped sm_61)."
)


def _cache_dir() -> Path:
    d = os.environ.get(_ENV_CACHE) or (Path.home() / ".cache" / "scitex-ml" / "ecapa")
    return Path(d)


class ECAPAEmbedder:
    """Lazily-loaded ECAPA-TDNN speaker embedder.

    Construct cheaply; the model is loaded on first ``embed`` (or explicit
    ``load()``). Reused across calls so enrolment (many windows) and
    verification (per segment) share one resident model.

    Args:
        source: HuggingFace/SpeechBrain source id or local dir. Defaults to
            ``$SCITEX_ML_ECAPA_SOURCE`` then ``spkrec-ecapa-voxceleb``.
        device: torch device string; ``None`` auto-resolves via
            ``scitex_ml.inference.resolve_device``.
    """

    def __init__(
        self, source: Optional[str] = None, device: Optional[str] = None
    ) -> None:
        self.source = source or os.environ.get(_ENV_SOURCE) or _DEFAULT_SOURCE
        self.device = resolve_device(device)
        self._model = None  # loaded lazily

    def load(self) -> "ECAPAEmbedder":
        """Load the underlying SpeechBrain model (idempotent).

        Raises:
            RuntimeError: if torch/speechbrain are not installed, with an
                install hint (see ``_INSTALL_HINT``).
        """
        if self._model is not None:
            return self
        try:
            from speechbrain.inference.speaker import (  # noqa: PLC0415
                EncoderClassifier,
            )
        except Exception as exc:  # pragma: no cover - exercised without deps
            raise RuntimeError(_INSTALL_HINT) from exc

        savedir = _cache_dir() / self.source.replace("/", "__")
        self._model = EncoderClassifier.from_hparams(
            source=self.source,
            savedir=str(savedir),
            run_opts={"device": self.device},
        )
        return self

    def embed(self, wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Embed one mono waveform into a ``(EMBEDDING_DIM,)`` vector.

        Args:
            wav: 1-D float waveform, mono. Must be 16 kHz (ECAPA's rate);
                resampling is the caller's job (scitex-audio io/).
            sample_rate: Sample rate of ``wav``; must be 16000.

        Returns:
            L2-normalised float32 embedding, shape ``(EMBEDDING_DIM,)``.
        """
        if sample_rate != 16000:
            raise ValueError(
                f"ECAPA expects 16 kHz mono; got {sample_rate} Hz. "
                "Resample upstream in scitex_audio.voice.io."
            )
        self.load()
        import torch  # noqa: PLC0415

        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_batch(tensor).squeeze().cpu().numpy()
        emb = emb.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


# EOF
