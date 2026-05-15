"""
src/main.py — Entry point del progetto Applied Statistics.

Esegue la pipeline di preprocessing e carica il dataset ML-ready.
Qui verranno aggiunte le fasi successive (EDA, modelling, ecc.).

Utilizzo:
    python src/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Assicura che la root del progetto sia nel sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dataset.preprocess import main as run_preprocessing  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("  Applied Statistics Project — Pipeline")
    print("=" * 60)

    # --------------------------------------------------------------------------
    # Fase A: Data Engineering & Wrangling
    # --------------------------------------------------------------------------
    print("\n[Fase A] Preprocessing dataset Qiskit Calibration Drift...")
    df = run_preprocessing()
    print(f"\n[Fase A] Completata. Dataset shape: {df.shape}")

    # --------------------------------------------------------------------------
    # Fase B: EDA — TODO
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Fase C: Modelling — TODO
    # --------------------------------------------------------------------------


if __name__ == "__main__":
    main()
