"""Encoding cost estimator: qubit count, depth, memory, and feasibility.

Usage::

    from qupepfold.model.cost_estimator import estimate_cost, compare_encodings

    spec = estimate_cost("turn", n_turns=10, n_residues=11, ansatz_depth=2)
    print(spec.feasibility)  # "simulator_ok"
    print(spec.n_qubits)     # 20

    table = compare_encodings(n_turns=10, n_residues=11)
    print(table)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .encodings import get_encoding
from .encodings.base import EncodingSpec


def estimate_cost(
    encoding_name: str,
    n_turns: int,
    n_residues: int,
    ansatz_depth: int = 2,
    **encoding_kwargs,
) -> EncodingSpec:
    """Estimate quantum resource cost for a given encoding and fragment size.

    Args:
        encoding_name: One of "turn", "hp", "constrained_local".
        n_turns: Number of conformational turns in the fragment.
        n_residues: Number of residues (= n_turns + 1).
        ansatz_depth: Ansatz circuit depth (layers).
        **encoding_kwargs: Extra args forwarded to encoding constructor.

    Returns:
        EncodingSpec with qubit count, depth, memory, and feasibility verdict.
    """
    enc = get_encoding(encoding_name, **encoding_kwargs)
    return enc.estimate_cost(n_turns, n_residues, ansatz_depth)


def compare_encodings(
    n_turns: int,
    n_residues: int,
    ansatz_depth: int = 2,
) -> str:
    """Compare all available encodings for the same fragment size.

    Args:
        n_turns: Number of turns.
        n_residues: Number of residues.
        ansatz_depth: Circuit depth.

    Returns:
        Human-readable comparison table.
    """
    encodings = ["turn", "hp", "constrained_local"]
    rows = []
    headers = ["Encoding", "Qubits", "Terms", "Depth", "CNOTs", "Mem(MB)", "Verdict"]
    rows.append(headers)

    for name in encodings:
        spec = estimate_cost(name, n_turns, n_residues, ansatz_depth)
        rows.append([
            name,
            str(spec.n_qubits),
            str(spec.n_terms),
            str(spec.circuit_depth_estimate),
            str(spec.cnot_estimate),
            f"{spec.memory_estimate_mb:.3f}",
            spec.feasibility,
        ])

    # Format table
    col_widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    lines = []
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    lines.append(sep)
    for i, row in enumerate(rows):
        line = "| " + " | ".join(row[j].ljust(col_widths[j]) for j in range(len(row))) + " |"
        lines.append(line)
        if i == 0:
            lines.append(sep)
    lines.append(sep)
    header = f"Encoding comparison: {n_residues} residues, {n_turns} turns, depth={ansatz_depth}"
    return header + "\n" + "\n".join(lines)


def hardware_feasible(encoding_name: str, n_turns: int) -> bool:
    """Quick check if an encoding is hardware-feasible for the given fragment."""
    spec = estimate_cost(encoding_name, n_turns, n_turns + 1)
    return spec.feasibility in ("hardware_feasible", "simulator_ok")


def recommend_encoding(n_turns: int) -> str:
    """Recommend the best encoding for a given fragment size.

    Rules:
    - n_turns <= 6: "constrained_local" (most physical)
    - n_turns <= 12: "turn" (efficient, hardware-feasible)
    - n_turns <= 15: "hp" (fewer qubits, coarser model)
    - n_turns > 15: fragment should be split further

    Args:
        n_turns: Number of turns in the fragment.

    Returns:
        Recommended encoding name.
    """
    if n_turns <= 6:
        return "constrained_local"
    elif n_turns <= 12:
        return "turn"
    elif n_turns <= 15:
        return "hp"
    else:
        return "hp"  # best we can do; caller should warn about fragment size
