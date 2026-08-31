"""Plotting utilities for QuPepFold results.

Includes:
- Bitstring histogram
- CVaR scatter across iterations
- Energy breakdown bar chart
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "qupepfold-matplotlib")
    import matplotlib

    if not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError("matplotlib is required. Install with: pip install matplotlib") from e

from ..types import FoldResult


def plot_bitstring_histogram(
    prob_dict: Dict[str, float],
    output_path: str,
    min_prob: float = 0.02,
    title: Optional[str] = None,
) -> None:
    """Plot histogram of high-probability bitstrings.
    
    Args:
        prob_dict: {bitstring: probability} mapping.
        output_path: Path to save PNG.
        min_prob: Minimum probability threshold for display.
        title: Optional custom title.
    """
    # Filter by minimum probability
    filtered = {k: v for k, v in prob_dict.items() if v >= min_prob}
    
    plt.figure(figsize=(12, 5))
    
    if filtered:
        # Sort by probability descending
        items = sorted(filtered.items(), key=lambda x: -x[1])
        xs = [k[:15] + "..." if len(k) > 15 else k for k, _ in items]  # Truncate long bitstrings
        ys = [v for _, v in items]
        
        plt.bar(xs, ys, edgecolor='black', color='steelblue')
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.ylabel("Probability")
    plt.xlabel("Bitstring")
    plt.title(title or f"High-Probability Bitstrings (≥ {min_prob:.2%})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cvar_scatter(
    cvar_trace: List[float],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Plot CVaR energy across optimization iterations.
    
    Args:
        cvar_trace: List of CVaR values per iteration.
        output_path: Path to save PNG.
        title: Optional custom title.
    """
    plt.figure(figsize=(8, 5))
    
    iterations = range(1, len(cvar_trace) + 1)
    plt.scatter(iterations, cvar_trace, marker='o', c='darkorange', edgecolors='black', s=50)
    plt.plot(iterations, cvar_trace, 'k--', alpha=0.3, linewidth=1)
    
    plt.xlabel("Iteration")
    plt.ylabel("CVaR Energy")
    plt.title(title or "CVaR Energies Across Iterations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_energy_breakdown(
    components: Dict[str, float],
    output_path: str,
    bitstring: Optional[str] = None,
) -> None:
    """Plot energy breakdown by component.
    
    Args:
        components: Dict with keys like 'backbone', 'mj', 'distance', 'locality'.
        output_path: Path to save PNG.
        bitstring: Optional bitstring label for title.
    """
    # Compute total
    total = sum(components.values())
    
    # Prepare data
    labels = list(components.keys()) + ["Total"]
    values = list(components.values()) + [total]
    
    # Color coding: negative=green, positive=red
    colors = ['forestgreen' if v < 0 else 'indianred' for v in values[:-1]]
    colors.append('steelblue')  # Total
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)
    
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.ylabel("Energy")
    plt.title(f"Energy Breakdown" + (f" — {bitstring[:20]}..." if bitstring else ""))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_fragment_energies(
    fragment_energies: List[float],
    output_path: str,
) -> None:
    """Plot energy distribution across fragments.
    
    Args:
        fragment_energies: List of best energy per fragment.
        output_path: Path to save PNG.
    """
    plt.figure(figsize=(8, 4))
    
    fragments = range(1, len(fragment_energies) + 1)
    plt.bar(fragments, fragment_energies, color='teal', edgecolor='black')
    
    plt.xlabel("Fragment Index")
    plt.ylabel("Best Energy")
    plt.title("Fragment Energies")
    plt.xticks(fragments)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(
    result: FoldResult,
    output_dir: str,
    prob_dict: Optional[Dict[str, float]] = None,
    cvar_trace: Optional[List[float]] = None,
    energy_components: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Generate all standard plots for a folding result.
    
    Args:
        result: FoldResult object.
        output_dir: Directory to save plots.
        prob_dict: Optional probability distribution for histogram.
        cvar_trace: Optional CVaR trace for scatter plot.
        energy_components: Optional breakdown for bar chart.
        
    Returns:
        List of generated plot paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    
    # Fragment energies
    if result.fragment_candidates:
        frag_energies = [
            min(c.energy for c in cands) if cands else 0.0
            for cands in result.fragment_candidates
        ]
        frag_path = os.path.join(output_dir, "fragment_energies.png")
        plot_fragment_energies(frag_energies, frag_path)
        generated.append(frag_path)
    
    # Bitstring histogram
    if prob_dict:
        hist_path = os.path.join(output_dir, "bitstring_histogram.png")
        plot_bitstring_histogram(prob_dict, hist_path)
        generated.append(hist_path)
    
    # CVaR scatter
    if cvar_trace:
        scatter_path = os.path.join(output_dir, "cvar_scatter.png")
        plot_cvar_scatter(cvar_trace, scatter_path)
        generated.append(scatter_path)
    
    # Energy breakdown
    if energy_components:
        breakdown_path = os.path.join(output_dir, "energy_breakdown.png")
        plot_energy_breakdown(energy_components, breakdown_path, result.stitched_bits)
        generated.append(breakdown_path)
    
    return generated
