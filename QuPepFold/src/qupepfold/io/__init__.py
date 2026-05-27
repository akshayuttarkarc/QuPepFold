"""I/O utilities for QuPepFold.

Exports:
- JSON report building and writing
- Provenance tracking
- CSV export utilities
- Plotting utilities
"""

from .report_json import write_report, build_report
from .provenance import get_provenance
from .csv_export import (
    write_bitstring_summary,
    write_fragment_candidates_csv,
    write_energy_breakdown_csv,
    write_output_summary,
    generate_all_csvs,
)
from .plots import (
    plot_bitstring_histogram,
    plot_cvar_scatter,
    plot_energy_breakdown,
    plot_fragment_energies,
    generate_all_plots,
)

__all__ = [
    # JSON
    "write_report",
    "build_report",
    "get_provenance",
    # CSV
    "write_bitstring_summary",
    "write_fragment_candidates_csv",
    "write_energy_breakdown_csv",
    "write_output_summary",
    "generate_all_csvs",
    # Plots
    "plot_bitstring_histogram",
    "plot_cvar_scatter",
    "plot_energy_breakdown",
    "plot_fragment_energies",
    "generate_all_plots",
]
