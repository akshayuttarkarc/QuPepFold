"""Encoding schemes for quantum peptide folding.

An EncodingScheme converts a fragment's conformational space into a set of
qubit states, providing:
  - encode()   : build per-state energy array
  - decode()   : convert bitstring → turn codes
  - qubit_estimate() : how many qubits are needed
  - term_count()     : number of Hamiltonian terms
  - expected_circuit_depth() : rough circuit depth
  - penalty_summary()        : active penalty weights

Available schemes:
  "turn"              → TurnEncoding    (2 bits/turn, 4 HP-lattice directions)
  "hp"                → HPEncoding      (binary H/P labelling, 1–2 bits/turn)
  "constrained_local" → ConstrainedLocalEncoding (turn + overlap + SS restraints)

Usage::

    from qupepfold.model.encodings import get_encoding, TurnEncoding
    enc = get_encoding("turn")
    n_qubits = enc.qubit_estimate(fragment.n_turns)
"""

from .base import EncodingScheme, EncodingSpec
from .turn_encoding import TurnEncoding
from .hp_encoding import HPEncoding
from .constrained_local import ConstrainedLocalEncoding

_REGISTRY = {
    "turn": TurnEncoding,
    "hp": HPEncoding,
    "constrained_local": ConstrainedLocalEncoding,
}


def get_encoding(name: str, **kwargs) -> EncodingScheme:
    """Return an EncodingScheme instance by name.

    Args:
        name: One of "turn", "hp", "constrained_local".
        **kwargs: Forwarded to the scheme constructor.

    Returns:
        EncodingScheme instance.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown encoding '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return cls(**kwargs)


__all__ = [
    "EncodingScheme",
    "EncodingSpec",
    "TurnEncoding",
    "HPEncoding",
    "ConstrainedLocalEncoding",
    "get_encoding",
]
