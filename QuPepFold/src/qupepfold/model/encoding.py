"""Turn encoding and decoding for lattice models.

Convention: 2 bits per turn, big-endian encoding.
Turn codes: 0, 1, 2, 3 map to different lattice directions.

In 2D square lattice:
  0 = straight (+x direction relative to current heading)
  1 = left turn (+y / counterclockwise 90°)
  2 = right turn (-y / clockwise 90°)
  3 = reverse (-x / 180°, penalized in energy)
"""

from typing import List


# Valid turn codes
VALID_TURN_CODES = {0, 1, 2, 3}


def encode_turns(turn_codes: List[int]) -> str:
    """Encode a list of turn codes to a bitstring.
    
    Args:
        turn_codes: List of integers in {0, 1, 2, 3}.
        
    Returns:
        Bitstring of length 2 * len(turn_codes), big-endian.
        
    Raises:
        ValueError: If any turn code is invalid.
        
    Example:
        >>> encode_turns([0, 1, 2, 3])
        '00011011'
    """
    bits = []
    for i, code in enumerate(turn_codes):
        if code not in VALID_TURN_CODES:
            raise ValueError(f"Invalid turn code {code} at position {i}. Must be in {VALID_TURN_CODES}")
        # Big-endian: MSB first for each 2-bit code
        bits.append(format(code, '02b'))
    return ''.join(bits)


def decode_turns(bitstring: str) -> List[int]:
    """Decode a bitstring to a list of turn codes.
    
    Args:
        bitstring: String of '0' and '1' characters with even length.
        
    Returns:
        List of turn codes (integers in {0, 1, 2, 3}).
        
    Raises:
        ValueError: If bitstring has odd length or invalid characters.
        
    Example:
        >>> decode_turns('00011011')
        [0, 1, 2, 3]
    """
    if len(bitstring) % 2 != 0:
        raise ValueError(f"Bitstring length {len(bitstring)} must be even")
    
    # Validate characters
    if not all(c in '01' for c in bitstring):
        raise ValueError("Bitstring must contain only '0' and '1' characters")
    
    turns = []
    for i in range(0, len(bitstring), 2):
        pair = bitstring[i:i+2]
        code = int(pair, 2)
        if code not in VALID_TURN_CODES:
            # This shouldn't happen with 2 bits, but defensive check
            raise ValueError(f"Invalid turn code {code} from bits '{pair}'")
        turns.append(code)
    
    return turns


def bitstring_to_index(bitstring: str) -> int:
    """Convert bitstring to integer index (big-endian).
    
    Args:
        bitstring: Binary string.
        
    Returns:
        Integer value of the bitstring.
    """
    return int(bitstring, 2)


def index_to_bitstring(index: int, n_bits: int) -> str:
    """Convert integer index to bitstring (big-endian).
    
    Args:
        index: Integer value.
        n_bits: Desired bitstring length.
        
    Returns:
        Padded binary string.
    """
    return format(index, f'0{n_bits}b')
