import pytest

from electricpy import compute


def test_largest_integer_validates_num_bits():
    """Validate largest integer validates num bits behavior."""
    with pytest.raises(ValueError):
        compute.largest_integer(0)
    with pytest.raises(ValueError):
        compute.largest_integer(-3)


def test_largest_integer_signed_unsigned():
    """Validate largest integer signed unsigned behavior."""
    assert compute.largest_integer(1, signed=True) == 0
    assert compute.largest_integer(8, signed=True) == 127
    assert compute.largest_integer(8, signed=False) == 255


def test_crc_sender_and_remainder():
    """Validate crc sender and remainder behavior."""
    data = "1101011011"
    key = "10011"
    codeword = compute.crcsender(data, key)
    remainder = compute.crcremainder(codeword, key)

    assert len(codeword) == len(data) + len(key) - 1
    assert set(codeword) <= {"0", "1"}
    assert len(remainder) == len(key) - 1
    assert set(remainder) == {"0"}


@pytest.mark.parametrize("character_string, expected_bits", [
    ("A", "01000001"),
    ("B", "01000010"),
    ("C", "01000011"),
    ("a", "01100001"),
    ("b", "01100010"),
    ("c", "01100011"),
    ("0", "00110000"),
    ("1", "00110001"),
    ("2", "00110010"),
])
def test_string_to_bits(character_string, expected_bits):
    """Validate string to bits behavior."""
    bits = compute.string_to_bits(character_string)
    assert bits == expected_bits
