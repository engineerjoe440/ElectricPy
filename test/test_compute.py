import pytest

from electricpy import compute


def test_largest_integer_validates_num_bits():
    with pytest.raises(ValueError):
        compute.largest_integer(0)
    with pytest.raises(ValueError):
        compute.largest_integer(-3)


def test_largest_integer_signed_unsigned():
    assert compute.largest_integer(8, signed=True) == 127
    assert compute.largest_integer(8, signed=False) == 255


def test_crc_sender_and_remainder():
    data = "1101011011"
    key = "10011"
    codeword = compute.crcsender(data, key)
    remainder = compute.crcremainder(codeword, key)

    assert len(codeword) == len(data) + len(key) - 1
    assert set(codeword) <= {"0", "1"}
    assert len(remainder) == len(key) - 1
    assert set(remainder) == {"0"}


def test_string_to_bits():
    bits = compute.string_to_bits("A")
    assert bits == "01000001"
