################################################################################
"""
Functions to Support Computer-Science Related Formulas.

>>> from electricpy import compute as cmp

Filled with calculators, evaluators, and plotting functions related to
electrical phasors, this package will provide a wide array of capabilities to
any electrical engineer.

Built to support operations similar to Numpy and Scipy, this package is designed
to aid in scientific calculations.
"""
################################################################################


# Define Simple Maximum Integer Evaluator
def largest_integer(numBits, signed=True):
    """
    Evaluate the largest integer that may be represented by an n-bit variable.

    This function will evaluate the largest integer an n-bit variable may hold
    in computers. It can evaluate signed or unsigned integers. The formulas for
    which it determines these values are as follows:

    **Signed Integers**

    .. math:: 2^{n-1} - 1

    **Unsigned Integers**

    .. math:: 2^{n} - 1

    Parameters
    ----------
    numBits:    int
                The number of bits that should be used to interpret the overall
                maximum size.
    signed:     bool, optional
                Control to specify whether the value should be evaluated for
                signed, or unsigned, integers. Defaults to True.
    
    Returns
    -------
    int:        The maximum value that can be stored in an integer of numBits.

    Examples
    --------
    >>> import electricpy.compute as cmp
    >>> cmp.largest_integer(8, signed=False)
    255
    >>> cmp.largest_integer(32, signed=False)
    4294967295
    >>> cmp.largest_integer(32, signed=True)
    2147483647
    """
    # Use Signed or Unsigned Formula
    if signed:
        return int(2 ** (numBits - 1) - 1)
    else:
        return int(2 ** (numBits) - 1)


# Define CRC Generator (Sender Side)
def crcsender(data, key):
    """
    CRC Sender Function.

    Function to generate a CRC-embedded message ready for transmission.

    Contributing Author Credit:
    Shaurya Uppal
    Available from: geeksforgeeks.org

    Parameters
    ----------
    data:       string of bits
                The bit-string to be encoded.
    key:        string of bits
                Bit-string representing key.

    Returns
    -------
    codeword:   string of bits
                Bit-string representation of
                encoded message.
    """
    # Define Sub-Functions
    def xor(a, b):
        # initialize result
        result = []

        # Traverse all bits, if bits are
        # same, then XOR is 0, else 1
        for i in range(1, len(b)):
            if a[i] == b[i]:
                result.append('0')
            else:
                result.append('1')

        return ''.join(result)

    # Performs Modulo-2 division
    def mod2div(divident, divisor):
        # Number of bits to be XORed at a time.
        pick = len(divisor)

        # Slicing the divident to appropriate
        # length for particular step
        tmp = divident[0: pick]

        while pick < len(divident):

            if tmp[0] == '1':

                # replace the divident by the result
                # of XOR and pull 1 bit down
                tmp = xor(divisor, tmp) + divident[pick]

            else:  # If leftmost bit is '0'

                # If the leftmost bit of the dividend (or the
                # part used in each step) is 0, the step cannot
                # use the regular divisor; we need to use an
                # all-0s divisor.
                tmp = xor('0' * pick, tmp) + divident[pick]

                # increment pick to move further
            pick += 1

        # For the last n bits, we have to carry it out
        # normally as increased value of pick will cause
        # Index Out of Bounds.
        if tmp[0] == '1':
            tmp = xor(divisor, tmp)
        else:
            tmp = xor('0' * pick, tmp)

        checkword = tmp
        return checkword

    # Condition data
    data = str(data)
    # Condition Key
    key = str(key)
    l_key = len(key)

    # Appends n-1 zeroes at end of data
    appended_data = data + '0' * (l_key - 1)
    remainder = mod2div(appended_data, key)

    # Append remainder in the original data
    codeword = data + remainder
    return codeword


# Define CRC Generator (Sender Side)
def crcremainder(data, key):
    """
    CRC Remainder Function.

    Function to calculate the CRC remainder of a CRC message.

    Contributing Author Credit:
    Shaurya Uppal
    Available from: geeksforgeeks.org

    Parameters
    ----------
    data:       string of bits
                The bit-string to be decoded.
    key:        string of bits
                Bit-string representing key.

    Returns
    -------
    remainder: string of bits
                Bit-string representation of
                encoded message.
    """
    # Define Sub-Functions
    def xor(a, b):
        # initialize result
        result = []

        # Traverse all bits, if bits are
        # same, then XOR is 0, else 1
        for i in range(1, len(b)):
            if a[i] == b[i]:
                result.append('0')
            else:
                result.append('1')

        return ''.join(result)

    # Performs Modulo-2 division
    def mod2div(divident, divisor):
        # Number of bits to be XORed at a time.
        pick = len(divisor)

        # Slicing the divident to appropriate
        # length for particular step
        tmp = divident[0: pick]

        while pick < len(divident):

            if tmp[0] == '1':

                # replace the divident by the result
                # of XOR and pull 1 bit down
                tmp = xor(divisor, tmp) + divident[pick]

            else:  # If leftmost bit is '0'

                # If the leftmost bit of the dividend (or the
                # part used in each step) is 0, the step cannot
                # use the regular divisor; we need to use an
                # all-0s divisor.
                tmp = xor('0' * pick, tmp) + divident[pick]

                # increment pick to move further
            pick += 1

        # For the last n bits, we have to carry it out
        # normally as increased value of pick will cause
        # Index Out of Bounds.
        if tmp[0] == '1':
            tmp = xor(divisor, tmp)
        else:
            tmp = xor('0' * pick, tmp)

        checkword = tmp
        return checkword

    # Condition data
    data = str(data)
    # Condition Key
    key = str(key)
    l_key = len(key)

    # Appends n-1 zeroes at end of data
    appended_data = data + '0' * (l_key - 1)
    remainder = mod2div(appended_data, key)

    return remainder


# Define String to Bits Function
def string_to_bits(str):
    # noqa: D401   "String" is an intended leading word.
    """
    String to Bits Converter.

    Converts a Pythonic string to the string's binary representation.

    Parameters
    ----------
    str:        string
                The string to be converted.

    Returns
    -------
    data:       string
                The binary representation of the
                input string.
    """
    data = (''.join(format(ord(x), 'b') for x in str))
    return data

# END
