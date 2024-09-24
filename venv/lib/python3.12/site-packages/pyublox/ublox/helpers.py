


def print_bytes(byte_collection, delimiter=" "):
    """
    Output a human readable string (with spaces) of a byte list

    >>> print_bytes(b'\x02\x65)
    '02 65'

    >>> print_bytes(b'\xff\xaa', delimiter"-")
    'ff-aa'
    """

    byte_list = [bytes([b]).hex() for b in byte_collection]
    bytes_str = delimiter.join(byte_list)

    return bytes_str
