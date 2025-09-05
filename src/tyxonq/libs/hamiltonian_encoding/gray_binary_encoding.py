from __future__ import annotations

def get_gray_codes(n: int):
    """Return n-bit Gray code in a list of bitstrings."""
    if n == 0:
        return [""]
    sub_gray_codes = get_gray_codes(n - 1)
    gray_codes0 = ["0" + code for code in sub_gray_codes]
    gray_codes1 = ["1" + code for code in reversed(sub_gray_codes)]
    return gray_codes0 + gray_codes1


def get_binary_codes(n: int):
    codes = [bin(i)[2:] for i in range(1 << n)]
    return ["0" * (n - len(code)) + code for code in codes]


__all__ = ["get_gray_codes", "get_binary_codes"]


def get_encoding(m: int, boson_encoding: str | None):
    if boson_encoding is None:
        assert m == 2
        encoding_order = "01"
    elif boson_encoding == "unary":
        encoding_order = ["0" * (m - 1 - i) + "1" + "0" * i for i in range(m)]
    elif boson_encoding == "binary":
        encoding_order = get_binary_codes((m - 1).bit_length())[:m]
    else:
        assert boson_encoding == "gray"
        encoding_order = get_gray_codes((m - 1).bit_length())[:m]
    return encoding_order



