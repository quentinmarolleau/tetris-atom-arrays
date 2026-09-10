"""Names for the target reshapes the benchmark sweeps over.

A reshape is a (rows, columns) pair added to the natural target side.
`variant_key` turns one into the string a cached result is filed under;
`variant_label` turns one into the phrase a plot legend or a paragraph
uses. Neither decides anything the benchmark measures, so this module
stays out of the fingerprint.
"""

__all__ = ["variant_key", "variant_label"]

_LABELS = {
    (0, 0): "No margin",
    (0, -1): "One column off",
    (-1, 0): "One row off",
    (-1, -1): "One row and one column off",
}


def variant_key(reshape: tuple[int, int]) -> str:
    """Result key for a reshape, e.g. (0, -1) -> `r0c-1`. Chosen to
    survive a round trip through JSON and to stay readable in a diff of
    the cache."""
    rows, columns = reshape
    return f"r{int(rows)}c{int(columns)}"


def variant_label(reshape: tuple[int, int]) -> str:
    """Human phrasing for a reshape, for legends and prose."""
    rows, columns = int(reshape[0]), int(reshape[1])
    named = _LABELS.get((rows, columns))
    if named is not None:
        return named
    return f"{rows:+d} rows, {columns:+d} columns"
