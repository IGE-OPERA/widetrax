"""get_matching_cycles: preamble-smart calendar parsing.

The historic implementation hard-dropped the first two calendar rows
(`iloc[2:]`), which silently discards cycles on clean (header-only) CSVs.
Both layouts must return the same matching list.
"""
from widetrax.DataPreprocessing import get_matching_cycles

ROWS = (
    "1,2023-12-20 00:00:00,2024-01-05 00:00:00\n"
    "2,2024-01-10 00:00:00,2024-01-25 00:00:00\n"
    "3,2024-02-05 00:00:00,2024-02-20 00:00:00\n"
    "4,2024-03-25 00:00:00,2024-04-10 00:00:00\n"
    "5,2024-04-15 00:00:00,2024-04-30 00:00:00\n"
)
HEADER = "cycle_number,start_time,end_time\n"


def _calendar(tmp_path, content):
    path = tmp_path / "cycles.csv"
    path.write_text(content)
    return str(path)


def test_calendar_with_junk_preamble(tmp_path):
    # The layout the historic implementation was written against: two junk
    # lines + header + data.
    content = "# downloaded from somewhere\n# second junk line\n" + HEADER + ROWS
    path = _calendar(tmp_path, content)
    assert get_matching_cycles(path, "01012024", "31032024") == ["1", "2", "3", "4"]


def test_clean_calendar(tmp_path):
    # Regression: the historic iloc[2:] dropped the first two DATA rows here.
    path = _calendar(tmp_path, HEADER + ROWS)
    assert get_matching_cycles(path, "01012024", "31032024") == ["1", "2", "3", "4"]


def test_returns_strings(tmp_path):
    path = _calendar(tmp_path, HEADER + ROWS)
    result = get_matching_cycles(path, "01012024", "31032024")
    assert all(isinstance(c, str) for c in result)


def test_empty_window(tmp_path):
    path = _calendar(tmp_path, HEADER + ROWS)
    assert get_matching_cycles(path, "01072025", "31072025") == []
