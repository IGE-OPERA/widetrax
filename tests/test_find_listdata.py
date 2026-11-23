"""Filename parsing and local list-search (no S3 needed)."""
import os

from widetrax.DataPreprocessing import _parse_swot_filename, find_listdata_local

PROTOCOL_NAME = "GLO12V4_GRID_L3_LR_SSH_011_532_20240215000000_20240225235929_v1.0.2.nc"
GEOMAR_NAME = "SWOT_GRID_L3_LR_SSH_011_532_20240213000000_20240222235959_eINALT100.nc"
L3_OBS_NAME = "SWOT_L3_LR_SSH_Expert_532_011_20240215T012345_20240217T012345_v1.0.2.nc"


def test_parse_protocol_name():
    # _SSH_<cycle>_<pass>_<start>_<end>: pass is the 2nd token, month comes
    # from the end date.
    assert _parse_swot_filename(PROTOCOL_NAME) == (532, "20240225")


def test_parse_geomar_name():
    assert _parse_swot_filename(GEOMAR_NAME) == (532, "20240222")


def test_parse_l3_obs_name_via_token_fallback():
    # The regex does not cover this layout (non-numeric token after SSH), the
    # token fallback reproduces the historic parts-based result exactly:
    # pass = parts[idx+2], date = parts[idx+4][:8] = the START date token.
    assert _parse_swot_filename(L3_OBS_NAME) == (532, "20240215")


def test_parse_custom_index_name():
    name = "A_B_ABC_011_532_20240215000000_20240225235929.nc"
    assert _parse_swot_filename(name, index_name="ABC") == (532, "20240225")


def test_parse_unknown_name_returns_none():
    # Historic behaviour: parts.index("SSH") -> ValueError crash mid-scan.
    assert _parse_swot_filename("not_a_swot_file.nc") is None


def test_find_listdata_local_end_to_end(tmp_path):
    cycle_dir = tmp_path / "cycle_011"
    cycle_dir.mkdir()
    files = {
        "inside": PROTOCOL_NAME,
        "wrong_pass": "GLO12V4_GRID_L3_LR_SSH_011_999_20240215000000_20240225235929_v1.0.2.nc",
        "wrong_month": "GLO12V4_GRID_L3_LR_SSH_011_532_20240801000000_20240815235929_v1.0.2.nc",
        "unparsable": "no_token_here.nc",
        "junk": "readme.txt",
    }
    for name in files.values():
        (cycle_dir / name).write_text("x")

    result = find_listdata_local(
        ["011"], str(tmp_path), [532], "01012024", "31032024"
    )

    # Unparsable names are skipped with a warning instead of crashing; the
    # .txt is never globbed; wrong pass/month are filtered out.
    assert [os.path.basename(f) for f in result] == [PROTOCOL_NAME]

    # empty when no pass matches
    result_none = find_listdata_local(
        ["011"], str(tmp_path), [1], "01012024", "31032024"
    )
    assert result_none == []
