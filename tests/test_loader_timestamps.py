import pandas as pd
import pytest

from src.data.loader import _parse_concatenated_timestamps, _split_month_day_candidates


def test_split_month_day_unambiguous_two_digits():
    assert _split_month_day_candidates("11") == [(1, 1)]


def test_split_month_day_unambiguous_four_digits():
    assert _split_month_day_candidates("1130") == [(11, 30)]


def test_split_month_day_ambiguous_three_digits_returns_both_candidates():
    candidates = _split_month_day_candidates("127")
    assert set(candidates) == {(1, 27), (12, 7)}


def test_split_month_day_invalid_length_raises():
    with pytest.raises(ValueError):
        _split_month_day_candidates("1")


def test_parse_concatenated_timestamps_resolves_ambiguity_using_previous_row():
    # "1272006" -> Jan 27 2006 or Dec 7 2006. The previous row anchors at
    # Dec 6 2006 23:00, so the correct split (hourly continuity) is Dec 7.
    raw = pd.Series(["12062006 23:00", "1272006 0:00"])
    parsed, prev = _parse_concatenated_timestamps(raw, prev=None)

    assert parsed.iloc[0] == pd.Timestamp("2006-12-06 23:00")
    assert parsed.iloc[1] == pd.Timestamp("2006-12-07 00:00")
    assert prev == parsed.iloc[1]


def test_parse_concatenated_timestamps_first_row_ambiguous_picks_earliest():
    # No previous timestamp to disambiguate against -> falls back to the
    # earliest valid calendar date among the candidates.
    raw = pd.Series(["1272006 0:00"])
    parsed, _ = _parse_concatenated_timestamps(raw, prev=None)

    assert parsed.iloc[0] == pd.Timestamp("2006-01-27 00:00")


def test_parse_concatenated_timestamps_invalid_format_raises():
    raw = pd.Series(["not-a-timestamp"])
    with pytest.raises(ValueError):
        _parse_concatenated_timestamps(raw, prev=None)
