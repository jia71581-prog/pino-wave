import pytest

from scripts.select_saved_time_physical_microbatch import select_largest_safe_microbatch


GIB = 1024**3


def _report(size, *, peak_gib=None, oom=False):
    return {
        "physical_microbatch_records": size,
        "oom": oom,
        "return_code": 1 if oom else 0,
        "peak_cuda_bytes": None if peak_gib is None else int(peak_gib * GIB),
    }


def test_selects_largest_safe_fresh_process_result():
    reports = [
        _report(12, oom=True),
        _report(6, peak_gib=22.7),
        _report(4, peak_gib=19.0),
        _report(3, peak_gib=16.0),
    ]

    assert select_largest_safe_microbatch(reports, maximum_gib=23.0) == 6


def test_selector_rejects_missing_duplicate_or_invalid_probe_evidence():
    complete = [
        _report(12, oom=True),
        _report(6, oom=True),
        _report(4, peak_gib=19.0),
        _report(3, peak_gib=16.0),
    ]

    with pytest.raises(ValueError, match="exactly one"):
        select_largest_safe_microbatch(complete[:-1], maximum_gib=23.0)
    with pytest.raises(ValueError, match="exactly one"):
        select_largest_safe_microbatch(complete + [_report(3, peak_gib=13.0)], maximum_gib=23.0)
    broken = list(complete)
    broken[2] = _report(4, peak_gib=19.0)
    broken[2]["return_code"] = 7
    with pytest.raises(ValueError, match="probe evidence"):
        select_largest_safe_microbatch(broken, maximum_gib=23.0)


def test_selector_fails_when_even_microbatch_three_is_unsafe():
    reports = [_report(size, oom=True) for size in (12, 6, 4, 3)]

    with pytest.raises(RuntimeError, match="no safe"):
        select_largest_safe_microbatch(reports, maximum_gib=23.0)
