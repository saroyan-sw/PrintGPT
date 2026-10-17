"""Paper-loading profile and printer readiness checks (pure logic, no hardware)."""

from datetime import UTC, datetime

import pytest

from printgpt_mvp.print_readiness import (
    PaperLoadingProfile,
    check_print_readiness,
    load_paper_profile,
    mark_verified,
    save_paper_profile,
    with_loading,
)


READY = {"queue": "Canon Generic Plus UFR II", "found": True, "allowlisted": True, "network_reachable": True,
         "ready": True, "states": ["idle"], "queued_jobs": 0}


def test_missing_profile_is_an_unverified_default(tmp_path) -> None:
    profile = load_paper_profile(tmp_path / "missing.json")
    assert profile == PaperLoadingProfile()
    assert not profile.verified
    assert any("bypass" in step for step in profile.instructions())


def test_profile_round_trip_and_verification(tmp_path) -> None:
    path = tmp_path / "paper.json"
    verified = mark_verified(PaperLoadingProfile(face="up", leading_edge="bottom"), datetime(2026, 10, 17, tzinfo=UTC))
    save_paper_profile(verified, path)
    loaded = load_paper_profile(path)
    assert loaded == verified and loaded.verified
    assert "drawing side up" in loaded.confirmation() and "bottom edge first" in loaded.confirmation()


def test_changing_the_loading_direction_clears_verification() -> None:
    verified = mark_verified(PaperLoadingProfile(), datetime(2026, 10, 17, tzinfo=UTC))
    assert with_loading(verified, tray=verified.tray, face="down", leading_edge="top") is verified
    changed = with_loading(verified, tray=verified.tray, face="up", leading_edge="top")
    assert not changed.verified and changed.verified_at is None


def test_invalid_profiles_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        PaperLoadingProfile(face="sideways")
    path = tmp_path / "other.json"
    path.write_text('{"schema": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_paper_profile(path)


def test_ready_printer_passes_every_check() -> None:
    readiness = check_print_readiness(READY, physical_print_enabled=True)
    assert readiness.ready and not readiness.failed


@pytest.mark.parametrize(
    ("change", "failed_check"),
    [
        ({"network_reachable": False, "ready": False}, "Printer reachable on the network"),
        ({"states": ["paper_out"], "ready": False}, "Paper path clear"),
        ({"states": ["door_open"], "ready": False}, "Paper path clear"),
        ({"queued_jobs": 2}, "No other jobs waiting"),
        ({"found": False, "ready": False}, "Printer queue found"),
    ],
)
def test_each_problem_blocks_printing(change, failed_check) -> None:
    readiness = check_print_readiness({**READY, **change}, physical_print_enabled=True)
    assert not readiness.ready
    assert failed_check in {check.name for check in readiness.failed}


def test_manual_feed_waiting_for_the_bypass_sheet_is_not_a_paper_problem() -> None:
    readiness = check_print_readiness({**READY, "states": ["manual_feed"]}, physical_print_enabled=True)
    assert readiness.ready


def test_disabled_physical_printing_is_reported() -> None:
    readiness = check_print_readiness(READY, physical_print_enabled=False)
    assert [check.name for check in readiness.failed] == ["Physical printing enabled"]
    assert "PRINTGPT_HW_ALLOW_PRINT=1" in readiness.failed[0].detail
