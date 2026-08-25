import pytest

from fmharness import check2

@pytest.fixture(autouse=True)
def _small_random_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the noise-control draw count for the whole suite.

    Every draw in score_check2 is a full per-drug cross-validated fit. The production value of
    20 exists so p_random -- a rank statistic with floor 1/(draws+1) -- can resolve below 0.05.
    Tests only need to prove the columns appear and the arithmetic runs, and leaving the
    production value in place took the suite from about two minutes to over ten.
    """
    monkeypatch.setattr(check2, "RANDOM_DRAWS", 2)
