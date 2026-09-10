from counterfactual_microscopy.verdicts import Verdict, verdict_from_posterior


def test_verdict_thresholds() -> None:
    assert verdict_from_posterior(0.90) is Verdict.SUPPORTED
    assert verdict_from_posterior(0.10) is Verdict.FALSIFIED
    assert verdict_from_posterior(0.50) is Verdict.ABSTAIN
