from enum import Enum


class Verdict(str, Enum):
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    ABSTAIN = "abstain"


def verdict_from_posterior(
    p_biology: float,
    *,
    support_threshold: float = 0.85,
    falsify_threshold: float = 0.15,
) -> Verdict:
    if not 0.0 <= p_biology <= 1.0:
        raise ValueError("p_biology must be within [0, 1]")
    if not 0.0 <= falsify_threshold < support_threshold <= 1.0:
        raise ValueError("require 0 <= falsify_threshold < support_threshold <= 1")

    if p_biology >= support_threshold:
        return Verdict.SUPPORTED
    if p_biology <= falsify_threshold:
        return Verdict.FALSIFIED
    return Verdict.ABSTAIN
