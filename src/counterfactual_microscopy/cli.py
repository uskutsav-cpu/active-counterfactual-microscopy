from pathlib import Path

import yaml

from .actions import AcquisitionAction
from .hypotheses import DiscretePredictiveModel
from .policy import InformationGainPolicy
from .simulator import SimulatedMicroscope, update_biology_posterior
from .verdicts import verdict_from_posterior


def run_from_config(path: str | Path) -> None:
    cfg = yaml.safe_load(Path(path).read_text())
    spec = {}
    actions = []
    for name, acfg in cfg["actions"].items():
        spec[name] = {"biology": acfg["biology"], "nuisance": acfg["nuisance"]}
        actions.append(
            AcquisitionAction(
                name=name,
                dose_cost=float(acfg["dose_cost"]),
                time_cost=float(acfg["time_cost"]),
            )
        )

    model = DiscretePredictiveModel(spec)
    policy = InformationGainPolicy(
        dose_weight=float(cfg["dose_weight"]),
        time_weight=float(cfg["time_weight"]),
    )
    scope = SimulatedMicroscope(
        model,
        ground_truth=cfg["ground_truth"],
        seed=int(cfg["seed"]),
    )

    posterior = float(cfg["prior_biology"])
    remaining = list(actions)
    for step in range(1, int(cfg["max_steps"]) + 1):
        score = policy.select(remaining, model, posterior)
        obs = scope.acquire(score.action)
        posterior = update_biology_posterior(posterior, obs, model)
        verdict = verdict_from_posterior(
            posterior,
            support_threshold=float(cfg["support_threshold"]),
            falsify_threshold=float(cfg["falsify_threshold"]),
        )
        print(
            f"step={step} action={score.action.name} EIG={score.information_gain:.3f} "
            f"utility={score.utility:.3f} outcome={obs.outcome} "
            f"P(biology)={posterior:.3f} verdict={verdict.value}"
        )
        if verdict.value != "abstain":
            break
        remaining = [a for a in remaining if a.name != score.action.name]
        if not remaining:
            break


def main() -> None:
    run_from_config(Path("configs/simulated.yaml"))


if __name__ == "__main__":
    main()
