from pathlib import Path

from counterfactual_microscopy.cli import run_from_config


if __name__ == "__main__":
    run_from_config(Path(__file__).resolve().parents[1] / "configs" / "simulated.yaml")
