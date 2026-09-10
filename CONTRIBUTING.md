# Contributing

This is a research repository. Contributions should preserve scientific traceability.

## Before opening a PR

1. Add or update tests for changes to inference, action selection, or cost accounting.
2. Document any change that alters a scientific definition, metric, threshold, or benchmark.
3. Keep hardware-specific behavior behind `hardware/` adapters.
4. Do not commit raw human/animal imaging data, credentials, vendor SDK binaries, or protected metadata.
5. For numerical experiments, include the config and random seed needed to reproduce the result.

## Development

```bash
pip install -e '.[dev]'
ruff check .
pytest -q
```

## Scientific changes

PRs that change the formal definition of a hypothesis, acquisition utility, verdict, or primary endpoint should include a short **Scientific impact** section explaining what conclusions may change.
