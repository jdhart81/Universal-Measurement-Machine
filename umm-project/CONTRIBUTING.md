# Contributing to UMM

We welcome contributions to the Universal Measurement Machine project!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/umm-project.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests: `pytest tests/`
6. Commit: `git commit -m "Description of changes"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
pip install -e ".[dev]"
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all public functions
- Keep line length ≤ 100 characters

## Testing

All new features should include tests:

```bash
pytest tests/ -v
```

## Documentation

Update documentation when adding features:

```bash
cd docs/
make html
```

## Questions?

Open an issue or contact justin@viridis.llc
