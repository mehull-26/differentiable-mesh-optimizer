# Contributing to 3D Reconstruction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
4. **Make your changes**
5. **Test your changes** (`pytest`)
6. **Commit your changes** (`git commit -m 'Add amazing feature'`)
7. **Push to GitHub** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/3d-reconstruction.git
cd 3d-reconstruction

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable names

### Formatting

Format code with Black:

```bash
black src/ tests/
```

Check style with flake8:

```bash
flake8 src/ tests/
```

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Include type hints
- Update README.md if adding features

Example docstring:

```python
def optimize_mesh(vertices: torch.Tensor,
                 faces: torch.Tensor,
                 config: dict) -> torch.Tensor:
    """
    Optimize mesh to match target images.
    
    Args:
        vertices: Initial vertex positions (V, 3)
        faces: Triangle face indices (F, 3)
        config: Configuration dictionary
        
    Returns:
        Optimized vertex positions (V, 3)
        
    Example:
        >>> vertices, faces = create_icosphere(1)
        >>> optimized = optimize_mesh(vertices, faces, config)
    """
    pass
```

## Testing

### Writing Tests

- Write tests for all new features
- Place tests in `tests/` directory
- Use descriptive test names: `test_feature_does_something`
- Test both success and failure cases

Example test:

```python
def test_create_icosphere():
    """Test icosphere creation."""
    vertices, faces = create_icosphere(subdivisions=1)
    
    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert vertices.dtype == torch.float32
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_mesh.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

## Pull Request Process

### Before Submitting

1. **Update documentation** - README, docstrings, etc.
2. **Add tests** - Cover new functionality
3. **Run tests** - Ensure all tests pass
4. **Format code** - Run black and flake8
5. **Update CHANGELOG** - Describe your changes

### PR Description

Include in your PR description:

- **What** - What does this PR do?
- **Why** - Why is this change needed?
- **How** - How does it work?
- **Testing** - How did you test it?
- **Screenshots** - If applicable

Example:

```markdown
## What
Adds RGB texture optimization support.

## Why
Users requested ability to reconstruct textured meshes, not just geometry.

## How
- Added texture parameter to mesh class
- Implemented texture loss function
- Updated renderer to handle textures

## Testing
- Added unit tests for texture functions
- Tested on synthetic textured cube
- All existing tests still pass

## Screenshots
[Before/after comparison images]
```

### Code Review

- Be responsive to feedback
- Make requested changes promptly
- Keep discussions professional and constructive

## Types of Contributions

### Bug Reports

Submit bug reports via [GitHub Issues](https://github.com/mehull-26/differentiable-mesh-optimizer/issues).

Include:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Error messages and stack traces

### Feature Requests

Submit feature requests via [GitHub Issues](https://github.com/mehull-26/differentiable-mesh-optimizer/issues).

Include:
- Clear use case
- Expected behavior
- Benefits to users
- Willingness to implement

### Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Translate documentation

### Code Contributions

Areas where contributions are especially welcome:

- **Performance optimization** - Speed up rendering or optimization
- **New features** - Texture optimization, normal maps, etc.
- **Better visualizations** - Interactive viewers, better plots
- **More examples** - Real-world use cases
- **Better error messages** - More helpful error handling

## Community Guidelines

### Be Respectful

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Communication

- Use clear, concise language
- Provide context and examples
- Ask questions if something is unclear
- Thank contributors for their work

## Recognition

Contributors will be:

- Listed in README.md acknowledgements
- Credited in CHANGELOG.md
- Mentioned in release notes

Significant contributors may be invited as maintainers.

## Questions?

  - **General questions**: [GitHub Discussions](https://github.com/mehull-26/differentiable-mesh-optimizer/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/mehull-26/differentiable-mesh-optimizer/issues)
- **Email**: mehulyadav2605@gmail.com

Thank you for contributing! 🎉
