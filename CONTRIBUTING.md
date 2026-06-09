# Contributing to Lung Cancer Diagnostic System

Thank you for your interest in contributing to the Lung Cancer Diagnostic System! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, but recommended)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/lung-cancer-diagnostic-system.git
   cd lung-cancer-diagnostic-system
   ```

3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/original-repo/lung-cancer-diagnostic-system.git
   ```

## Development Setup

### Using Docker (Recommended)

1. **Start development environment:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

2. **Run the application:**
   ```bash
   docker-compose exec app python run.py
   ```

### Manual Setup

1. **Create virtual environment:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Validate setup:**
   ```bash
   python scripts/validate_setup.py
   ```

5. **Run the application:**
   ```bash
   python run.py
   ```

## Development Workflow

### Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical fixes for production

### Creating a Feature Branch

```bash
# Sync with upstream
git checkout develop
git pull upstream develop

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write tests first** (TDD approach)
2. **Follow code style guidelines**
3. **Keep commits atomic and well-documented**
4. **Test your changes thoroughly**

### Commit Guidelines

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat(api): add CT scan analysis endpoint
fix(validation): correct image size validation
docs(api): update endpoint documentation
```

## Code Style

This project uses several tools to maintain code quality:

### Python Code Style

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Running Code Quality Checks

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run quality checks:

```bash
pip install pre-commit
pre-commit install
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── fixtures/      # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run tests in verbose mode
pytest -v

# Run tests matching pattern
pytest -k "test_ct_scan"
```

### Writing Tests

- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Use fixtures for test data
- Mock external dependencies
- Test both success and failure cases

Example:
```python
def test_ct_scan_analysis_success(client, sample_image):
    # Arrange
    data = {'file': sample_image}

    # Act
    response = client.post('/api/v1/ct-scan/analyze', data=data)

    # Assert
    assert response.status_code == 200
    assert response.json['success'] is True
    assert 'prediction' in response.json
```

## Documentation

### Documentation Types

- **Code Documentation**: Docstrings for all public functions/classes
- **API Documentation**: OpenAPI/Swagger specs
- **User Guides**: Step-by-step instructions
- **Architecture Docs**: System design and decisions

### Documentation Standards

- Use Google-style docstrings
- Keep documentation up-to-date
- Include code examples where helpful
- Document API changes in release notes

### Building Documentation

```bash
# Generate API documentation
python scripts/generate_api_docs.py

# Build user documentation
mkdocs build
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up-to-date:**
   ```bash
   git checkout develop
   git pull upstream develop
   git checkout your-feature-branch
   git rebase develop
   ```

2. **Run full test suite:**
   ```bash
   pytest --cov=src --cov-report=xml
   ```

3. **Update documentation if needed**

4. **Create pull request:**
   - Use descriptive title
   - Provide detailed description
   - Reference related issues
   - Add screenshots for UI changes

5. **Code Review:**
   - Address review comments
   - Ensure CI checks pass
   - Get approval from maintainers

### PR Template

```
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No breaking changes
```

## Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 91]

**Additional context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem? Please describe.**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
A clear description of any alternative solutions.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## Recognition

Contributors will be recognized in:
- Repository contributors list
- Release notes
- Project documentation

Thank you for contributing to the Lung Cancer Diagnostic System! 🎉</content>
<parameter name="filePath">c:\Users\Anand Singh\OneDrive\Desktop\Major\CONTRIBUTING.md