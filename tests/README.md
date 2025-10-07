# Test Suite for Chain-of-Metric-Predictors

This directory contains the test suite for validating log file formats and other functionality.

## Log Format Validation Tests

The `test_log_format.py` module provides comprehensive validation of log files against the standardized format specified in `LOG_FORMAT_SPECIFICATION.md`.

### Running Tests

#### Option 1: Run all unit tests
```bash
# From project root
python -m pytest tests/

# Or using unittest
python -m unittest discover tests/

# Run with verbose output
python -m pytest tests/ -v
```

#### Option 2: Run tests on a specific directory
```bash
# Validate all log files in a specific run directory
python tests/test_log_format.py runs/cifar100

# Validate a specific dataset
python tests/test_log_format.py runs/mnist

# Validate all runs
python tests/test_log_format.py runs/
```

#### Option 3: Run specific test cases
```bash
# Run only CIFAR-100 tests
python -m pytest tests/test_log_format.py::TestLogFormat::test_cifar100_greedy_log -v

# Run all MNIST tests
python -m pytest tests/test_log_format.py -k mnist -v
```

### What Gets Validated

The test suite checks:

#### Required Fields
- **Training entries**: `phase`, `epoch`, `iter`, `step`, `loss`, `acc`, `timestamp`
- **Validation entries**: `phase`, `epoch`, `iter`, `acc`, `timestamp`

#### Field Types
- `phase`: string ('train' or 'val')
- `epoch`: integer (≥ 1)
- `iter`: integer (≥ 1 for train, 0 for val)
- `step`: integer (≥ 1, train only)
- `loss`: float/int (train only)
- `acc`: float/int
- `timestamp`: string (ISO format)
- `metrics`: dict (optional)
- `train_metrics`: dict (optional)

#### Forbidden Fields
- Validation entries should NOT have `loss` field

#### Metrics Structure
- Metrics keys follow `layer_N` convention
- Each layer contains a dictionary of metric names to numeric values
- Metric values must be numeric (or NaN)

### Test Output

Tests provide clear feedback:

```
✅ Valid - All checks passed
⚠️  Warnings - Non-critical issues found (test still passes)
❌ Invalid - Critical errors found (test fails)
```

### Adding New Tests

To add tests for new log files:

1. Add a new test method to `TestLogFormat` class:
```python
def test_my_new_log(self):
    """Test description."""
    log_path = self.test_data_dir / "path" / "to" / "log.jsonl"
    if log_path.exists():
        self._validate_log_file(str(log_path))
    else:
        self.skipTest(f"Log file not found: {log_path}")
```

2. Run the tests to verify:
```bash
python -m pytest tests/test_log_format.py::TestLogFormat::test_my_new_log -v
```

### Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Validate log formats
  run: |
    python -m pytest tests/test_log_format.py -v
```

### Validation in Scripts

You can use the validator in your own scripts:

```python
from tests.test_log_format import LogFormatValidator

validator = LogFormatValidator()
is_valid, errors, warnings = validator.validate_log_file("path/to/log.jsonl")

if is_valid:
    print("✅ Log file is valid")
else:
    print("❌ Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

## Dependencies

The test suite requires:
- Python 3.8+
- pytest (optional, but recommended)

Install test dependencies:
```bash
pip install pytest
```

## Test Coverage

Current test coverage includes:
- ✅ All CIFAR-100 logs (greedy + MLP)
- ✅ All CIFAR-10 logs (greedy + MLP)
- ✅ All MNIST logs (greedy + MLP)
- ✅ Unit tests for individual entry validation
- ✅ Tests for edge cases and error conditions

## Troubleshooting

### Tests are skipped
If tests are being skipped, the log files may not exist at the expected paths. Check that:
1. The runs directory exists
2. Log files are in the expected subdirectories
3. File names match `log.jsonl`

### Tests fail after migration
If tests fail after running the migration script:
1. Check the migration summary for any warnings
2. Verify backup files were created
3. Review specific error messages from the test output
4. You can restore from backups if needed

### Performance
For large log files (>50K entries), validation may take a few seconds. This is normal and ensures comprehensive checking.

