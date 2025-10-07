#!/usr/bin/env python3
"""
Test suite for validating log file format compliance.

This test suite validates that log files comply with the standardized format
specified in LOG_FORMAT_SPECIFICATION.md.
"""

import json
import os
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional


class LogFormatValidator:
    """Validator class for checking log format compliance."""
    
    # Required base fields for all entries
    BASE_FIELDS = {'phase', 'epoch', 'iter'}
    
    # Required fields for training entries
    TRAIN_REQUIRED_FIELDS = BASE_FIELDS | {'step', 'loss', 'acc'}
    
    # Required fields for validation entries
    VAL_REQUIRED_FIELDS = BASE_FIELDS | {'acc'}
    
    # Optional but recommended fields
    OPTIONAL_RECOMMENDED_FIELDS = {'timestamp'}
    
    # Fields that should NOT appear in validation entries
    VAL_FORBIDDEN_FIELDS = {'loss'}  # loss should not be in validation
    
    # Optional fields
    OPTIONAL_FIELDS = {'metrics', 'train_metrics'}
    
    # Expected field types
    FIELD_TYPES = {
        'phase': str,
        'epoch': int,
        'iter': int,
        'step': int,
        'loss': (float, int),
        'acc': (float, int),
        'timestamp': str,
        'metrics': dict,
        'train_metrics': dict,
    }
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_entry(self, entry: Dict[str, Any], line_num: int) -> bool:
        """
        Validate a single log entry.
        
        Returns True if valid, False otherwise.
        Errors are accumulated in self.errors.
        """
        valid = True
        
        # Check phase field exists
        if 'phase' not in entry:
            self.errors.append(f"Line {line_num}: Missing 'phase' field")
            return False
        
        phase = entry['phase']
        
        if phase not in ['train', 'val']:
            self.errors.append(f"Line {line_num}: Invalid phase '{phase}', must be 'train' or 'val'")
            return False
        
        # Check required fields based on phase
        if phase == 'train':
            valid &= self._validate_train_entry(entry, line_num)
        elif phase == 'val':
            valid &= self._validate_val_entry(entry, line_num)
        
        # Check field types
        valid &= self._validate_field_types(entry, line_num)
        
        # Check for recommended optional fields
        if 'timestamp' not in entry:
            self.warnings.append(f"Line {line_num}: Missing recommended field 'timestamp'")
        
        # Check metrics structure if present
        if 'metrics' in entry:
            valid &= self._validate_metrics_structure(entry['metrics'], line_num, 'metrics')
        
        if 'train_metrics' in entry:
            valid &= self._validate_metrics_structure(entry['train_metrics'], line_num, 'train_metrics')
        
        return valid
    
    def _validate_train_entry(self, entry: Dict[str, Any], line_num: int) -> bool:
        """Validate training entry has all required fields."""
        valid = True
        
        for field in self.TRAIN_REQUIRED_FIELDS:
            if field not in entry:
                self.errors.append(f"Line {line_num}: Training entry missing required field '{field}'")
                valid = False
        
        # Check that iter is 1-indexed for training
        if 'iter' in entry and entry['iter'] < 1:
            self.errors.append(f"Line {line_num}: Training 'iter' should be 1-indexed, got {entry['iter']}")
            valid = False
        
        # Check that step is present and positive
        if 'step' in entry and entry['step'] < 1:
            self.errors.append(f"Line {line_num}: Training 'step' should be >= 1, got {entry['step']}")
            valid = False
        
        return valid
    
    def _validate_val_entry(self, entry: Dict[str, Any], line_num: int) -> bool:
        """Validate validation entry has required fields and no forbidden fields."""
        valid = True
        
        for field in self.VAL_REQUIRED_FIELDS:
            if field not in entry:
                self.errors.append(f"Line {line_num}: Validation entry missing required field '{field}'")
                valid = False
        
        # Check forbidden fields
        for field in self.VAL_FORBIDDEN_FIELDS:
            if field in entry:
                self.errors.append(f"Line {line_num}: Validation entry should not have '{field}' field")
                valid = False
        
        # Check that iter is 0 for validation
        if 'iter' in entry and entry['iter'] != 0:
            self.errors.append(f"Line {line_num}: Validation 'iter' should be 0, got {entry['iter']}")
            valid = False
        
        # Validation entries should not have step field
        if 'step' in entry:
            self.warnings.append(f"Line {line_num}: Validation entry has 'step' field (not typically expected)")
        
        return valid
    
    def _validate_field_types(self, entry: Dict[str, Any], line_num: int) -> bool:
        """Validate field types are correct."""
        valid = True
        
        for field, value in entry.items():
            if field in self.FIELD_TYPES:
                expected_type = self.FIELD_TYPES[field]
                if not isinstance(value, expected_type):
                    self.errors.append(
                        f"Line {line_num}: Field '{field}' has wrong type. "
                        f"Expected {expected_type}, got {type(value).__name__}"
                    )
                    valid = False
        
        return valid
    
    def _validate_metrics_structure(self, metrics: Dict, line_num: int, field_name: str) -> bool:
        """Validate metrics structure follows the expected format."""
        valid = True
        
        if not isinstance(metrics, dict):
            self.errors.append(f"Line {line_num}: '{field_name}' must be a dictionary")
            return False
        
        # Check that keys are layer names (layer_0, layer_1, etc.)
        for layer_key, layer_metrics in metrics.items():
            if not layer_key.startswith('layer_'):
                self.warnings.append(
                    f"Line {line_num}: Metrics key '{layer_key}' doesn't follow 'layer_N' convention"
                )
            
            if not isinstance(layer_metrics, dict):
                self.errors.append(
                    f"Line {line_num}: Metrics for '{layer_key}' must be a dictionary"
                )
                valid = False
                continue
            
            # Check that metric values are numeric
            for metric_name, metric_value in layer_metrics.items():
                if not isinstance(metric_value, (int, float, type(None))):
                    # Allow NaN as a string (common in JSON)
                    if not (isinstance(metric_value, str) and metric_value.lower() == 'nan'):
                        self.errors.append(
                            f"Line {line_num}: Metric '{layer_key}.{metric_name}' "
                            f"must be numeric, got {type(metric_value).__name__}"
                        )
                        valid = False
        
        return valid
    
    def validate_log_file(self, log_path: str) -> tuple[bool, List[str], List[str]]:
        """
        Validate an entire log file.
        
        Returns:
            tuple: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        if not os.path.exists(log_path):
            self.errors.append(f"Log file not found: {log_path}")
            return False, self.errors, self.warnings
        
        valid = True
        with open(log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    self.errors.append(f"Line {line_num}: Invalid JSON - {e}")
                    valid = False
                    continue
                
                if not self.validate_entry(entry, line_num):
                    valid = False
        
        return valid, self.errors, self.warnings


class TestLogFormat(unittest.TestCase):
    """Test cases for log format validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = LogFormatValidator()
        self.test_data_dir = Path(__file__).parent.parent / "runs"
    
    def _validate_log_file(self, log_path: str):
        """Helper to validate a log file and report results."""
        is_valid, errors, warnings = self.validator.validate_log_file(log_path)
        
        # Print warnings (don't fail on warnings)
        if warnings:
            print(f"\n⚠️  Warnings for {log_path}:")
            for warning in warnings[:10]:  # Limit to first 10
                print(f"  - {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more warnings")
        
        # Assert no errors
        if not is_valid:
            error_msg = f"\n❌ Validation failed for {log_path}:\n"
            for error in errors[:20]:  # Limit to first 20 errors
                error_msg += f"  - {error}\n"
            if len(errors) > 20:
                error_msg += f"  ... and {len(errors) - 20} more errors\n"
            self.fail(error_msg)
        
        print(f"✅ {log_path} - Valid")
    
    def test_cifar100_greedy_log(self):
        """Test CIFAR-100 greedy trainer log format."""
        log_path = self.test_data_dir / "cifar100" / "greedy" / "0002_2025-10-04_14-06-27" / "log.jsonl"
        if log_path.exists():
            self._validate_log_file(str(log_path))
        else:
            self.skipTest(f"Log file not found: {log_path}")
    
    def test_cifar100_mlp_log(self):
        """Test CIFAR-100 MLP trainer log format."""
        log_path = self.test_data_dir / "cifar100" / "mlp" / "0001_2025-10-04_14-06-57" / "log.jsonl"
        if log_path.exists():
            self._validate_log_file(str(log_path))
        else:
            self.skipTest(f"Log file not found: {log_path}")
    
    def test_cifar10_greedy_log(self):
        """Test CIFAR-10 greedy trainer log format."""
        log_path = self.test_data_dir / "cifar10" / "greedy" / "0001_2025-10-03_22-30-20" / "log.jsonl"
        if log_path.exists():
            self._validate_log_file(str(log_path))
        else:
            self.skipTest(f"Log file not found: {log_path}")
    
    def test_cifar10_mlp_log(self):
        """Test CIFAR-10 MLP trainer log format."""
        log_path = self.test_data_dir / "cifar10" / "mlp" / "0001_2025-10-03_22-32-54" / "log.jsonl"
        if log_path.exists():
            self._validate_log_file(str(log_path))
        else:
            self.skipTest(f"Log file not found: {log_path}")
    
    def test_mnist_greedy_log(self):
        """Test MNIST greedy trainer log format."""
        log_path = self.test_data_dir / "mnist" / "greedy" / "0007_2025-10-03_19-08-01" / "log.jsonl"
        if log_path.exists():
            self._validate_log_file(str(log_path))
        else:
            self.skipTest(f"Log file not found: {log_path}")
    
    def test_mnist_mlp_log(self):
        """Test MNIST MLP trainer log format."""
        log_path = self.test_data_dir / "mnist" / "mlp" / "0001_2025-10-03_21-33-01" / "log.jsonl"
        if log_path.exists():
            self._validate_log_file(str(log_path))
        else:
            self.skipTest(f"Log file not found: {log_path}")
    
    def test_training_entry_structure(self):
        """Test that a well-formed training entry passes validation."""
        entry = {
            "phase": "train",
            "epoch": 1,
            "iter": 100,
            "step": 100,
            "loss": 2.5,
            "acc": 0.75,
            "timestamp": "2025-10-05T12:00:00.000000"
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertTrue(is_valid, f"Errors: {self.validator.errors}")
    
    def test_training_entry_without_timestamp(self):
        """Test that training entry without timestamp is still valid (with warning)."""
        entry = {
            "phase": "train",
            "epoch": 1,
            "iter": 100,
            "step": 100,
            "loss": 2.5,
            "acc": 0.75
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertTrue(is_valid, f"Errors: {self.validator.errors}")
        # Should have a warning about missing timestamp
        self.assertTrue(any("timestamp" in warning.lower() for warning in self.validator.warnings))
    
    def test_validation_entry_structure(self):
        """Test that a well-formed validation entry passes validation."""
        entry = {
            "phase": "val",
            "epoch": 1,
            "iter": 0,
            "acc": 0.85,
            "timestamp": "2025-10-05T12:00:00.000000"
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertTrue(is_valid, f"Errors: {self.validator.errors}")
    
    def test_validation_entry_no_loss(self):
        """Test that validation entries with loss field fail validation."""
        entry = {
            "phase": "val",
            "epoch": 1,
            "iter": 0,
            "acc": 0.85,
            "loss": None,  # Should not be present
            "timestamp": "2025-10-05T12:00:00.000000"
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertFalse(is_valid, "Validation entry should not have 'loss' field")
        self.assertTrue(any("should not have 'loss'" in err for err in self.validator.errors))
    
    def test_missing_required_field(self):
        """Test that entries missing required fields fail validation."""
        entry = {
            "phase": "train",
            "epoch": 1,
            "iter": 100,
            # Missing 'step', 'loss', 'acc'
            "timestamp": "2025-10-05T12:00:00.000000"
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertFalse(is_valid, "Entry missing required fields should fail")
        self.assertGreaterEqual(len(self.validator.errors), 3)
    
    def test_wrong_field_type(self):
        """Test that entries with wrong field types fail validation."""
        entry = {
            "phase": "train",
            "epoch": "1",  # Should be int, not string
            "iter": 100,
            "step": 100,
            "loss": 2.5,
            "acc": 0.75,
            "timestamp": "2025-10-05T12:00:00.000000"
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertFalse(is_valid, "Entry with wrong field type should fail")
        self.assertTrue(any("wrong type" in err for err in self.validator.errors))
    
    def test_metrics_structure(self):
        """Test that metrics structure is validated correctly."""
        entry = {
            "phase": "val",
            "epoch": 1,
            "iter": 0,
            "acc": 0.85,
            "timestamp": "2025-10-05T12:00:00.000000",
            "metrics": {
                "layer_0": {
                    "accuracy": 0.8,
                    "cross_entropy": 0.5,
                    "alignment": 0.9
                },
                "layer_1": {
                    "accuracy": 0.85,
                    "cross_entropy": 0.4
                }
            }
        }
        
        is_valid = self.validator.validate_entry(entry, 1)
        self.assertTrue(is_valid, f"Errors: {self.validator.errors}")


def validate_run_directory(run_dir: str) -> bool:
    """
    Validate all log files in a run directory.
    
    Args:
        run_dir: Path to run directory containing log.jsonl files
    
    Returns:
        True if all logs are valid, False otherwise
    """
    validator = LogFormatValidator()
    run_path = Path(run_dir)
    
    if not run_path.exists():
        print(f"❌ Directory not found: {run_dir}")
        return False
    
    # Find all log.jsonl files
    log_files = list(run_path.rglob('log.jsonl'))
    
    if not log_files:
        print(f"⚠️  No log.jsonl files found in {run_dir}")
        return True
    
    print(f"Found {len(log_files)} log file(s) to validate")
    print("=" * 80)
    
    all_valid = True
    for log_file in log_files:
        print(f"\nValidating: {log_file.relative_to(run_path.parent.parent)}")
        print("-" * 80)
        
        is_valid, errors, warnings = validator.validate_log_file(str(log_file))
        
        if warnings:
            print(f"⚠️  {len(warnings)} warning(s)")
            for warning in warnings[:5]:
                print(f"  - {warning}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more")
        
        if is_valid:
            print("✅ Valid")
        else:
            print(f"❌ Invalid - {len(errors)} error(s)")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
            all_valid = False
    
    print("\n" + "=" * 80)
    if all_valid:
        print("✅ All log files are valid!")
    else:
        print("❌ Some log files have validation errors")
    
    return all_valid


if __name__ == '__main__':
    import sys
    
    # If a directory path is provided, validate that directory
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        run_dir = sys.argv[1]
        success = validate_run_directory(run_dir)
        sys.exit(0 if success else 1)
    
    # Otherwise run unit tests
    unittest.main()

