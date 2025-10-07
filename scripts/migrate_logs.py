#!/usr/bin/env python3
"""
Log format migration script for chain-of-metric-predictors.

This script migrates old log files to the new standardized format:
- Greedy logs: Rename acc_last to acc, remove acc_layers, remove loss from validation
- Baseline logs: Add step field to training entries, add acc field (set to 0 if not present)

Usage:
    python scripts/migrate_logs.py --log_file <path_to_log.jsonl>
    
Or to migrate both runs:
    python scripts/migrate_logs.py --run_dir runs/cifar100
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


def backup_log_file(log_path: str) -> str:
    """Create a backup of the log file with timestamp."""
    backup_path = f"{log_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(log_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path


def migrate_greedy_entry(entry: dict) -> dict:
    """Migrate a greedy trainer log entry to new format."""
    migrated = entry.copy()
    
    # Rename acc_last to acc
    if 'acc_last' in migrated:
        migrated['acc'] = migrated.pop('acc_last')
    
    # Remove loss field from validation entries
    if migrated.get('phase') == 'val' and 'loss' in migrated:
        del migrated['loss']
    
    # Remove acc_layers from validation entries
    if migrated.get('phase') == 'val' and 'acc_layers' in migrated:
        del migrated['acc_layers']
    
    return migrated


def migrate_baseline_entry(entry: dict, global_step: int, train_loader_len: int) -> tuple:
    """
    Migrate a baseline trainer log entry to new format.
    
    Returns: (migrated_entry, new_global_step)
    """
    migrated = entry.copy()
    
    if migrated.get('phase') == 'train':
        # Add step field if not present
        if 'step' not in migrated:
            migrated['step'] = global_step
            global_step += 1
        
        # Add acc field if not present (set to 0 as placeholder - cannot compute retroactively)
        if 'acc' not in migrated:
            # Note: We cannot compute accuracy retroactively from logs
            # Setting to 0 as a placeholder to indicate missing data
            migrated['acc'] = 0.0
    
    return migrated, global_step


def detect_trainer_type(log_path: str) -> str:
    """Detect whether log is from greedy or baseline trainer."""
    with open(log_path, 'r') as f:
        # Check first training entry
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get('phase') == 'train':
                    if 'acc_last' in entry:
                        return 'greedy'
                    else:
                        return 'baseline'
            except json.JSONDecodeError:
                continue
    return 'unknown'


def migrate_log_file(log_path: str, trainer_type: str = None, create_backup: bool = True) -> dict:
    """
    Migrate a log file to the new format.
    
    Args:
        log_path: Path to the log file
        trainer_type: 'greedy' or 'baseline' (auto-detected if None)
        create_backup: Whether to create a backup before migration
    
    Returns:
        Dictionary with migration statistics
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    # Auto-detect trainer type if not specified
    if trainer_type is None:
        trainer_type = detect_trainer_type(log_path)
        print(f"Detected trainer type: {trainer_type}")
    
    if trainer_type == 'unknown':
        raise ValueError(f"Could not detect trainer type for {log_path}")
    
    # Create backup
    if create_backup:
        backup_path = backup_log_file(log_path)
    
    # Read all entries
    entries = []
    with open(log_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Read {len(entries)} log entries")
    
    # Migrate entries
    migrated_entries = []
    global_step = 1
    stats = {
        'total': len(entries),
        'migrated': 0,
        'train_entries': 0,
        'val_entries': 0,
        'changes': {
            'acc_last_to_acc': 0,
            'removed_loss': 0,
            'removed_acc_layers': 0,
            'added_step': 0,
            'added_acc_placeholder': 0,
        }
    }
    
    for entry in entries:
        if trainer_type == 'greedy':
            migrated = migrate_greedy_entry(entry)
            
            # Track changes
            if 'acc_last' in entry:
                stats['changes']['acc_last_to_acc'] += 1
            if entry.get('phase') == 'val' and 'loss' in entry:
                stats['changes']['removed_loss'] += 1
            if entry.get('phase') == 'val' and 'acc_layers' in entry:
                stats['changes']['removed_acc_layers'] += 1
        
        elif trainer_type == 'baseline':
            migrated, global_step = migrate_baseline_entry(entry, global_step, 0)
            
            # Track changes
            if entry.get('phase') == 'train':
                if 'step' not in entry:
                    stats['changes']['added_step'] += 1
                if 'acc' not in entry:
                    stats['changes']['added_acc_placeholder'] += 1
        
        migrated_entries.append(migrated)
        stats['migrated'] += 1
        
        if migrated.get('phase') == 'train':
            stats['train_entries'] += 1
        elif migrated.get('phase') == 'val':
            stats['val_entries'] += 1
    
    # Write migrated entries back to the file
    with open(log_path, 'w') as f:
        for entry in migrated_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\nMigration complete!")
    print(f"  Total entries: {stats['total']}")
    print(f"  Training entries: {stats['train_entries']}")
    print(f"  Validation entries: {stats['val_entries']}")
    print(f"\nChanges made:")
    for change_type, count in stats['changes'].items():
        if count > 0:
            change_name = change_type.replace('_', ' ').title()
            print(f"  {change_name}: {count}")
    
    if stats['changes']['added_acc_placeholder'] > 0:
        print(f"\n⚠️  Warning: Added {stats['changes']['added_acc_placeholder']} placeholder accuracy values (0.0)")
        print("   Accuracy cannot be computed retroactively from logs without predictions.")
    
    return stats


def migrate_run_directory(run_dir: str, create_backup: bool = True):
    """Migrate all log.jsonl files in a run directory."""
    run_path = Path(run_dir)
    
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Find all log.jsonl files
    log_files = list(run_path.rglob('log.jsonl'))
    
    if not log_files:
        print(f"No log.jsonl files found in {run_dir}")
        return
    
    print(f"Found {len(log_files)} log files to migrate")
    print("=" * 80)
    
    for i, log_file in enumerate(log_files, 1):
        print(f"\n[{i}/{len(log_files)}] Migrating: {log_file}")
        print("-" * 80)
        
        try:
            migrate_log_file(str(log_file), create_backup=create_backup)
        except Exception as e:
            print(f"❌ Error migrating {log_file}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✅ All migrations complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate log files to new standardized format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate a single log file
  python scripts/migrate_logs.py --log_file runs/cifar100/greedy/0002_2025-10-04_14-06-27/log.jsonl
  
  # Migrate all logs in a directory
  python scripts/migrate_logs.py --run_dir runs/cifar100
  
  # Migrate without creating backups (not recommended)
  python scripts/migrate_logs.py --run_dir runs/cifar100 --no-backup
        """
    )
    
    parser.add_argument('--log_file', type=str,
                       help='Path to a single log file to migrate')
    parser.add_argument('--run_dir', type=str,
                       help='Path to run directory (will migrate all log.jsonl files found)')
    parser.add_argument('--trainer_type', type=str, choices=['greedy', 'baseline'],
                       help='Trainer type (auto-detected if not specified)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files (not recommended)')
    
    args = parser.parse_args()
    
    if not args.log_file and not args.run_dir:
        parser.error("Must specify either --log_file or --run_dir")
    
    create_backup = not args.no_backup
    
    try:
        if args.log_file:
            migrate_log_file(args.log_file, args.trainer_type, create_backup)
        elif args.run_dir:
            migrate_run_directory(args.run_dir, create_backup)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()

