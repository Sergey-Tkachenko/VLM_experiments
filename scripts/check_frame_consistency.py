#!/usr/bin/env python3
"""Check if frames in DADA dataset folders are consistent for FFmpeg image2 demuxer.

FFmpeg's image2 demuxer requires:
- Numeric filenames with consistent zero-padding (e.g., %04d.png)
- Usually contiguous numbering (no gaps)
- All files in the same format

This script validates these requirements across all dataset folders.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables
load_dotenv(REPO_ROOT / ".env")


def extract_frame_number(filename: str) -> Optional[int]:
    """Extract frame number from filename like '0001.png' or '1.png'.

    Args:
        filename: The filename to parse.

    Returns:
        Frame number if valid, None otherwise.
    """
    match = re.match(r'^(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    return None


def analyze_frame_sequence(image_paths: List[Path]) -> Dict[str, Any]:
    """Analyze a sequence of frame images for FFmpeg image2 compatibility.

    Args:
        image_paths: List of paths to image files.

    Returns:
        Dictionary with analysis results.
    """
    if not image_paths:
        return {
            'valid': False,
            'error': 'No image files found',
            'frame_count': 0,
            'issues': ['Empty directory']
        }

    # Extract frame numbers and validate filenames
    frame_numbers = []
    invalid_files = []
    non_png_files = []

    for path in image_paths:
        if path.suffix.lower() != '.png':
            non_png_files.append(path.name)
            continue

        frame_num = extract_frame_number(path.name)
        if frame_num is None:
            invalid_files.append(path.name)
        else:
            frame_numbers.append(frame_num)

    if non_png_files:
        return {
            'valid': False,
            'error': f'Non-PNG files found: {non_png_files}',
            'frame_count': len(frame_numbers),
            'issues': [f'Non-PNG files: {non_png_files}']
        }

    if invalid_files:
        return {
            'valid': False,
            'error': f'Invalid filename format: {invalid_files}',
            'frame_count': len(frame_numbers),
            'issues': [f'Invalid filenames: {invalid_files}']
        }

    # Sort frame numbers
    frame_numbers.sort()

    # Check for zero-padding consistency by looking at actual filename lengths
    filename_lengths = [len(path.stem) for path in image_paths]
    min_digits = min(filename_lengths) if filename_lengths else 0
    max_digits = max(filename_lengths) if filename_lengths else 0
    padding_consistent = min_digits == max_digits

    # Check for gaps in numbering
    if frame_numbers:
        expected_range = set(range(frame_numbers[0], frame_numbers[-1] + 1))
        actual_range = set(frame_numbers)
        gaps = expected_range - actual_range
        has_gaps = len(gaps) > 0
    else:
        has_gaps = False
        gaps = set()

    # Check if numbering starts from 1 (typical FFmpeg expectation)
    starts_from_one = frame_numbers and frame_numbers[0] == 1

    # Determine if valid for FFmpeg image2
    issues = []
    if not padding_consistent:
        issues.append(f'Inconsistent zero-padding: {min_digits}-{max_digits} digits')
    if has_gaps:
        issues.append(f'Gaps in numbering: missing {sorted(gaps)}')
    if not starts_from_one:
        issues.append(f'Does not start from frame 1 (starts from {frame_numbers[0] if frame_numbers else "N/A"})')

    valid = len(issues) == 0

    return {
        'valid': valid,
        'frame_count': len(frame_numbers),
        'first_frame': frame_numbers[0] if frame_numbers else None,
        'last_frame': frame_numbers[-1] if frame_numbers else None,
        'padding_digits': max_digits if padding_consistent else f'{min_digits}-{max_digits}',
        'has_gaps': has_gaps,
        'gaps': sorted(gaps) if has_gaps else [],
        'starts_from_one': starts_from_one,
        'issues': issues
    }


def check_dataset_consistency(dataset_root: Path, max_samples: Optional[int] = None) -> Dict[str, Any]:
    """Check frame consistency across the entire dataset.

    Args:
        dataset_root: Root directory of the DADA dataset.
        max_samples: Maximum number of samples to check (for testing).

    Returns:
        Dictionary with overall analysis results.
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    logger.info("Starting dataset consistency check for: {}", dataset_root)

    results = {
        'total_folders_checked': 0,
        'valid_folders': 0,
        'invalid_folders': 0,
        'folders_with_issues': [],
        'summary_stats': {
            'total_frames': 0,
            'avg_frames_per_folder': 0,
            'min_frames_per_folder': float('inf'),
            'max_frames_per_folder': 0,
            'padding_consistency': True,
            'no_gaps_found': True,
            'all_start_from_one': True,
        }
    }

    # Walk through all type_id/video_id folders
    type_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()])

    for type_dir in type_dirs:
        video_dirs = sorted([d for d in type_dir.iterdir() if d.is_dir()])

        for video_dir in video_dirs:
            if max_samples and results['total_folders_checked'] >= max_samples:
                logger.info("Reached max_samples limit: {}", max_samples)
                break

            results['total_folders_checked'] += 1
            folder_path = f"{type_dir.name}/{video_dir.name}"
            images_dir = video_dir / "images"

            if not images_dir.exists():
                results['invalid_folders'] += 1
                results['folders_with_issues'].append({
                    'folder': folder_path,
                    'issues': ['Missing images/ directory']
                })
                continue

            # Get all files in images directory
            image_files = list(images_dir.glob("*.png"))
            image_files.sort()

            # Analyze the sequence
            analysis = analyze_frame_sequence(image_files)

            if analysis['valid']:
                results['valid_folders'] += 1
            else:
                results['invalid_folders'] += 1
                results['folders_with_issues'].append({
                    'folder': folder_path,
                    'issues': analysis['issues']
                })

            # Update summary stats
            frame_count = analysis['frame_count']
            results['summary_stats']['total_frames'] += frame_count
            results['summary_stats']['min_frames_per_folder'] = min(
                results['summary_stats']['min_frames_per_folder'], frame_count
            )
            results['summary_stats']['max_frames_per_folder'] = max(
                results['summary_stats']['max_frames_per_folder'], frame_count
            )

            if not analysis['starts_from_one']:
                results['summary_stats']['all_start_from_one'] = False

            if analysis['has_gaps']:
                results['summary_stats']['no_gaps_found'] = False

            logger.info(
                "Checked {}/{}: {} frames, valid={}",
                results['total_folders_checked'],
                folder_path,
                frame_count,
                analysis['valid']
            )

    # Calculate average frames per folder
    if results['total_folders_checked'] > 0:
        results['summary_stats']['avg_frames_per_folder'] = (
            results['summary_stats']['total_frames'] / results['total_folders_checked']
        )

    # Handle min_frames edge case
    if results['summary_stats']['min_frames_per_folder'] == float('inf'):
        results['summary_stats']['min_frames_per_folder'] = 0

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the consistency check results."""
    print("\n" + "="*60)
    print("DADA Dataset Frame Consistency Check Summary")
    print("="*60)

    print(f"Total folders checked: {results['total_folders_checked']}")
    print(f"Valid folders: {results['valid_folders']} ({results['valid_folders']/results['total_folders_checked']*100:.1f}%)")
    print(f"Invalid folders: {results['invalid_folders']} ({results['invalid_folders']/results['total_folders_checked']*100:.1f}%)")

    print("\nSummary Statistics:")
    stats = results['summary_stats']
    print(f"  Total frames across dataset: {stats['total_frames']:,}")
    print(f"  Average frames per folder: {stats['avg_frames_per_folder']:.1f}")
    print(f"  Min frames per folder: {stats['min_frames_per_folder']}")
    print(f"  Max frames per folder: {stats['max_frames_per_folder']}")
    print(f"  All folders start from frame 1: {stats['all_start_from_one']}")
    print(f"  No gaps found in any folder: {stats['no_gaps_found']}")

    if results['folders_with_issues']:
        print(f"\nFolders with issues ({len(results['folders_with_issues'])}):")
        for folder_info in results['folders_with_issues'][:10]:  # Show first 10
            print(f"  {folder_info['folder']}: {', '.join(folder_info['issues'])}")

        if len(results['folders_with_issues']) > 10:
            print(f"  ... and {len(results['folders_with_issues']) - 10} more")

        print("\nFor FFmpeg image2 compatibility, all folders should:")
        print("  - Have only PNG files")
        print("  - Use consistent zero-padding (e.g., all filenames have same digit count)")
        print("  - Have contiguous frame numbering (no gaps)")
        print("  - Start from frame 1")
    else:
        print("\n✓ All folders are FFmpeg image2 compatible!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check DADA dataset frame consistency for FFmpeg image2 demuxer"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/datasets/mm-au/Origin/DADA2000/DADA2000"),
        help="Root directory of the DADA dataset"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to check (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    try:
        results = check_dataset_consistency(args.dataset_root, args.max_samples)
        print_summary(results)
    except Exception as e:
        logger.error("Error during consistency check: {}", e)
        raise


if __name__ == "__main__":
    main()
