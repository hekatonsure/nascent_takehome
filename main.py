"""
Data Quality Pipeline Orchestrator

Coordinates the full data quality analysis, repair, and validation pipeline.

Usage:
    python main.py                          # Run full pipeline (default)
    python main.py --all                    # Run full pipeline (explicit)
    python main.py --quality                # Data quality analysis only
    python main.py --repair                 # Data repair only
    python main.py --validate               # Validation only
    python main.py --explore                # Exploratory analysis only
    python main.py --quality --repair       # Quality + repair
"""

import subprocess, sys, argparse
from pathlib import Path


def _run_script(script_path: str, script_args: list[str]|None = None, description: str = "") -> bool:
    if description:
        print(f"\n{'='*70}")
        print(f"RUNNING: {description}")
        print(f"{'='*70}\n")

    cmd = [sys.executable, script_path]
    if script_args:
        cmd.extend(script_args)

    # Convert subprocess errors to domain errors (fail fast on unexpected errors)
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Script {script_path} failed with exit code {e.returncode}")
        print(f"Command: {' '.join(cmd)}")
        return False


def _run_pipeline(args: argparse.Namespace) -> bool:
    scripts_dir = Path(__file__).parent / "5scripts"
    all_success = True

    # Stage 1: Data Quality Analysis
    if args.quality or args.all:
        success = _run_script(
            str(scripts_dir / "1_quality.py"),
            description="Stage 1: Data Quality Analysis"
        )
        if not success:
            print("\n[FAIL] Data quality analysis failed.")
            all_success = False
            if not args.continue_on_error:
                return False

    # Stage 2: Data Repair (Conservative)
    if args.repair or args.all:
        success = _run_script(
            str(scripts_dir / "2_repair.py"),
            description="Stage 2: Conservative Data Repair"
        )
        if not success:
            print("\n[FAIL] Data repair failed.")
            all_success = False
            if not args.continue_on_error:
                return False

    # Stage 3: Validation
    if args.validate or args.all:
        validation_args = []
        if args.validate_dataset:
            validation_args.extend(['--cleaned', args.validate_dataset])

        success = _run_script(
            str(scripts_dir / "3_validation.py"),
            script_args=validation_args,
            description="Stage 3: Validation"
        )
        if not success:
            print("\n[FAIL] Validation failed.")
            all_success = False
            if not args.continue_on_error:
                return False

    # Stage 4: Exploratory Analysis
    if args.explore or args.all:
        explore_args = ['--analysis', args.explore_type]
        if args.explore_data:
            explore_args.extend(['--data', args.explore_data])

        success = _run_script(
            str(scripts_dir / "4_statistics.py"),
            script_args=explore_args,
            description="Stage 4: Exploratory Analysis"
        )
        if not success:
            print("\n[FAIL] Exploratory analysis failed.")
            all_success = False

    return all_success


def _print_summary() -> None:
    print("\n" + "="*70)
    print("PIPELINE OUTPUTS")
    print("="*70)

    output_dir = Path(__file__).parent / "4output"
    figures_dir = Path(__file__).parent / "3figures"

    if output_dir.exists():
        print("\nData Files (4output/):")
        for file in sorted(output_dir.glob("*.csv")):
            print(f"  - {file.name}")
        for file in sorted(output_dir.glob("*.json")):
            print(f"  - {file.name}")

    if figures_dir.exists():
        print(f"\nVisualizations (3figures/):")
        png_files = list(figures_dir.glob("*.png"))
        print(f"  - {len(png_files)} PNG files")

    print("\n" + "="*70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Data Quality Pipeline Orchestrator',
        epilog='Example: python main.py --all'
    )

    # Pipeline stages
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline (quality + repair + validation + exploratory)')
    parser.add_argument('--quality', action='store_true',
                        help='Run data quality analysis only')
    parser.add_argument('--repair', action='store_true',
                        help='Run data repair only')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation only')
    parser.add_argument('--explore', action='store_true',
                        help='Run exploratory analysis')

    # Options
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue pipeline even if a stage fails')
    parser.add_argument('--validate-dataset', type=str,
                        help='Path to cleaned dataset for validation (default: auto-detect)')
    parser.add_argument('--explore-type', choices=['sentinel', 'outliers', 'statistical', 'all'],
                        default='all', help='Type of exploratory analysis')
    parser.add_argument('--explore-data', type=str,
                        help='Path to data for exploratory analysis')

    args = parser.parse_args()

    # If no flags specified, default to --all
    if not any([args.all, args.quality, args.repair, args.validate, args.explore]):
        args.all = True

    # Run pipeline
    print("="*70)
    print("DATA QUALITY PIPELINE ORCHESTRATOR")
    print("="*70)
    print(f"\nStages to run:")
    if args.all:
        print("  - Data Quality Analysis")
        print("  - Conservative Data Repair")
        print("  - Validation")
        print(f"  - Exploratory Analysis ({args.explore_type})")
    else:
        if args.quality:
            print("  - Data Quality Analysis")
        if args.repair:
            print("  - Conservative Data Repair")
        if args.validate:
            print("  - Validation")
        if args.explore:
            print(f"  - Exploratory Analysis ({args.explore_type})")

    print()

    # Execute pipeline
    success = _run_pipeline(args)

    # Print summary
    _print_summary()

    # Final status
    if success:
        print("\n[SUCCESS] Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n[WARNING] Pipeline completed with errors. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
