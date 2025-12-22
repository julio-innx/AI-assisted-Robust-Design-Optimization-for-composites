# run_all_cases.py
import subprocess
import time
from pathlib import Path

def run_case(case_dir, ansys_exe="ansys222", timeout=3600):
    """
    Run one ANSYS case with correct working directory.
    
    Args:
        case_dir (Path): Folder containing main_script.mac
        ansys_exe (str): e.g., 'ansys222' or full path like 'C:\\Program Files\\ANSYS Inc\\v222\\ansys\\bin\\winx64\\ansys222'
        timeout (int): Max seconds to wait
    """
    case_dir = Path(case_dir).resolve()
    main_script = case_dir / "main_script.mac"
    log_file = case_dir / "ansys_output.out"

    if not main_script.exists():
        print(f"❌ Missing: {main_script}")
        return False

    # Build command
    cmd = [
        str(ansys_exe),
        "-b",                           # Batch mode
        "-i", str(main_script),         # Input script
        "-o", str(log_file)             # Output log
    ]

    print(f"🔧 Running: {case_dir.name}")
    try:
        result = subprocess.run(
            cmd,
            cwd=case_dir,                # ✅ CRITICAL: Set working dir!
            timeout=timeout,
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Optional: pop up window
        )

        # Save logs
        with open(case_dir / "stdout.log", "w", encoding="utf-8") as f:
            f.write(result.stdout)
        with open(case_dir / "stderr.log", "w", encoding="utf-8") as f:
            f.write(result.stderr)

        if result.returncode == 0:
            print(f"✅ Success: {case_dir.name}")
            return True
        else:
            print(f"❌ Failed: {case_dir.name} | Code: {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {case_dir.name}")
        return False
    except FileNotFoundError:
        print(f"💥 ANSYS executable not found: {ansys_exe}")
        print("💡 Try using full path:")
        print("   C:\\Program Files\\ANSYS Inc\\v222\\ansys\\bin\\winx64\\ansys222")
        return False
    except Exception as e:
        print(f"🔥 Error on {case_dir.name}: {e}")
        return False


def run_all_cases(cases_root="cases", ansys_exe=None, start_from=None, max_jobs=None):
    """
    Run all simulation cases.
    """
    cases_root = Path(cases_root)
    if not cases_root.exists():
        print(f"❌ Cases folder not found: {cases_root}")
        return

    # 🔧 Default ANSYS path for Windows
    if ansys_exe is None:
        # Try common install path (v222)
        ansys_exe = r"C:\Program Files\ANSYS Inc\v242\ansys\bin\winx64\MAPDL.exe"
        print(f"🔍 Using default ANSYS: {ansys_exe}")

    # Check if exists, fallback to PATH
    ansys_path = Path(ansys_exe)
    if not ansys_path.exists():
        print(f"⚠️  ANSYS not found at {ansys_path}")
        print("    Falling back to system PATH (e.g., 'ansys222')")
        ansys_exe = "ansys222"  # Hope it's in PATH

    # Get all case folders
    case_dirs = sorted([d for d in cases_root.iterdir() if d.is_dir() and d.name.startswith("case_")])
    print(f"🔍 Found {len(case_dirs)} cases")

    # Filter by start/max
    if start_from is not None:
        case_dirs = [d for d in case_dirs if int(d.name.split("_")[1]) >= start_from]
    if max_jobs is not None:
        case_dirs = case_dirs[:max_jobs]

    if not case_dirs:
        print("🎉 Nothing to run.")
        return

    print(f"🚀 Running {len(case_dirs)} simulations...\n")

    failed = []
    for i, case_dir in enumerate(case_dirs):
        success = run_case(case_dir, ansys_exe=ansys_exe)
        if not success:
            failed.append(case_dir.name)
        print("")  # Spacer

    # Final report
    print("="*60)
    print("📊 BATCH SIMULATION COMPLETE")
    print(f"  Total: {len(case_dirs)}")
    print(f"  Success: {len(case_dirs) - len(failed)}")
    print(f"  Failed: {len(failed)}")
    if failed:
        print("📋 Failed Jobs:")
        for name in failed:
            print(f"  ❌ {name}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ANSYS batch on all case folders (Windows)")
    parser.add_argument("--cases-root", type=str, default="cases", help="Folder with case_xxx dirs")
    parser.add_argument("--ansys-exe", type=str, help="Full path to ANSYS exe (optional)")
    parser.add_argument("--start-from", type=int, help="Resume from case_xxx")
    parser.add_argument("--max-jobs", type=int, help="Limit number of runs")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")

    args = parser.parse_args()

    if args.dry_run:
        root = Path(args.cases_root)
        dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("case_")])
        pending = [d for d in dirs]
        if args.start_from:
            pending = [d for d in pending if int(d.name.split("_")[1]) >= args.start_from]
        if args.max_jobs:
            pending = pending[:args.max_jobs]
        print(f"🎯 Dry run: {len(pending)} jobs")
        for d in pending:
            print(f"  → {d}")
    else:
        run_all_cases(
            cases_root=args.cases_root,
            ansys_exe=args.ansys_exe,
            start_from=args.start_from,
            max_jobs=args.max_jobs
        )
