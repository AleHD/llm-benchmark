from argparse import ArgumentParser
from pathlib import Path

from llm_benchmark.reports import make_report
from llm_benchmark.runs import schedule_runs, show_status


if __name__ == "__main__":
    # General arguments.
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    subparsers = parser.add_subparsers(dest="action")

    # Run arguments.
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("run_file", type=Path, help="The .toml runfile")
    run_parser.add_argument("-l", "--launcher", type=Path, help="Path to the launcher",
                            default=Path(__file__).parent.parent/"pretrain"/"launcher.py")

    # Status arguments.
    run_parser = subparsers.add_parser("status")

    # Analyze arguments.
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("out", type=Path, default=Path("reports/report"))
    analyze_parser.add_argument("-e", "--exist-ok", action="store_true")

    # Call main.
    args = parser.parse_args()
    if args.action == "run":
        schedule_runs(args.run_dir.absolute(), args.run_file.absolute(), args.launcher.absolute())
    elif args.action == "status":
        show_status(args.run_dir.absolute())
    elif args.action == "analyze":
        make_report(args.run_dir.absolute(), args.out.absolute(), args.exist_ok)
    else:
        raise KeyError(f"Unknown action {args.action}")
