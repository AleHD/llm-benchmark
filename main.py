from argparse import ArgumentParser
from pathlib import Path

import toml

from llm_benchmark.runs import schedule_runs
from llm_benchmark.reports import make_report
from llm_benchmark.runconfig import Model, RunConfig, from_dict



def run(run_files: list[Path], gpu_budget: int, gpu_per_node: int, run_dir: Path,
        template_dir: Path):
    # Get configurations to run.
    configs = []
    for run_file in run_files:
        configs += toml.load(run_file)["configs"]
    configs = [from_dict(RunConfig, config) for config in configs]

    # Run the configs.
    schedule_runs(configs, gpu_budget, gpu_per_node, run_dir, template_dir)


if __name__ == "__main__":
    # General arguments.
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    subparsers = parser.add_subparsers(dest="action")


    # Run arguments.
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-r", "--run-files", type=Path, required=True, nargs="+",
                            help="The .toml runfile")
    run_parser.add_argument("-g", "--gpu-budget", type=int, default=128,
                            help="Limit the nodes to ask for")
    run_parser.add_argument("--gpu-per-node", type=int, default=4,
                            help="How many gpus are per each node")
    run_parser.add_argument("--template-dir", type=Path, default=Path("templates"))

    # Analyze arguments.
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("-o", "--out", type=Path, default=Path("reports/report"))
    analyze_parser.add_argument("--exists-ok", action="store_true")

    # Call main.
    args = parser.parse_args()
    if args.action == "run":
        run(args.run_files, args.gpu_budget, args.gpu_per_node, args.run_dir.absolute(),
            args.template_dir.absolute())
    else:
        make_report(args.run_dir, args.out, args.exists_ok)
