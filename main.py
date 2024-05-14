import math
from argparse import ArgumentParser
from pathlib import Path

import toml

from llm_benchmark.runs import schedule_runs, schedule_mbz_adapt_runs
from llm_benchmark.generate_config import Range, generate
from llm_benchmark.reports import make_report
from llm_benchmark.runconfig import Model, RunConfig, from_dict


def int_or_interval(val: str) -> Range:
    try:
        # Simple integer case.
        val = int(val)
        return val, val
    except ValueError:
        assert val.count(":") == 1
        start, end = val.split(":")
        start = 1 if start == "" else int(start)
        end = math.inf if end == "" else int(end)
        return start, end

# Determine the maximum mb size that fits into gpu memory and write it to original config file.
def test_mbz_run(run_files: list[Path], gpu_budget: int, gpu_per_node: int, run_dir: Path,
        template_dir: Path, testrun_dir: Path):
    # Get configurations to run.
    configs = []
    for run_file in run_files:
        configs += toml.load(run_file)["configs"]
    configs = [from_dict(RunConfig, config) for config in configs] 

    # Run the configs.
    schedule_mbz_adapt_runs(configs, gpu_budget, gpu_per_node, run_dir, testrun_dir, template_dir)
    

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
    parser.add_argument("--testrun-dir", type=Path, default=Path("testruns"))
    subparsers = parser.add_subparsers(dest="action")


    # Run arguments.
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-r", "--run_files", type=Path, nargs="+",
                            help="The .toml runfiles")
    run_parser.add_argument("--run-mode", type=str, default="mbz", choices=["default", "mbz"],
                            help="The run mode, mbz stands for micro batch size adaptation.")
    run_parser.add_argument("-g", "--gpu-budget", type=int, default=128,
                            help="Limit the nodes to ask for")
    run_parser.add_argument("--gpu-per-node", type=int, default=4,
                            help="How many gpus are per each node")
    run_parser.add_argument("--template-dir", type=Path, default=Path("templates"))

    # Analyze arguments.
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("-o", "--out", type=Path, default=Path("reports/report"))
    analyze_parser.add_argument("--exists-ok", action="store_true")

    # Generate config arguments.
    gen_parser = subparsers.add_parser(
        "generate-config",
        help=("Automatically generate all configurations with the specifications given. "
              "For gpu,tp,dp,pp,mbz, these specifications must take the form of either "
              "a single integer (e.g. `1`), an open interval (e.g. `:`, `1:` or `:10`), or a "
              "closed interval (e.g. `1:10`). These intervals will be taken inclusively. "
              "The micro batch size must not be an infinite interval (e.g `:10` is allowed "
              "but `10:` is not. And either the gpu must be finite interval or all "
              "tp,pp,dp must be finite")
    )
    gen_parser.add_argument("out", type=Path, help="Output .toml file to dump configurations")
    gen_parser.add_argument("-g", "--gpu", type=int_or_interval, default=int_or_interval("2"))
    gen_parser.add_argument("-t", "--tp", type=int_or_interval, default=int_or_interval(":"))
    gen_parser.add_argument("-d", "--dp", type=int_or_interval, default=int_or_interval(":"))
    gen_parser.add_argument("-p", "--pp", type=int_or_interval, default=int_or_interval(":"))
    gen_parser.add_argument("-z", "--mbz", type=int_or_interval, default=int_or_interval(":8"))
    gen_parser.add_argument("-m", "--model", type=lambda name: Model[name], nargs="+",
                            default=[Model.llama_3_8b])
    gen_parser.add_argument("-s", "--seq", type=int, nargs="+", default=[4096])

    # Call main.
    args = parser.parse_args()
    if args.action == "run":
        if args.run_mode == "default":
            run(args.run_files, args.gpu_budget, args.gpu_per_node, args.run_dir.absolute(),
                args.template_dir.absolute())
        elif args.run_mode == "mbz":
            args.testrun_dir.mkdir(exist_ok=True)     
            test_mbz_run(args.run_files, args.gpu_budget, args.gpu_per_node, args.run_dir.absolute(),
                args.template_dir.absolute(), args.testrun_dir.absolute())
    elif args.action == "analyze":
        make_report(args.run_dir, args.out, args.exists_ok)
    else:
        generate(args.out, args.tp, args.dp, args.pp, args.gpu, args.mbz, args.model, args.seq)
