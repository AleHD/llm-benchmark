import math
from argparse import ArgumentParser
from pathlib import Path

import toml

from llm_benchmark.runs import RunStatus, schedule_runs
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


def run(run_files: list[Path], gpu_budget: int, gpu_per_node: int, run_dir: Path,
        template_dir: Path):
    # Get configurations to run.
    configs = []
    for run_file in run_files:
        configs += toml.load(run_file)["configs"]
    configs = [from_dict(RunConfig, config) for config in configs]

    # Run the configs.
    schedule_runs(configs, gpu_budget, gpu_per_node, run_dir, template_dir)


def lookup(statuses: list[RunStatus], gpus: Range, tps: Range, dps: Range, pps: Range,
           mbzs: Range, models: list[Model], seqs: Range, run_dir: Path):

    def matches(config: RunConfig, status: RunStatus) -> bool:
        if len(statuses) > 0 and status not in statuses:
            return False
        if len(models) > 0 and config.model not in models:
            return False
        if gpus[0] > config.gpu or gpus[1] < config.gpu:
            return False
        if tps[0] > config.tp or tps[1] < config.tp:
            return False
        if dps[0] > config.dp or dps[1] < config.dp:
            return False
        if pps[0] > config.pp or pps[1] < config.pp:
            return False
        if mbzs[0] > config.micro_batch_size or mbzs[1] < config.micro_batch_size:
            return False
        if seqs[0] > config.sequence_length or seqs[1] < config.sequence_length:
            return False
        return True

    for path in sorted(run_dir.iterdir(), key=lambda path: int(path.name)):
        run_config = from_dict(RunConfig, toml.load(path/"run_config.toml"))
        with open(path/"status.txt") as f:
            status = RunStatus[f.read().strip()]
        if matches(run_config, status):
            run_id = path.name
            print("Match:", run_id, run_config)


if __name__ == "__main__":
    # General arguments.
    parser = ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    subparsers = parser.add_subparsers(dest="action")


    # Run arguments.
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("run_files", type=Path, nargs="+",
                            help="The .toml runfiles")
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
    gen_parser.add_argument("-i", "--if", default="True", dest="condition",
                            help=("String to filter out some configuration combinations. "
                                  "Examples: `tp*pp == 1`, `pp in {2, 4, 8}`, `pp/mbz == 1`, etc. "
                                  "Possible variables to use=tp,pp,dp,gpu,mbz,model,seq and the math std library."))

    # Lookup.
    lookup_parser = subparsers.add_parser(
        "lookup",
        help=("Look for runs that match the arguments given. Look at `generate-config` help "
              "for more information about the arguments expected. For arguments that "
              "expect a list of names (--status and --model), passing an empty list of names "
              "assumes match all")
    )
    lookup_parser.add_argument("-s", "--status", type=lambda name: RunStatus[name], nargs="*", default=[])
    lookup_parser.add_argument("-g", "--gpu", type=int_or_interval, default=int_or_interval(":"))
    lookup_parser.add_argument("-t", "--tp", type=int_or_interval, default=int_or_interval(":"))
    lookup_parser.add_argument("-d", "--dp", type=int_or_interval, default=int_or_interval(":"))
    lookup_parser.add_argument("-p", "--pp", type=int_or_interval, default=int_or_interval(":"))
    lookup_parser.add_argument("-z", "--mbz", type=int_or_interval, default=int_or_interval(":"))
    lookup_parser.add_argument("-m", "--model", type=lambda name: Model[name], nargs="*", default=[])
    lookup_parser.add_argument("--seq", type=int_or_interval, default=int_or_interval(":"))

    # Call main.
    args = parser.parse_args()
    if args.action == "run":
        run(args.run_files, args.gpu_budget, args.gpu_per_node, args.run_dir.absolute(),
            args.template_dir.absolute())
    elif args.action == "analyze":
        make_report(args.run_dir.absolute(), args.out.absolute(), args.exists_ok)
    elif args.action == "generate-config":
        generate(args.out.absolute(), args.tp, args.dp, args.pp, args.gpu, args.mbz,
                 args.model, args.seq, args.condition)
    else:
        lookup(args.status, args.gpu, args.tp, args.dp, args.pp, args.mbz, args.model, args.seq,
               args.run_dir.absolute())
