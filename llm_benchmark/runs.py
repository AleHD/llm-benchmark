from __future__ import annotations

import dataclasses
import shutil
import time
from enum import Enum
from pathlib import Path
from subprocess import Popen

import toml
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import squeue
from .runconfig import (RunConfig, from_dict, to_dict, get_llama_config,
                        get_llama_tokenizer)


class RunStatus(Enum):
    init = "init"
    running = "running"
    failed = "failed"
    completed = "completed"
    oom = "oom"


class Run:
    def __init__(self, config: RunConfig, gpu_budget: int, gpu_per_node: int,
                 run_dir: Path, template_dir: Path):
        self.config = config
        self.gpu_budget = gpu_budget
        self.gpu_per_node = gpu_per_node

        self.gpu = config.tp*config.pp*config.dp
        assert self.gpu <= gpu_budget
        if self.gpu <= gpu_per_node:
            self.n_nodes = 1
            self.n_proc_per_node = self.gpu
        else:
            assert self.gpu % gpu_per_node == 0
            self.n_nodes = self.gpu//gpu_per_node
            self.n_proc_per_node = gpu_per_node

        # Try to get id from identical previous run.
        run_id = None
        for path in run_dir.iterdir():
            if not path.is_dir():
                continue
            other_config = toml.load(path/"run_config.toml")
            if config == from_dict(RunConfig, other_config):  # Other config matches.
                run_id = path.name
                with open(path/"status.txt") as f:
                    other_status = RunStatus[f.read().strip()]
                if other_status == RunStatus.failed and not is_oom(path):
                    shutil.rmtree(path)
                    status = RunStatus.init
                elif other_status == RunStatus.failed:  # and is oom
                    status = RunStatus.oom
                    with open(path/"status.txt", "w+") as f:
                        print(status.value, file=f)
                else:
                    status = other_status
                break

        # There aren't any other runs with our configuration, get new id.
        if run_id is None:
            status = RunStatus.init
            run_id = "0"
            while (run_dir/run_id).exists():
                run_id = str(int(run_id) + 1)
        self.run_id = run_id
        self.home = run_dir/run_id

        # If the run is not already started or done, initialize directory.
        self.home.mkdir(exist_ok=True)
        (self.home/"logs").mkdir(exist_ok=True)
        if status == RunStatus.init:
            # Write status.
            with open(self.home/"status.txt", "w+") as f:
                print(status.value, file=f)
            # Write config toml.
            with open(self.home/"run_config.toml", "w+") as f:
                toml.dump(to_dict(config), f)
            # Fill sbatch template.
            sbatch = self._get_sbatch(template_dir)
            with open(self.home/"nanotron.sbatch", "w+") as f:
                print(sbatch, file=f)
            # Fill config.
            nano_config = self._get_nano_config(template_dir)
            with open(self.home/"nanotron_config.yaml", "w+") as f:
                print(nano_config, file=f)
            # Copy slurm.toml.
            shutil.copy(template_dir/"slurm.toml", self.home/"slurm.toml")

    def __lt__(self, other: Run) -> bool:
        return int(self.run_id) < int(other.run_id)

    @property
    def status(self) -> RunStatus:
        with open(self.home/"status.txt") as f:
            status = RunStatus["".join(f.read().strip())]
        if status == RunStatus.failed and is_oom(self.home):
            self.status = RunStatus.oom
        with open(self.home/"status.txt") as f:
            return RunStatus["".join(f.read().strip())]

    @status.setter
    def status(self, status: RunStatus):
        self._status = status
        with open(self.home/"status.txt", "w+") as f:
            print(self._status.value, file=f)

    def _get_sbatch(self, template_dir: Path) -> str:
        env = Environment(loader=FileSystemLoader(template_dir),
                          autoescape=select_autoescape())
        template = env.get_template("nanotron.sbatch")
        return template.render(
            run_id=self.run_id,
            run_root=self.home,
            n_nodes=self.n_nodes,
            n_proc_per_node=self.n_proc_per_node,
        )

    def _get_nano_config(self, template_dir: Path) -> str:
        with open(template_dir/"config.yaml") as f:
            data = yaml.safe_load(f)
        data["general"]["run"] = f"benchmark-{self.run_id}"
        data["model"]["model_config"].update(get_llama_config(self.config.model))
        data["parallelism"].update({"tp": self.config.tp, "dp": self.config.dp,
                                    "pp": self.config.pp})
        data["tokenizer"] = {"tokenizer_name_or_path": get_llama_tokenizer(self.config.model)}
        acc = self.config.batch_size//(self.config.micro_batch_size*self.config.dp)
        data["tokens"].update({
            "micro_batch_size": self.config.micro_batch_size,
            "sequence_length": self.config.sequence_length,
            "batch_accumulation_per_replica": acc,
        })
        data["optimizer"]["zero_stage"] = 1 if self.config.dp > 1 else 0
        data["wandb"] = {"dir": str(self.home)}
        return yaml.dump(data)


def is_oom(home: Path) -> bool:
    # Check log, if "OutOfMemoryError" is mentioned, then we set OOM as status.
    if not (home/"logs").exists():
        return False
    logs = list((home/"logs").iterdir())
    if len(logs) == 0:
        return False
    assert len(logs) == 1
    with open(logs[0]) as f:
        try:
            for line in f:
                if "OutOfMemoryError" in line:
                    return True
        except UnicodeError:
            return False
    return False


def wait(t: float):
    time.sleep(2)
    t -= 2
    while t > 0:
        print(".", end="", flush=True)
        time.sleep(2)
        t -= 2
    print()


def schedule_runs(configs: list[RunConfig], gpu_budget: int, gpu_per_node: int,
                  run_dir: Path, template_dir: Path):
    # Print info.
    print("Scheduling", len(configs), "runs:")
    print(*configs, sep="\n")
    print("-"*10)
    print()

    # Make sure all of the configurations are valid.
    runs = [Run(config, gpu_budget, gpu_per_node, run_dir, template_dir)
            for config in configs]

    print("Starting runs...")
    skipped = 0
    for run in runs:
        # Skip if needed.
        print("Inspecting run", run.run_id)
        if run.status != RunStatus.init:
            print("Skipping", run.run_id, "because has status", run.status.value)
            print("---")
            print()
            skipped += 1
            continue

        # Wait until we have all available GPUs.
        while squeue.used_nodes() + run.n_nodes > gpu_budget//gpu_per_node:
            current_runs = ",".join(str(run.run_id) for run in runs
                                    if run.status == RunStatus.running)
            print("Can't start this run with current GPU budget, used nodes:",
                  squeue.used_nodes(), "Runs:", current_runs, "Waiting",
                  end="", flush=True)
            wait(10)
        print("GPU resources available, starting run!")

        # Starting run.
        run.status = RunStatus.running
        with Popen(["sbatch", "--contiguous", "nanotron.sbatch"], cwd=run.home) as proc:
            proc.wait()
        time.sleep(2)  # Just to make sure squeue is updated next time we call.
        print("---")
        print()
    print()

    # Wait until all jobs end.
    print("All runs have been scheduled, waiting for them to end...")
    while len(remaining := sorted([run for run in runs if run.status == RunStatus.running])):
        print("Waiting for", len(remaining), "runs, using", squeue.used_nodes(),
              "nodes. Runs:", ",".join(run.run_id for run in remaining), end="", flush=True)
        wait(10)
    print("All runs have ended! Congrats! Now run the analyze tool")
    print("---")
    print()

    # Print final stats.
    print("Final stats:")
    print("Total runs requested:", len(runs))
    print("Total runs skipped:", skipped)
    print("Total runs completed:", len([run for run in runs if run.status == RunStatus.completed]))
    print("Total runs failed:", len([run for run in runs if run.status == RunStatus.failed]))
    print("Total runs oom:", len([run for run in runs if run.status == RunStatus.oom]))
    print("Goodbye")
