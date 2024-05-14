from __future__ import annotations

from copy import deepcopy
import math
import re
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
                 run_dir: Path, template_dir: Path, test_mbz_run: bool=False):
        self.config = config
        self.gpu_budget = gpu_budget
        self.gpu_per_node = gpu_per_node
        
        self.test_mbz_run = test_mbz_run

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

    def __lt__(self, other: Run) -> bool:
        return int(self.run_id) < int(other.run_id)

    @property
    def status(self) -> RunStatus:
        with open(self.home/"status.txt") as f:
            status = RunStatus["".join(f.read().strip())]
        if status == "failed" and self._is_oom():
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
        data["general"]["run"] = f"{"testrun" if self.test_mbz_run else "benchmark"}-{self.run_id}"
        data["model"]["model_config"].update(get_llama_config(self.config.model))
        data["parallelism"].update({"tp": self.config.tp, "dp": self.config.dp,
                                    "pp": self.config.pp})
        # zero-1 for dp > 1, otherwise zero-0
        if self.config.dp > 1:
            data["optimizer"].update({"zero_stage": 1})
        else:
            data["optimizer"].update({"zero_stage": 0})
        data["tokenizer"] = {"tokenizer_name_or_path": get_llama_tokenizer(self.config.model)}
        acc = self.config.num_micro_batch
        data["tokens"].update({
            "micro_batch_size": self.config.micro_batch_size,
            "sequence_length": self.config.sequence_length,
            "batch_accumulation_per_replica": acc,
        })
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
        for line in f:
            if "OutOfMemoryError" in line:
                return True
    return False


def wait(t: float):
    time.sleep(2)
    t -= 2
    while t > 0:
        print(".", end="", flush=True)
        time.sleep(2)
        t -= 2
    print()

def get_max_mbz(mapping: list[str], mbz_list: list[int], run_dir: Path):
    # mapping maps the index of mbz_list to the run_id
    assert len(mbz_list) == 2, "TODO: support more than 2 mbz"
    max_alloc_mem = [math.inf for _ in range(len(mbz_list))]
    device_mem_avail = 96 * 1024 * 0.95 # 96 GB
    
    for i, mbz in enumerate(mbz_list):
        print(f"Checking mbz: {mbz}")
        path = run_dir/mapping[i]
        if not path.is_dir():
            print(f"Path {path} does not exist")
            continue
        with open(path/"status.txt") as f:
            status = RunStatus[f.read().strip()]
        if not (status == RunStatus.completed or status == RunStatus.failed):
            print(f"Run {path} is not completed or failed")
            continue
        # TODO: some case crashed after all iterations.
        log_files = [file for file in (path/"logs").iterdir() if file.suffix == ".log"]
        if not len(log_files) == 1:
            print(f"Log files {log_files} does not exist or more than one")
            continue
        log_file = log_files[0]
        pattern = re.compile(r"Peak reserved: (\d+)")
        finished_pattern = re.compile(r"wandb: Waiting for W&B process to finish")
        max_alloc_history = []
        valid_run = False
        line_no = 0
        with open(log_file, 'rb') as f:
            for line in f:
                line_no += 1
                try:
                    decoded_line = line.decode('utf-8', errors='ignore')
                    match = pattern.search(decoded_line)
                    if match:
                        max_alloc_history.append(int(match.group(1)))
                        print(f"Found max alloc {max_alloc_history[-1]} in line {line_no}")
                    # Check if actually finished
                    match_finished = finished_pattern.search(decoded_line)
                    if match_finished:
                        valid_run = True
                    
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError in {log_file}, line {line_no}")

        if not valid_run or len(max_alloc_history) == 0:
            print(f"Run {path} is not valid")
            continue
        max_alloc_mem[i] = max(max_alloc_history)
    
    print(f"max_alloc_mem: {max_alloc_mem}")
    if max_alloc_mem[0] == math.inf:
        return 1
    if max_alloc_mem[1] == math.inf:
        return mbz_list[0]
    result = mbz_list[0] + math.floor((device_mem_avail - max_alloc_mem[0]) / (max_alloc_mem[1] - max_alloc_mem[0])) * (mbz_list[1] - mbz_list[0])
    print(f"-----mbz1: {mbz_list[0]}, mbz2: {mbz_list[1]}, max_alloc1: {max_alloc_mem[0]}, max_alloc2: {max_alloc_mem[1]}, max_mbz: {result}")
    return result

def schedule_mbz_adapt_runs(configs: list[RunConfig], gpu_budget: int, gpu_per_node: int,
                  run_dir: Path, testrun_dir: Path, template_dir: Path):
    def cp_config_set_mbz(config: RunConfig, mbz: int=1) -> RunConfig:
        new_config = deepcopy(config)
        new_config.micro_batch_size = mbz
        return new_config
    mbz_list = [1,2]
    mbz_list.sort()
    # remove everything in testrun_dir
    for path in testrun_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
    
    print("Scheduling ", len(mbz_list), '*', len(configs), " testruns:")
    print("-"*15)
    mapping = []
    tot_configs = []
    for mbz in mbz_list:
        test_configs = [cp_config_set_mbz(config, mbz) for config in configs]
        newruns = [Run(config, gpu_budget, gpu_per_node, testrun_dir, template_dir)
                for config in test_configs]
        # store the mapping
        mbz_mapping = [run.run_id for run in newruns]
        mapping.append(mbz_mapping)
        tot_configs += test_configs
        
    schedule_runs(tot_configs, gpu_budget, gpu_per_node, testrun_dir, template_dir, test_mbz_run=True)
    
    # adjust the mbz from testruns
    for i, config in enumerate(configs):
        estimated_max_mbz = get_max_mbz([mapping[idx][i] for idx in range(len(mbz_list))], mbz_list, testrun_dir)
        configs[i].micro_batch_size = estimated_max_mbz
    
    schedule_runs(configs, gpu_budget, gpu_per_node, run_dir, template_dir)
        

def schedule_runs(configs: list[RunConfig], gpu_budget: int, gpu_per_node: int,
                  run_dir: Path, template_dir: Path, test_mbz_run: bool=False):
    # Print info.
    print("Scheduling", len(configs), "runs:")
    print(*configs, sep="\n")
    print("-"*10)
    print()

    # Make sure all of the configurations are valid.
    runs = [Run(config, gpu_budget, gpu_per_node, run_dir, template_dir, test_mbz_run)
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
            print("Can't start this run with current GPU budget, used nodes:",
                  squeue.used_nodes(), "Waiting", end="", flush=True)
            wait(10)
        print("GPU resources available, starting run!")

        # Starting run.
        run.status = RunStatus.running
        with Popen(["sbatch", run.home/"nanotron.sbatch"]) as proc:
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
