from enum import Enum
import shutil
from subprocess import Popen
from typing import Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape
from omegaconf import DictConfig
from pathlib import Path
import time
import yaml
from llm_benchmark import squeue

class RunStatus(Enum):
    init = "init"
    running = "running"
    failed = "failed"
    completed = "completed"
    oom = "oom"

class Run:
    def __init__(self, config: Dict):
        self.config = config
        self.framework = config["framework"]["framework_name"]
        assert self.framework in ["megatron"], f"Framework {self.framework} not supported"
        
        self.n_proc_per_node = config["train_settings"]["gpus_per_node"]
        self.n_nodes = config["train_settings"]["num_nodes"]
        self.n_gpus = self.n_proc_per_node * self.n_nodes

        # Try to get id from identical previous run.
        run_dir = Path(config["logs_dir"])
        run_id = None
        for path in run_dir.iterdir():
            if not path.is_dir():
                continue
            try:
                with open(path/"config.yaml") as f:
                    other_config = yaml.safe_load(f)
            except FileNotFoundError:
                continue
            
            if config["framework"] == other_config["framework"]:  # Other config matches.
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
        template_dir = Path(config["template_dir"])
        if status == RunStatus.init:
            # Write status.
            with open(self.home/"status.txt", "w+") as f:
                print(status.value, file=f)
            # Write config toml.
            with open(self.home/"config.yaml", "w+") as f:
                yaml.dump(config, f)
            # Fill sbatch template.
            sbatch = self._get_sbatch(template_dir)
            with open(self.home/f"{self.framework}.sbatch", "w+") as f:
                print(sbatch, file=f)

    def __lt__(self, other: "Run") -> bool:
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
        template = env.get_template(f"{self.framework}.sbatch")
        if self.framework == "megatron":
            return template.render(
                run_id=self.run_id,
                run_root=self.home,
                n_nodes=self.n_nodes,
                n_proc_per_node=self.n_proc_per_node,
                max_time=str(self.config["cluster"]["max_minutes_per_run"]) + ":00",
                hf_home=self.config["HF_HOME"],
                wandb_api_key=self.config["WANDB_API_KEY"],
                container_toml=self.config["cluster"]["container"],
                megatron_path=self.config["framework"]["FRAMEWORK_HOME"],
                megatron_args=self.get_megatron_args(self.config["framework"]["framework_setting"]),
            )
        else: 
            raise NotImplementedError()

    def get_megatron_args(self, config: Dict) -> str:
        args_list = []
        for k, v in config.items():
            k: str = k.replace("_", "-")
            if isinstance(v, dict):
                args_list.append(self.get_megatron_args(v))
            else:
                if isinstance(v, bool) and v:
                    args_list.append(f"--{k}")
                else:
                    assert v is not None, f"Value for {k} is None"
                    args_list.append(f"--{k} {v}")
        return " ".join(args_list)

def is_oom(home: Path) -> bool:
    # Check log, if "OutOfMemoryError" is mentioned, then we set OOM as status.
    if not (home/"logs").exists():
        return False
    logs = list((home/"logs").glob("*.err"))
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


def schedule_runs(configs: list[Dict], gpu_budget: int):
    # Print info.
    print("Scheduling", len(configs), "runs:")
    print(*configs, sep="\n")
    print("-"*10)
    print()

    # Make sure all of the configurations are valid.
    runs = [Run(config) for config in configs]

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
        user = run.config["cluster"]["user"]
        while squeue.used_nodes(user) + run.n_nodes > gpu_budget//run.n_proc_per_node:
            current_runs = ",".join(str(run.run_id) for run in runs
                                    if run.status == RunStatus.running)
            print("Can't start this run with current GPU budget, used nodes:",
                  squeue.used_nodes(), "Runs:", current_runs, "Waiting",
                  end="", flush=True)
            wait(10)
        print("GPU resources available, starting run!")

        # Starting run.
        run.status = RunStatus.running
        framework = run.config["framework"]["framework_name"]
        with Popen(["sbatch", f"{framework}.sbatch"], cwd=run.home) as proc:
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
