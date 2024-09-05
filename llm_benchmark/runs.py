import re
import os
import shutil
import time
from pathlib import Path
from subprocess import Popen, PIPE
from tempfile import TemporaryDirectory
from typing import Optional

import toml
import yaml

from llm_benchmark.status import RunStatus, get_status
from llm_benchmark.utils import with_context


def schedule_run(config: dict[str, str], logs_root: Path, project_name: str,
                 launcher: Path) -> tuple[str, Path]:

    pwd = Path(__file__).parent.parent.absolute()

    defaults = {
        "run.paths.nanotron_logs": str(logs_root),
        "run.slurm.time": "00:30:00",
        "run.paths.nanotron_src": "/users/ahernnde/project/repos/nanotron-swissai",
        "run.paths.additional_setup": str(pwd/"templates"/"additional_setup.sh"),
        "run.env.NCCL_DEBUG": "VERSION",
        "+run.env.http_proxy": "http://proxy.cscs.ch:8080",
        "+run.env.https_proxy": "http://proxy.cscs.ch:8080",
        "run.container.environment": "nanotron-swissai",
        "nanotron.checkpoints.checkpoints_path": str(Path(os.environ.get("SCRATCH", pwd))/".devnull"),
        "nanotron.tokens.train_steps": "15",
        "nanotron.general.project": project_name,
        "nanotron.general.run": "benchmark",
    }

    config = {**config}
    for key, val in defaults.items():
        if key not in config:
            config[key] = val

    cmd = ["python", str(launcher)] + [f"{key}={value}" for key, value in config.items()]

    with Popen(cmd, stdout=PIPE, stderr=PIPE, text=True) as proc:
        assert proc.wait() == 0, "".join(proc.stderr)
        jobid = None
        run_dir = None
        for line in proc.stdout:
            if (rmatch := re.match("^Submitted batch job ([0-9]+)$", line.strip())):
                jobid = rmatch.group(1)
            elif (rmatch := re.match("^log folder created: (.+)$", line.strip())):
                run_dir = rmatch.group(1)
    assert jobid is not None
    assert run_dir is not None
    return jobid, Path(run_dir)


def handle_maybe_schedulable_run(config: dict[str, str], logs_root: Path,
                                 project_name: str, launcher: Path):
    # In order to schedule runs in a smart way we will perform the following steps for each run:
    # 1. Run the launcher with the selected config, but using a temporary directory as output.
    #    Immediately `scancel` the launched job. This is done to get the nanotron config generated.
    # 2. Try to match the nanotron config with any other run in the `run_dir` directory.
    #    If there are no matches, then we can run the launcher for real this time.
    # 3. If there are any matches we will need to determine the status of the match.
    #    It will either be crashed, OOM'd, running or succeeded.
    #    If it succeeded, OOM'd, running then we ignore simply ignore the config as it was done before.
    # 4. If the match crashed before, then we will `rm -R` the old match and schedule again the job using the launcher.

    # First, we launch again with temporary directory.
    print("Attempting fake launch...")
    with TemporaryDirectory() as temp_dir:
        job_id, run_dir = schedule_run(config, Path(temp_dir), project_name, launcher)
        with Popen(["scancel", job_id], stdout=PIPE, stderr=PIPE) as proc:
            assert proc.wait() == 0, "".join(proc.stderr)
        with open(run_dir/"nanotron_config.yaml") as f:
            nano_config = yaml.safe_load(f)

    # Now we try to get matches.
    print("Determining if there are matches of config supplied...")
    matches = [run_dir for run_dir in (logs_root/project_name).iterdir()
               if nano_config == with_context(open(run_dir/"nanotron_config.yaml"),
                                              lambda f: yaml.safe_load(f))]
    assert len(matches) <= 1

    if len(matches) == 1:
        run_dir, = matches
        print("Found one match:", run_dir, "now determining status")
        status = get_status(run_dir)
        if status in {RunStatus.success, RunStatus.running, RunStatus.oom}:
            print(f"Match status is {status}, therefore we won't schedule this config anymore")
            return
        assert status is RunStatus.failure
        print(f"Match status is {status}, therefore we will remove the old run_dir and schedule the config again")
        shutil.rmtree(run_dir)
    else:
        print("No matches found, scheduling the job now!")
    jobid, run_dir = schedule_run(config, logs_root, project_name, launcher)
    with open(run_dir/"jobid.txt", "w+") as f:
        print(jobid, file=f)
    print("Scheduled jobid", jobid, "with directory", run_dir)
    time.sleep(1)  # to make sure the directories have different name.


def schedule_runs(run_dir: Path, run_file: Path, launcher: Path):
    # Get run_dir root, as the launcher will automatically create a new directory under `root/project_name`.
    # We assume the user of llm_benchmark wants to have all runs directly under `run_dir`.
    project_name = run_dir.name
    logs_root = run_dir.parent
    run_dir.mkdir(exist_ok=True)

    # Now we get all the configurations.
    contents = toml.load(run_file)
    configs = contents["configs"]
    print("Run file has", len(configs), "configurations")
    if (defaults := contents.get("defaults")) is not None:
        print("defaults supplied")
        for config in configs:
            for key, value in defaults.items():
                if key not in config:
                    config[key] = value
    print("Starting scheduling of each config")
    print("-"*20)

    # Run quick check of all configs.
    for config in configs:
        for key, value in config.items():
            assert isinstance(key, str)
            assert key != "run.paths.additional_setup", "Overriding additional_setup not yet supported"
            assert key != "run.paths.nanotron_logs", "Don't specify the nanotron_logs explicitly"
            assert key != "nanotron.checkpoints.checkpoints_path", "Don't specify the checkpoint path explicitly"
            if not isinstance(value, str):
                raise ValueError(f"Config file badly specified. All keys and values should "
                                 f"be strings, but found a {type(value)} under the key {key}. "
                                 f'Make sure to specify the keys as "a.b.c" = "d" in the toml '
                                 f'Instead of a.b.c = "d" to handle "a.b.c" as a string instead '
                                 f"of as a nested dictionary")

    for i, config in enumerate(configs):
        print("Analysing", config)
        handle_maybe_schedulable_run(config, logs_root, project_name, launcher)
        print("---")
        print()
    print("All planned runs have now been scheduled!")


def show_status(run_dir: Path):
    print("Inspecting directory", run_dir)
    for path in run_dir.iterdir():
        print("Status of", path)
        print(get_status(path))
        print("---")
    print("That's all folks!")
