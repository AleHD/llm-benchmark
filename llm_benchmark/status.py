import json
import warnings
from pathlib import Path
from enum import StrEnum
from subprocess import Popen, PIPE


class RunStatus(StrEnum):
    success = "success"
    running = "running/pending"
    failure = "failure"
    oom = "oom"


def get_status(run_dir: Path) -> RunStatus:
    def is_oom() -> bool:
        if not (run_dir/"output.err").exists():
            return False
        with open(run_dir/"output.err") as f:
            for line in f:
                if "OutOfMemoryError" in line or "CUDA error: out of memory" in line:
                    return True
        return False

    # Check if there is a success cache file.
    if (run_dir/".status.txt").exists():
        with open(run_dir/".status.txt") as f:
            return RunStatus[f.read().strip()]

    # Get jobid
    with open(run_dir/"jobid.txt") as f:
        jobid = f.read().strip()

    # Look for the job with such job_id.
    job_state = None
    with Popen(["squeue", "--jobs", jobid, "--json"], stdout=PIPE, stderr=PIPE, text=True) as proc:
        json_txt = "".join(proc.stdout)
        if proc.wait() == 0:  # If the job is too old squeue won't be able to find it.
            job_info, = json.loads(json_txt)["jobs"]
            job_state = job_info["job_state"]
    if job_state is None:  # i.e. the job is too old, we need to `sacct` instead.
        with Popen(["sacct", "--jobs", jobid, "--json"], stdout=PIPE, stderr=PIPE, text=True) as proc:
            json_txt = "".join(proc.stdout)
            assert proc.wait() == 0, "".join(proc.stderr)
            job_state = json.loads(json_txt)["jobs"][0]["state"]["current"]
    if isinstance(job_state, list):  # I don't know why sometimes the state is returned as list.
        job_state, = job_state
    assert isinstance(job_state, str)

    # Now return the appropriate run status.
    if job_state == "CANCELLED":
        status = RunStatus.failure
    elif job_state == "TIMEOUT":
        warnings.warn(f"Timeout status detected for {run_dir}. Restarting job anyway")
        status = RunStatus.failure
    elif job_state == "COMPLETED":
        status = RunStatus.success
    elif job_state in {"COMPLETING", "PENDING", "RUNNING"}:
        return RunStatus.running
    elif job_state in {"FAILURE", "FAILED"}:  # Determine if it is a misc failure, or an OOM.
        status = RunStatus.oom if is_oom() else RunStatus.failure
    elif job_state == "OUT_OF_MEMORY":
        status = RunStatus.oom
    else:
        raise ValueError(f"Unknown slurm state {job_state}")

    # Write status file and return.
    with open(run_dir/".status.txt", "w+") as f:
        print(status.value, file=f)
    return status
