import json
from dataclasses import dataclass
from typing import Optional
from subprocess import Popen, PIPE


@dataclass
class Job:
    job_id: str
    name: str
    user: str
    nodes: int
    state: str


def get_jobs(only_running: bool = True) -> list[Job]:
    with Popen(["squeue", "--json"], stdout=PIPE, text=True) as proc:
        json_txt = "".join(proc.stdout)
        proc.wait()
        jobs_raw = json.loads(json_txt)
    return [Job(job["job_id"], job["name"], job["user_name"],
                job["node_count"]["number"], job["job_state"])
            for job in jobs_raw["jobs"]
            if not only_running or job["job_state"] == "RUNNING"]

def search(jobid: str) -> Optional[Job]:
    with Popen(["squeue", "-j", jobid, "--json"], stdout=PIPE, stderr=PIPE,
                text=True) as proc:
        json_txt = "".join(proc.stdout)
        if proc.wait() != 0:  # Couldn't find job.
            return None
        job_info, = json.loads(json_txt)["jobs"]
    return Job(jobid, job_info["name"], job_info["user_name"],
               job_info["node_count"]["number"], job_info["job_state"])
    return None


def used_nodes(user: str = "ahernnde") -> int:
    return sum(job.nodes for job in get_jobs() if job.user == user and "nanotron" in job.name)
