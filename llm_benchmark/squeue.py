import json
from dataclasses import dataclass
from subprocess import Popen, PIPE


@dataclass
class Job:
    job_id: str
    name: str
    user: str
    nodes: int


def get_jobs() -> list[Job]:
    with Popen(["squeue", "--json"], stdout=PIPE, text=True) as proc:
        json_txt = "".join(proc.stdout)
        proc.wait()
        jobs_raw = json.loads(json_txt)
    return [Job(job["job_id"], job["name"], job["user_name"], job["node_count"]["number"])
            for job in jobs_raw["jobs"] if job["job_state"] == "RUNNING"]


def used_nodes(user: str = "ahernnde", framework: str = "megatron") -> int:
    return sum(job.nodes for job in get_jobs() if job.user == user and framework in job.name)
