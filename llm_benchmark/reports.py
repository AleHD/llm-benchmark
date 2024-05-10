import shutil
import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
import toml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
from google.protobuf.message import DecodeError
from matplotlib import pyplot as plt

from .runs import RunStatus


def extract_raw(path: Path) -> pd.DataFrame:
    def maybe_num(val: str) -> int | float | str:
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val


    ds = datastore.DataStore()
    ds.open_for_scan(path)
    raw = []
    while (data := ds.scan_record()) is not None:
        pb = wandb_internal_pb2.Record()
        try:
            pb.ParseFromString(data[1])  
        except DecodeError:
            warnings.warn("Couldn't decode the entire wandb file, using incomplete data")

        record_type = pb.WhichOneof("record_type")
        if record_type == "history":
            row = {item.key: maybe_num(item.value_json) for item in pb.history.item}
            row["step"] = row.pop("_step")
            raw.append(row)
    return pd.DataFrame(raw)


def get_raw(run_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame()
    for path in run_dir.iterdir():
        # If not completed run, ignore.
        with open(path/"status.txt") as f:
            status = "".join(f.read().strip())
        if status != RunStatus.completed.value:
            continue
        run_config = toml.load(path/"run_config.toml")
        run_config["run_id"] = path.name

        # Get wandb raw data.
        wandb_run_dir, = (path/"wandb").glob("run*")
        wandb_file, = (child for child in wandb_run_dir.iterdir()
                       if child.suffix == ".wandb" )
        logs = extract_raw(wandb_file)
        logs = logs.assign(**run_config)
        df = pd.concat([df, logs], ignore_index=True)
    return df


def make_report(run_dir: Path, out: Path, exists_ok: bool):
    if out.exists():
        assert exists_ok, f"Out path exists and exists_ok is false: {out}"
        shutil.rmtree(out)
    out.mkdir()

    df = get_raw(run_dir)
    df = df[df["step"] > 1]  # Skip the first two steps as they are generally slower.
    df["gpus"] = df["tp"]*df["pp"]*df["dp"]

    # Save raw csvs.
    df.to_csv(out/"raw.csv")
    mean_df = df.groupby(["run_id", "model"]).agg("mean")
    mean_df.to_csv(out/"mean.csv")

    # Simple scaling graph: Each model has a plot, in each plot x=num_gpus, y=tokens_per_second
    sns.relplot(data=df, x="gpus", y="tokens_per_sec_per_gpu", col="model")
    plt.savefig(out/"scaling.pdf")


