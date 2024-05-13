import shutil
from pathlib import Path

import numpy as np
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
            pass

        record_type = pb.WhichOneof("record_type")
        if record_type == "history":
            row = {item.key: maybe_num(item.value_json) for item in pb.history.item}
            row["step"] = row.pop("_step")
            raw.append(row)
    return pd.DataFrame(raw)


def get_raw(run_dir: Path, skip_first_steps: bool = True,
            return_failed: bool = False) -> pd.DataFrame:

    df = pd.DataFrame()
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        with open(path/"status.txt") as f:
            status = "".join(f.read().strip())

        if status == RunStatus.completed.value:
            run_config = toml.load(path/"run_config.toml")
            run_config["run_id"] = path.name
            run_config["success"] = True

            # Get wandb raw data.
            wandb_run_dir, = (path/"wandb").glob("run*")
            wandb_file, = (child for child in wandb_run_dir.iterdir()
                           if child.suffix == ".wandb" )
            logs = extract_raw(wandb_file)
            logs = logs.assign(**run_config)
            df = pd.concat([df, logs], ignore_index=True)
        elif status == RunStatus.failed.value and return_failed:
            run_config = toml.load(path/"run_config.toml")
            run_config["success"] = False
            df = pd.concat([df, pd.DataFrame([run_config])], ignore_index=True)

    df["gpus"] = df["tp"]*df["pp"]*df["dp"]
    if skip_first_steps:
        df = df[pd.isna(df["step"]) | (df["step"] > 1)]
    return df


def make_report(run_dir: Path, out: Path, exists_ok: bool):
    if out.exists():
        assert exists_ok, f"Out path exists and exists_ok is false: {out}"
        shutil.rmtree(out)
    out.mkdir()

    df = get_raw(run_dir)

    # Save raw csvs.
    print("Saving raw csv...")
    df.to_csv(out/"raw.csv")
    by = ["run_id", "model", "success", "gpus", "tp", "dp", "pp", "micro_batch_size"]
    mean_df = df.groupby(by).agg("mean").reset_index()
    mean_df.to_csv(out/"mean.csv")

    # Create simply summary.csv where we only list 
    print("Saving summary csv...")
    relevant = ["run_id", "model", "gpus", "tp", "dp", "pp",
                "micro_batch_size", "success", "tokens_per_sec_per_gpu",
                "tokens_per_sec"]
    all_df = get_raw(run_dir, return_failed=True)
    fail_df = all_df.loc[~all_df["success"], relevant]
    relevant_df = pd.concat([mean_df[relevant], fail_df], ignore_index=True)
    relevant_df = relevant_df.sort_values(by=["model", "gpus", "tp", "dp", "pp", "micro_batch_size"])
    relevant_df.to_csv(out/"summary.csv")

    # Matrix of heatmaps, for each model and for each GPU count, a heatmap where
    # x=tp, y=pp and color=mean_tokens_per_second.
    print("Saving heatmap grid plot...")
    models = df["model"].unique().tolist()
    gpus = sorted(df["gpus"].unique().tolist())
    fig = plt.figure(figsize=(4*len(gpus), 4*len(models)))
    for i, model in enumerate(models):
        for j, gpu in enumerate(gpus):
            plt.subplot(len(models), len(gpus), 1 + j + i*len(models))
            plt.title(f"Model={model}. GPU count={gpu}")
            #subdf = mean_df[(mean_df["model"] == model) & (mean_df["gpus"] == gpu)]
            subdf = relevant_df[(relevant_df["model"] == model) & (relevant_df["gpus"] == gpu)]
            subdf = subdf[["tp", "pp", "tokens_per_sec"]]
            subdf = subdf.groupby(["tp", "pp"]).agg("max").reset_index()  # Only show the mbz that maximizes tokens_per_sec.
            subdf.loc[pd.isna(subdf["tokens_per_sec"]), "tokens_per_sec"] = 0.0
            subdf = subdf.pivot(index="tp", columns="pp", values="tokens_per_sec")
            sns.heatmap(subdf, annot=True, cbar=False, ax=plt.gca())
    plt.savefig(out/"heatgrid.pdf")
    plt.close(fig)

    # Simple scaling graph: Each model has a plot, in each plot x=num_gpus, y=tokens_per_second_per_gpu.
    print("Saving scaling graph...")
    subdf = mean_df.groupby(["model", "gpus"]).agg("max").reset_index()
    sns.relplot(data=subdf, x="gpus", y="tokens_per_sec_per_gpu", col="model", kind="line")
    plt.savefig(out/"scaling_per_gpu.pdf")
    # Simple scaling graph: Each model has a plot, in each plot x=num_gpus, y=tokens_per_second.
    sns.relplot(data=subdf, x="gpus", y="tokens_per_sec", col="model", kind="line")
    plt.savefig(out/"scaling.pdf")

