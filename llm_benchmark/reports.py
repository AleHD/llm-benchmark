import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
from google.protobuf.message import DecodeError
from plotly import express as px
from plotly import graph_objects as go

from llm_benchmark.utils import unwrap_dict
from llm_benchmark.status import get_status, RunStatus


def extract_wandb(path: Path) -> pd.DataFrame:
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
            if len(row) == 0:
                continue
            row["step"] = row.pop("_step")
            raw.append(row)
    return pd.DataFrame(raw)


def scrape_from_logs(logpath: Path) -> dict:
    with open(logpath) as f:
        log = "".join(f)

    # Get peak memory.
    peak_mem_MiB = max(map(float, re.findall(".*Peak reserved: (.*)MiB.*", log)))
    peak_mem_GB = (2**20/10**9)*peak_mem_MiB

    return {
        "peak_mem_GB": peak_mem_GB
    }


def get_raw(run_dir: Path, skip_first_steps: bool = True,
            return_failed: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:

    keep_wandb = ["elapsed_time_per_iteration_ms", "tokens_per_sec",
                  "tokens_per_sec_per_gpu", "model_tflops_per_gpu",
                  "hardware_tflops_per_gpu"]
    keep_config = {"parallelism.dp", "parallelism.pp", "parallelism.tp",
                   "parallelism.tp_linear_async_communication", "parallelism.tp_recompute_allgather",
                   "parallelism.recompute_layer",
                   "tokens.batch_accumulation_per_replica", "tokens.micro_batch_size", "tokens.sequence_length",
                   "optimizer.zero_stage"}

    config_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    for path in run_dir.iterdir():
        print("Extracting logs from", path)
        status = get_status(path)
        if status in {RunStatus.success, RunStatus.oom}:
            # Get config and jobid.
            with open(path/"jobid.txt") as f:
                jobid = f.read().strip()
            with open(path/"nanotron_config.yaml") as f:
                config = yaml.safe_load(f)
            config = unwrap_dict(config)
            config_df = pd.concat([config_df, pd.DataFrame([config], index=[jobid])])

            if status is RunStatus.success:
                # Get wandb.
                wandb_path, = (path/"wandb_logs"/"wandb").glob("run-*/*.wandb")
                metrics = extract_wandb(wandb_path)
                metrics = metrics.loc[metrics["step"] > 4, keep_wandb]
                metrics = dict(metrics.mean())

                # Get data from logs (mainly just memory).
                metrics.update(scrape_from_logs(path/"output.log"))
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics], index=[jobid])])
            else:
                full_nans = {key: float("nan") for key in keep_wandb}
                metrics_df = pd.concat([metrics_df, pd.DataFrame([full_nans], index=[jobid])])

    # Clean config_df, removing columns where everyone uses the same config.
    nunique = config_df.nunique(axis=0)
    drop = set(nunique[nunique <= 1].index) - keep_config
    config_df = config_df.drop(columns=drop)
    config_df["gpus"] = config_df["parallelism.dp"]*config_df["parallelism.tp"]*config_df["parallelism.pp"]
    config_df["nodes"] = config_df["gpus"]/4
    metrics_df["oom"] = metrics_df["tokens_per_sec_per_gpu"].isna()
    return config_df, metrics_df


def make_report(run_dir: Path, out: Path, exist_ok: bool):
    # Prepare output dir.
    if out.exists():
        assert exist_ok, f"Out path exists and exist_ok is false: {out}"
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Get configuration columns that are not unique.
    config_df, metrics_df = get_raw(run_dir)
    nunique = config_df.nunique(axis=0)
    nonunique_conf = set(nunique[nunique > 1].index)

    # Merge config and metrics df into a single df.
    metrics_cols = metrics_df.columns
    config_cols = config_df.columns
    df = config_df.join(metrics_df, how="inner")
    df = df.sort_values(by=["gpus"])

    # Add best_in_budget node: true if tok_per_sec_per_gpu is maximized across all runs with the same amount of gpus.
    for gpu in df["gpus"].unique():
        mask = df["gpus"] == gpu
        best = df.loc[mask, "tokens_per_sec_per_gpu"].max()
        df.loc[mask, "best_in_budget"] = df.loc[mask, "tokens_per_sec_per_gpu"] == best

    # Some settings/useful variables for plotting.
    default_colors = np.array(px.colors.qualitative.Plotly)
    highlight = nonunique_conf | {"tokens_per_sec_per_gpu"}
    hovertemplate = "<br>".join(f"<b>{name}</b>: %{{customdata[{i}]}}" if name in highlight
                                else f"{name}: %{{customdata[{i}]}}"
                                for i, name in enumerate(config_cols.tolist() + metrics_cols.tolist()))
    hovertemplate_oom = "<br>".join(f"<b>{name}</b>: %{{customdata[{i}]}}" if name in highlight
                                    else f"{name}: %{{customdata[{i}]}}"
                                    for i, name in enumerate(config_cols.tolist()))


    # Make first plot: x=gpu y=tok/sec/gpu
    print("Making plots...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["gpus"], y=df["tokens_per_sec_per_gpu"], mode="markers",
                             marker={"color": default_colors[df["best_in_budget"].astype(int)]},
                             customdata=df, hovertemplate=hovertemplate, name="sucessful runs"))
    fig.add_trace(go.Scatter(x=df.loc[df["best_in_budget"], "gpus"],
                             y=df.loc[df["best_in_budget"], "tokens_per_sec_per_gpu"],
                             mode="lines", name="best runs"))
    y_oom = []  # because all tok/sec of OOM runs are 0, we will need to separate them by adding a fake offset to each run,
    carry = 0   # depending on the number of gpus it uses.
    prev_gpu = None
    max_tok = df["tokens_per_sec_per_gpu"].max()
    for gpu in df.loc[df["oom"], "gpus"]:
        carry = 0 if prev_gpu != gpu else carry + 1
        prev_gpu = gpu
        y_oom.append(carry*0.01*max_tok)
    fig.add_trace(go.Scatter(x=df.loc[df["oom"], "gpus"], y=y_oom, mode="markers", marker={"symbol": "x"},
                             customdata=df[df["oom"]], hovertemplate=hovertemplate_oom, name="OOM runs"))
    fig.update_layout(xaxis={"title": "gpus", "type": "log"}, yaxis={"title": "tokens_per_sec_per_gpu"},
                      title="Scaling runs")
    fig.write_html(out/"scaling_per_gpu.html")

    # Second plot: same as before but only the best in budget.
    fig = px.line(df[df["best_in_budget"]], x="gpus", y="tokens_per_sec_per_gpu", markers=True)
    fig.update_layout(xaxis={"title": "gpus", "type": "log"}, yaxis={"title": "tokens_per_sec_per_gpu"},
                      title="Scaling runs")
    fig.write_html(out/"scaling_per_gpu_clean.html")

    # Write output csv.
    df.to_csv(out/"raw.csv")

    print("All done!")
