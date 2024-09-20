# LLM Benchmark

Easily benchmark multiple configurations of LLMs using nanotron.
This codebase also allows easlily plotting the results for quick and interactive analyses.
Example:
![scaling](media/scaling.png)

## Setup

First clone this repository:
```
git clone https://github.com/AleHD/llm-benchmark.git
```

Then, download the [swiss-ai/pretrain](https://github.com/swiss-ai/pretrain) repository to get the launcher.
The latest commit tested to work with this version is `cbbf8b0ff339274cb7fe0b2731feefb51b471f1d`.
```
git clone https://github.com/swiss-ai/pretrain.git
```

The recommended setup to install the external libraries for this project is using [poetry](https://python-poetry.org/):
1. First [install poetry](https://python-poetry.org/docs/):
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```
1. Then, you are able to install the dependencies by running the following on the root of this repository:
   ```
   poetry install
   ```

## Usage

The configurations you want to test are specified in a `.toml` file.
See `configs/llama3_8b.toml` for an example.
The expected contents of such file consist on two keys:
- `defaults: Optional[dict[str, str]]`.
   An optional dictionary that contains the attributes that are defaults for all the configurations specified next.
- `configs: list[dict[str, str]]`.
   A list of configurations to test.
   Each dictionary corresponds to a different configuration to try on a run.
   You can also override the keys set in the default values.

Note that all configurations have a key type of string.
This means that you will need to specify properties quoting (e.g. "nanotron.parallelism.dp" instead of nanotron.parallelism.dp).
You are able to set any property that the `swiss-ai/pretrain/launcher.py` is able to handle.

Once you have the configurations to test, simply run the main file.
For instance:
```
poetry run python main.py --run-dir=runs/llama3_8b run configs/llama3_8b.toml
```

This will schedule all runs you specified.
Once all runs are finished (use `squeue --me`), you are able to run the plotting tool:
```
poetry run python main.py --run-dir=runs/llama3_8b/ analyze reports/llama3_8b
```

Look at `reports/llama3_8b/scaling_per_gpu.html` for the plots.
