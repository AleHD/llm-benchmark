from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

import toml
import dacite


class Model(Enum):
    llama_2_7b = "llama_2_7b"
    llama_2_13b = "llama_2_13b"
    llama_2_70b = "llama_2_70b"
    llama_3_8b = "llama_3_8b"
    llama_3_70b = "llama_3_70b"


@dataclass
class RunConfig:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    model: Model = Model.llama_3_8b
    sequence_length: int = 4096
    micro_batch_size: int = 1
    batch_accumulation: int = 4
    pp_engine: str = "1f1b"
    recompute_layer: bool = False
    async_tp: bool = False
    recompute_tp: bool = False

    def __post_init__(self):
        assert self.tp > 0
        assert self.dp > 0
        assert self.pp > 0
        assert self.sequence_length > 0
        assert self.batch_accumulation > 0
        assert self.micro_batch_size > 0

    def asdict(self) -> dict[str, str | int]:
        result = asdict(self)
        result["model"] = result["model"].value
        return result

    @property
    def gpu(self) -> int:
        return self.pp*self.dp*self.tp


def from_dict(data_class, data: dict):
    type_hooks = {Model: lambda x: Model[x]}
    dacite_config = dacite.Config(type_hooks=type_hooks)
    return dacite.from_dict(data_class=data_class, data=data, config=dacite_config)


def to_dict(data) -> dict:
    try:
        return data.asdict()
    except AttributeError:
        return asdict(data)


def get_llama_config(model: Model) -> dict:
    return toml.load(Path(__file__).parent/"model_config.toml")[model.value]


def get_llama_tokenizer(model: Model) -> str:
    _, version, size = model.value.split("_")
    if version == "2":
        return f"meta-llama/Llama-2-{size}-hf"
    return f"meta-llama/Meta-Llama-3-{size.upper()}"
