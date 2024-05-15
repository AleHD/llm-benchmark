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
    num_micro_batch: int = 16
    # batch_size: int = 256  # Roughly 1M tokens with 4096 seq len.
    micro_batch_size: int = 1

    def __post_init__(self):
        assert self.tp > 0
        assert self.dp > 0
        assert self.pp > 0
        assert self.sequence_length > 0
        assert self.num_micro_batch > 0
        # assert self.batch_size > 0
        assert self.micro_batch_size > 0
        
        if self.pp > 1:
            # if pp enabled, ensure num_micro_batch is a multiple (>=2) of pp to avoid large bubble.
            if self.num_micro_batch < 2 * self.pp:
                self.num_micro_batch = 2 * self.pp
                print(f'Warning: PP enabled, num_micro_batch adjusted to {self.num_micro_batch}')
            self.batch_size = self.micro_batch_size * self.dp * self.num_micro_batch

    def asdict(self) -> dict[str, str | int]:
        result = asdict(self)
        result["model"] = result["model"].value
        return result


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
