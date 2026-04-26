from dataclasses import dataclass
from typing import Dict, List, Literal
import yaml

ActivationType = Literal['relu', 'sigmoid', 'tanh']

@dataclass
class CommonConfig:
    seed: int = 114514
    gpu: str = '0'
    runs: int = 10

@dataclass
class DatasetConfig:
    layers: List[int]
    acts: List[ActivationType]
    learning_rate: float
    pretrain_learning_rate: float
    lamSC: int
    coeff_reg: float
    max_epoch: int
    max_iter: int
    pre_iter: int

@dataclass
class CompleteConfig:
    common: CommonConfig
    datasets: Dict[str, DatasetConfig]

    def __getattr__(self, name: str) -> DatasetConfig:
        if name in self.datasets:
            return self.datasets[name]
        raise AttributeError(f"No dataset config named {name}")


    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'CompleteConfig':
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        dataset_configs = {
            key: DatasetConfig(**value)
            for key, value in data.items()
            if key != 'common'
        }

        return cls(
            common=CommonConfig(**data['common']),
            datasets=dataset_configs,
        )
