from dataclasses import dataclass, field

@dataclass
class MatcherConfig:
    dataset_path: str = field(default="data/data.csv")
    embedder: str = field(default="deepvk/USER-bge-m3")
    matcher_col_name: str = field(default="local_name")