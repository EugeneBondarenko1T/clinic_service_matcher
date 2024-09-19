from dataclasses import dataclass, field

@dataclass
class MatcherConfig:
    matching_dataset_path: str = field(default="data/data_for_matcher.csv")
    input_dataset_path: str = field(default="data/data.csv")
    embedder: str = field(default="deepvk/USER-bge-m3")
    matcher_col_name: str = field(default="local_name")
    input_col_name: str = field(default="service_name")
    embeddings_path: str = field(default="data/embeddings.pickle")
    embeddings_save_path: str = field(default="data/embeddings.pickle")