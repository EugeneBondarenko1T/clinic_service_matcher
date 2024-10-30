from dataclasses import dataclass, field


@dataclass
class MatcherConfig:
    base_dataset_path: str = field(default="data/data.csv")
    negative_dataset_path_save: str = field(default="data/data_for_sentence_pair_classification/data_sentence_pair_cls")
    train_path_save: str = field(default="data/data_for_sentence_pair_classification")
    valid_path_save: str = field(default="data/data_for_sentence_pair_classification")
    model_checkpoint: str = field(default="roberta-base")
    max_length: int = 128
    num_labels: int = 2
    test_size: float = 0.2
    random_state: int = 42
    question_col_name: str = "service_name"
    context_col_name: str = "local_name"