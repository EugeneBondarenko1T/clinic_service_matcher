from dataclasses import dataclass, field


@dataclass
class ParametersTraining:
    output_dir: str = field(default="./results")
    logging_dir: str = field(default="./logs")
    model_checkpoint_trained_model: str = field(default="results/checkpoint-2176")
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    save_total_limit: int = 2
    logging_steps: int = 200
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    device: str = "cuda:0"
    max_length: int = 128