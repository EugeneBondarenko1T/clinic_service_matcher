class ParametrsTraining:
    output_dir: str = "./results"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    save_total_limit: int = 2
    logging_dir: str = "./logs"
    logging_steps: int = 100
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100