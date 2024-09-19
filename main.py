import yaml
import pandas as pd
from matcher.matcher import SentenceSimilarity
from configs.config_matcher import MatcherConfig
from tqdm import tqdm

tqdm.pandas()


def match_data():
    with open(
        "configs/config_matcher.yaml", "r"
    ) as file:
        yaml_config = yaml.safe_load(file)
        
    config = MatcherConfig(matching_dataset_path=yaml_config['data']['matching_dataset_path'],
                            input_dataset_path=yaml_config['data']['input_dataset_path'],
                            embedder=yaml_config['data']['embedder'],
                            matcher_col_name=yaml_config['data']['matcher_col_name'],
                            input_col_name=yaml_config['data']['input_col_name'],
                            embeddings_save_path=yaml_config['data']['embeddings_save_path'],
                            embeddings_path=yaml_config['data']['embeddings_path'],)

    model = SentenceSimilarity(config=config, train_embeddings=True)
    
    match_dataset = pd.read_csv(config.matching_dataset_path)
    input_dataset = pd.read_csv(config.input_dataset_path)
    
    input_dataset['preds_top_k'] = input_dataset['service_name'].progress_apply(model.get_top_k)
    input_dataset['preds_local_names_top_k'] = input_dataset['preds_top_k'].apply(lambda x: [i['local_name'] for i in x])
    
    input_dataset.to_csv(yaml_config['data']['matched_data_save_path'], index=False)
    
if __name__ == 'main':
    match_data()
    
    
        