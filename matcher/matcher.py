"""Этот скрипт содержит класс для решения задачи метчинга при помощи косинусного расстояния
с использованием эмбедера deepvk/USER-bge-m3.
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from configs.config_matcher import MatcherConfig
from matcher.base_matcher import BaseService
from utils.get_embbedings import ManagerEmbedding


class SentenceSimilarity(BaseService):
    """
    Класс для вычисления косинусной близости и получения топ-5 категорий.

    """
    def __init__(self, config: MatcherConfig = None, train_embeddings: bool = False):
        """
        Инициализация класса SentenceSimilarity.

        Args_
            config (MatcherConfig): Объект конфигурации, содержащий параметры для решения задачи мэтчинга
            train_embeddings (bool): Флаг для генерации эмбеддингов
        """
        self.embedder = SentenceTransformer(config.embedder)
        self.config = config
        self.manager = ManagerEmbedding(embedder=self.config.embedder, 
                                        data_path=self.config.matching_dataset_path, 
                                        name_col=self.config.matcher_col_name, 
                                        save_path=self.config.embeddings_save_path)
        if train_embeddings:
            self.manager.get_embedings_for_data()
            
        self.local_name_embedings = torch.tensor(self.manager.load_embeddings(config.embeddings_path)).to("cuda:0")
        self.data = self.load_data(self.config.matching_dataset_path)


    @staticmethod
    def load_data(path):
        """
        Загрузка данных из CSV файла.

        Args_
            path (str): Путь к CSV файлу

        Returns_
            pd.DataFrame: Загруженные данные в виде DataFrame
        """
        data = pd.read_csv(path)

        return data

    def encode_samples(self, texts: list):
        """
        Получение эмбедингов по входному тексту.

        Args_
            texts (list): Список предложений для кодирования

        Returns_
            torch.Tensor: Тензор с эмбеддингами предложений
        """
        return torch.tensor(self.embedder.encode(sentences=texts, convert_to_tensor=True)).clone().detach()


    def get_similarity_scores(self, embeddings_input, top_k: int):
        """
        Вычисление косинусного расстояния между входным текстом и локальными услугами.

        Args_
            embeddings_input (list): Входной текст для получения эмбедингов
            top_k (int): Количество top_k

        Returns_
            tuple: Кортеж из оценок схожести (scores) и индексов (indices) схожих услуг
        """
        similarity_scores = torch.matmul(self.encode_samples(embeddings_input), self.local_name_embedings.T)
        scores, indices = torch.topk(similarity_scores, k=top_k)

        return scores, indices


    def get_top_k(self, input_text: str, top_k: int = 5) -> list:
        """
        Возвращает top_k схожих услуг для входного предложения.

        Args_
            input_text (str): Входное предложение для поиска схожих услуг
            top_k (int): Количество top_k результатов для возврата

        Returns_
            list: Список словарей, где каждый словарь содержит название услуги и ее оценку схожести
        """
        scores, indices = self.get_similarity_scores(input_text, top_k)

        scores = scores.cpu().numpy()
        indices = indices.cpu().numpy()

        top_k_dict = [{"local_name": self.data[self.config.matcher_col_name].iloc[idx], "score": score} for idx, score in zip(indices, scores)]

        return top_k_dict