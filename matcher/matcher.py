"""Этот скрипт содержит класс для решения задачи метчинга при помощи косинусного расстояния
с использованием эмбедера deepvk/USER-bge-m3.
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from configs.config_matcher import MatcherConfig
from utils.get_embedings import load_embeddings


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


class SentenceSimilarity:
    """
    Класс для вычисления косинусной близости и получения топ-5 категорий.

    Attributes:
        embedder (SentenceTransformer): Модель для кодирования предложений
        config (MatcherConfig): Объект конфигурации, содержащий параметры для решения задачи мэтчинга
        local_name_embedings (torch.Tensor): Эмбеддинги для локальных названий услуг
    """
    def __init__(self, config: MatcherConfig):
        """
        Инициализация класса SentenceSimilarity.

        Args:
            config (MatcherConfig): Объект конфигурации, содержащий параметры для решения задачи мэтчинга
        """
        self.embedder = SentenceTransformer(config.embedder)
        self.config = config
        self.local_name_embedings = torch.tensor(load_embeddings(config.embeddings_path)).to("cuda:0")


    def encode_samples(self, texts: list):
        """
        Кодирование входных текстов.

        Args_
            texts (list): Список предложений для кодирования

        Returns_
            torch.Tensor: Тензор с эмбеддингами предложений
        """
        return torch.tensor(self.embedder.encode(sentences=texts, convert_to_tensor=True)).clone().detach()


    def get_similarity_scores(self, embeddings_input, top_k: int):
        """
        Вычисление косинусного расстояния между входным текстов и локальными услугами.

        Args_
            embeddings_input (list): Входной текст для получения эмбедингов
            top_k (int): Количество top_k

        Returns_
            tuple: Кортеж из оценок схожести (scores) и индексов (indices) схожих услуг
        """
        similarity_scores = torch.matmul(self.encode_samples(embeddings_input), self.local_name_embedings.T)
        scores, indices = torch.topk(similarity_scores, k=top_k)

        return scores, indices


    def get_top_k(self, embeddings_input: str, top_k: int) -> list:
        """
        Возвращает top_k схожих услуг для входного предложения.

        Args_
            embeddings_input (str): Входное предложение для поиска схожих услуг
            top_k (int): Количество top_k результатов для возврата.

        Returns_
            list: Список словарей, где каждый словарь содержит название услуги и ее оценку схожести
        """
        data = load_data(self.config.dataset_path)
        scores, indices = self.get_similarity_scores(embeddings_input, top_k)

        scores = scores.cpu().numpy()
        indices = indices.cpu().numpy()

        top_k_dict = []

        for idx, score in zip(indices, scores):
            top_k_dict.append({"local_name": data[self.config.matcher_col_name].iloc[idx], "score": score})

        return top_k_dict