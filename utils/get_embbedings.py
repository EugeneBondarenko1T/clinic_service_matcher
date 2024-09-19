"""Этот скрипт содержит класс для получения и сохранения эмбедингов локальных услуг, а также его подргузку
"""

import pickle
import pandas as pd


class ManagerEmbbeding:
    """
    Класс отвечающий за получение эмбедингов, последующее их сохрание и загрузку.

    Attributes_
        embbeder: Эмббедер, используемый для получения эмббедингов
        data (pd.DataFrame): Датасет, содержащий данные, для которых необходимо создать эмбеддинги
        name_col (str): Название столбца в датасете, содержащего текстовые данные для кодирования
        save_path (str): Путь для сохранения эмбеддингов по пути
    """
    def __init__(self, embbeder, data: pd.DataFrame, name_col: str, save_path: str):
        self.embbeder = embbeder
        self.data = data
        self.name_col = name_col
        self.save_path = save_path


    def get_embedings_for_data(self):
        """
        Генерирует эмбеддинги для данных и сохраняет их на диск.

        Использует колонку с текстом из датасета, чтобы сгенерировать эмбеддинги с помощью
        модели embbeder, затем сохраняет эмбеддинги в файл по указанному пути save_path.

        Returns_
            None
        """

        embeddings  = self.embedder.encode(sentences=self.data[self.name_col])

        with open(self.save_path, 'wb') as f:
            pickle.dump(embeddings, f)
    
    
    @staticmethod
    def load_embeddings(embeddings_path: str):
        """
        Загружает эмбеддинги из файла.

        Args_
            embeddings_path (str): Путь к файлу с сохраненными эмбеддингами

        Returns_
            embeddings: Загруженные эмбеддинги
        """
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        return embeddings