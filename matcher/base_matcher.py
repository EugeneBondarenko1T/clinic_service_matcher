from abc import ABC, abstractmethod


class BaseService(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def encode_samples(self, texts):
        pass

    @abstractmethod
    def get_similarity_scores(self, embeddings_input, top_k):
        pass

    @abstractmethod
    def get_top_k(self, embeddings_input, top_k):
        pass

