from abc import abstractmethod, ABC


class LLMModel(ABC):
    @abstractmethod
    def chat(self, prompt):
        pass

    @abstractmethod
    def get_embeddings(self, texts, model="text-embedding-ada-002", dimensions=None):
        pass
