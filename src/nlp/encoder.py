import os
from typing import List

import numpy as np

import os
import ssl
import certifi
from sentence_transformers import SentenceTransformer

from ..utils.config import Config
import pickle


class TextEncoder:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            self.model = SentenceTransformer(self.config.model_name)
            print(f"Loaded model: {self.config.model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """将文本列表编码为向量"""
        if not texts:
            return np.array([])

        print(f"Encoding {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """保存向量到文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, embeddings)
        print(f"Saved embeddings to {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """从文件加载向量"""
        embeddings = np.load(filepath)
        print(f"Loaded embeddings from {filepath}")
        return embeddings