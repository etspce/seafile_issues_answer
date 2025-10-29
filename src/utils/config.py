import os
from typing import Dict, Any
import yaml


class Config:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.settings = yaml.safe_load(f)

        # GitHub配置
        self.github_repo = self.settings['github']['repo']
        self.access_token = self.settings['github'].get('access_token')

        # 模型配置
        self.model_name = self.settings['model']['name']
        self.embedding_dim = self.settings['model']['embedding_dim']

        # 聚类配置
        self.cluster_method = self.settings['cluster']['method']
        self.n_clusters = self.settings['cluster']['n_clusters']

        # 数据路径
        self.raw_data_path = self.settings['paths']['raw_data']
        self.processed_data_path = self.settings['paths']['processed_data']
        self.embeddings_path = self.settings['paths']['embeddings']

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)