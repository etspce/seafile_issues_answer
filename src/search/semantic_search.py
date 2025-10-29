import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pandas as pd
from ..utils.config import Config


class SemanticSearch:
    def __init__(self, config: Config, df: pd.DataFrame, embeddings: np.ndarray):
        self.config = config
        self.df = df
        self.embeddings = embeddings
        self.encoder = None  # 可以复用之前的编码器

    def set_encoder(self, encoder):
        """设置文本编码器"""
        self.encoder = encoder

    def search(self, query: str, top_k: int = 10, cluster_filter: int = None) -> List[Dict]:
        """语义搜索"""
        if self.encoder is None:
            raise ValueError("Encoder must be set before searching")

        # 编码查询文本
        query_embedding = self.encoder.encode_texts([query])

        # 如果指定了聚类，只在该聚类内搜索
        if cluster_filter is not None:
            cluster_mask = self.df['cluster'] == cluster_filter
            search_embeddings = self.embeddings[cluster_mask]
            search_df = self.df[cluster_mask]
        else:
            search_embeddings = self.embeddings
            search_df = self.df

        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, search_embeddings)[0]

        # 获取最相似的结果
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 只返回有相似度的结果
                issue = search_df.iloc[idx]
                results.append({
                    'issue_number': int(issue['number']),
                    'title': issue['title'],
                    'body_preview': issue['body'][:200] + "..." if len(issue.get('body', '')) > 200 else issue.get(
                        'body', ''),
                    'similarity_score': float(similarities[idx]),
                    'cluster': int(issue['cluster']),
                    'state': issue['state'],
                    'created_at': issue['created_at'],
                    'user': issue['user_login']
                })

        return results

    def search_by_cluster(self, cluster_id: int, top_k: int = 20) -> pd.DataFrame:
        """获取指定聚类中的问题"""
        cluster_issues = self.df[self.df['cluster'] == cluster_id]
        return cluster_issues.head(top_k)