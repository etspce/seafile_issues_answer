import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
from ..utils.config import Config
import pickle


class IssueClusterer:
    def __init__(self, config: Config):
        self.config = config

    def perform_clustering(self, embeddings: np.ndarray, method: str = None) -> np.ndarray:
        """执行聚类分析"""
        if method is None:
            method = self.config.cluster_method

        if method == "kmeans":
            n_clusters = self.config.n_clusters
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(embeddings)

            # 评估聚类效果
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                print(f"KMeans Silhouette Score: {silhouette_avg:.4f}")

        elif method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(embeddings)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        print(f"Clustering completed. Found {len(set(cluster_labels))} clusters")
        return cluster_labels

    def add_clusters_to_data(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> pd.DataFrame:
        """将聚类结果添加到数据框"""
        df = df.copy()
        df['cluster'] = cluster_labels

        # 统计每个聚类的大小
        cluster_counts = df['cluster'].value_counts().sort_index()
        print("Cluster distribution:")
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} issues")

        return df