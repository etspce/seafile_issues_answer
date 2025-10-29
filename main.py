import os
import sys
from src.utils.config import Config
# from src.utils.logger import setup_logger
from src.crawler.github_crawler import GitHubCrawler
from src.crawler.data_processor import DataProcessor
from src.nlp.encoder import TextEncoder
from src.nlp.cluster import IssueClusterer
from src.search.semantic_search import SemanticSearch

# logger = setup_logger(__name__)


class SeafileQAKnowledgeBase:
    def __init__(self, config_path: str = "configs/settings.yaml"):
        self.config = Config(config_path)
        self.df = None
        self.embeddings = None
        self.cluster_labels = None

    def run_pipeline(self):
        """运行完整的数据处理管道"""
        # logger.info("Starting Seafile QA Knowledge Base pipeline")

        # 1. 抓取数据
        crawler = GitHubCrawler(self.config)
        if not os.path.exists(self.config.raw_data_path):
            raw_issues = crawler.run()
        else:
            print("Raw data already exists, skipping crawling")
            import json
            with open(self.config.raw_data_path, 'r', encoding='utf-8') as f:
                raw_issues = json.load(f)

        # 2. 预处理数据
        processor = DataProcessor()
        if not os.path.exists(self.config.processed_data_path):
            self.df = processor.preprocess_issues(raw_issues)
            processor.save_processed_data(self.df, self.config.processed_data_path)
        else:
            import pickle
            with open(self.config.processed_data_path, 'rb') as f:
                self.df = pickle.load(f)

        print(f"Processed {len(self.df)} issues")

        # 3. 文本向量化
        encoder = TextEncoder(self.config)
        if not os.path.exists(self.config.embeddings_path):
            texts = self.df['cleaned_content'].tolist()
            self.embeddings = encoder.encode_texts(texts)
            encoder.save_embeddings(self.embeddings, self.config.embeddings_path)
        else:
            self.embeddings = encoder.load_embeddings(self.config.embeddings_path)

        # 4. 聚类分析
        clusterer = IssueClusterer(self.config)
        self.cluster_labels = clusterer.perform_clustering(self.embeddings)
        self.df = clusterer.add_clusters_to_data(self.df, self.cluster_labels)

        print("Pipeline completed successfully")
        return self.df, self.embeddings

    def start_search_interface(self):
        """启动搜索界面"""
        if self.df is None or self.embeddings is None:
            raise ValueError("Please run the pipeline first")

        encoder = TextEncoder(self.config)
        search_engine = SemanticSearch(self.config, self.df, self.embeddings)
        search_engine.set_encoder(encoder)

        print("Search interface is ready")
        return search_engine


def main(query_problem):
    """主函数"""
    kb = SeafileQAKnowledgeBase()

    # 运行处理管道
    df, embeddings = kb.run_pipeline()

    # 启动搜索接口
    search_engine = kb.start_search_interface()

    # 示例搜索
    # sample_queries = ["How to setup Seafile with MySQL?", "File synchronization issues", "Docker deployment problems",
    #                   query_problem]

    # for query in sample_queries:
    # print(f"\n=== Search Results for: '{query_problem}' ===")
    print(f"\n=== 问题: '{query_problem}' 的搜索结果为===")
    results = search_engine.search(query_problem, top_k=5)
    for i, result in enumerate(results, 1):
        print(
            # f"{i}. [Cluster {result['cluster']}] [{result['similarity_score']:.4f}] #{result['issue_number']}: {result['title']}")
            f"{i}. #{result['issue_number']}: {result['title']}")


if __name__ == "__main__":
    query = input('请输入你想要问的问题: ')
    main(query)