import os

import requests
import json
import time
from typing import List, Dict, Optional
from src.utils.config import Config


class GitHubCrawler:
    def __init__(self, config: Config):
        self.config = config
        self.base_url = f"http://api.github.com/repos/{config.github_repo}/issues"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Seafile-QA-Knowledge-Base"
        }
        if config.access_token:
            self.headers["Authorization"] = f"token {config.access_token}"

    def fetch_issues(self, state: str = "all", per_page: int = 100) -> List[Dict]:
        """获取GitHub issues"""
        issues = []
        page = 1

        while True:
            params = {
                "state": state, "per_page": per_page, "page": page, "sort": "created", "direction": "desc"
            }

            try:
                response = requests.get(self.base_url, headers=self.headers, params=params, timeout=10, verify=False)
                if response.status_code == 422:
                    print('已经到最后一页。。')
                batch_issues = response.json()

                if not batch_issues:
                    break

                # 过滤出真正的问题（排除pull request）
                real_issues = [issue for issue in batch_issues if "pull_request" not in issue]
                issues.extend(real_issues)

                print(f"Fetched page {page} with {len(real_issues)} issues")

                # 检查是否还有更多页面
                if len(batch_issues) < per_page:
                    break

                page += 1
                time.sleep(1)  # 避免触发GitHub限流

            except requests.exceptions.RequestException as e:
                print(f"Error fetching issues: {e}")
                break

        return issues

    def save_issues(self, issues: List[Dict], filepath: str):
        """保存issues到JSON文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(issues)} issues to {filepath}")

    def run(self):
        """运行爬虫"""
        print(f"Starting to fetch issues from {self.config.github_repo}")
        issues = self.fetch_issues()
        self.save_issues(issues, self.config.raw_data_path)
        return issues