import os

import pandas as pd
import re
import string
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 确保已下载NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class DataProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """清理文本数据"""
        if not text:
            return ""

        # 转换为小写
        text = text.lower()

        # 移除URL
        text = re.sub(r'http\S+', '', text)

        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)

        # 移除标点符号和数字
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\d+', ' ', text)

        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_issues(self, raw_issues: List[Dict]) -> pd.DataFrame:
        """预处理issues数据"""
        processed_data = []

        for issue in raw_issues:
            # 组合标题和正文作为问题内容
            title = issue.get('title', '')
            body = issue.get('body', '')
            content = f"{title} {body}"

            # 清理文本
            cleaned_content = self.clean_text(content)

            # 如果清理后内容为空，跳过
            if not cleaned_content.strip():
                continue

            processed_issue = {
                'id': issue['id'],
                'number': issue['number'],
                'title': title,
                'body': body,
                'cleaned_content': cleaned_content,
                'state': issue['state'],
                'created_at': issue['created_at'],
                'updated_at': issue['updated_at'],
                'user_login': issue['user']['login'] if issue.get('user') else 'unknown',
                'comments': issue['comments'],
                'labels': [label['name'] for label in issue.get('labels', [])]
            }
            processed_data.append(processed_issue)

        return pd.DataFrame(processed_data)

    def save_processed_data(self, df: pd.DataFrame, filepath: str):
        """保存处理后的数据"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(df, f)