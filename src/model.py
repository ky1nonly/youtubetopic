# -*- coding: utf-8 -*-
# src/model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import re
from collections import Counter
import logging

# CUDA 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class ZeroShotCategoryClassifier:
    def __init__(self, categories, model_name="joeddav/xlm-roberta-large-xnli"):
        self.categories = categories
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.classifier = pipeline(
            "zero-shot-classification", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1  # GPU 사용 가능하면 0, 아니면 CPU
        )
        logging.info("ZeroShotCategoryClassifier initialized")
        
    def predict_category(self, text):
        result = self.classifier(text, self.categories, multi_label=False)
        return result['labels'][0]

def cluster_within_category(df, category_col='category', text_col='title', n_clusters=5):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    subtopics = []
    logging.info("Clustering within categories")
    
    for cat in df[category_col].unique():
        cat_df = df[df[category_col] == cat].copy()
        if len(cat_df) == 0:
            continue
        cluster_num = min(len(cat_df), n_clusters) if len(cat_df) >= 2 else 1
        embeddings = model.encode(cat_df[text_col].tolist(), show_progress_bar=False)
        if cluster_num > 1:
            kmeans = KMeans(n_clusters=cluster_num, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
        else:
            clusters = np.zeros(len(cat_df), dtype=int)
        
        cat_df['subtopic_cluster'] = clusters
        subtopics.append(cat_df)
        logging.info(f"Clustering completed for category: {cat}")
    
    if len(subtopics) > 0:
        result_df = pd.concat(subtopics, ignore_index=True)
    else:
        result_df = df
    return result_df

def get_subtopic_names(df, category_col='category', subtopic_col='subtopic_cluster', text_col='title'):
    subtopic_dict = {}
    logging.info("Generating subtopic names")
    for cat in df[category_col].unique():
        cat_df = df[df['category'] == cat]
        for cluster_id in cat_df[subtopic_col].unique():
            cluster_df = cat_df[cat_df['subtopic_cluster'] == cluster_id]
            all_text = " ".join(cluster_df[text_col].tolist())
            tokens = re.findall(r"[가-힣\w]+", all_text)
            freq = Counter(tokens)
            if len(freq) > 0:
                subtopic_name = freq.most_common(1)[0][0]
            else:
                subtopic_name = "Misc"
            subtopic_dict[(cat, cluster_id)] = subtopic_name
            logging.info(f"Subtopic for category '{cat}', cluster '{cluster_id}': {subtopic_name}")
    return subtopic_dict

def predict_new_video_category_subtopic(title, channel_name, zs_classifier, df_global, subtopic_names):
    """
    새로운 영상에 대한 카테고리 및 세부 토픽 예측
    """
    combined_text = title + " " + channel_name
    logging.info(f"Combined text for prediction: title='{title}', channel='{channel_name}'")

    predicted_category = zs_classifier.predict_category(combined_text)
    logging.info(f"Predicted category: {predicted_category}")

    cat_df = df_global[df_global['category'] == predicted_category]
    if len(cat_df) == 0:
        logging.warning(f"No data available for predicted category: {predicted_category}")
        return predicted_category, "Misc"

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
    cat_embeddings = model.encode(cat_df['title'].tolist(), show_progress_bar=False)
    new_embedding = model.encode([title], show_progress_bar=False)[0]

    # 클러스터 중심과 비교하여 가장 가까운 클러스터 찾기
    clusters = cat_df['subtopic_cluster'].unique()
    centroids = []
    for c in clusters:
        c_df = cat_df[cat_df['subtopic_cluster'] == c]
        c_emb = model.encode(c_df['title'].tolist(), show_progress_bar=False)
        centroid = np.mean(c_emb, axis=0)
        centroids.append((c, centroid))
    
    # 가장 가까운 centroid 찾기
    min_dist = float('inf')
    closest_cluster = None
    for c, cen in centroids:
        dist = np.linalg.norm(new_embedding - cen)
        if dist < min_dist:
            min_dist = dist
            closest_cluster = c
    
    subtopic_name = subtopic_names.get((predicted_category, closest_cluster), "Misc")
    logging.info(f"Predicted subtopic: {subtopic_name}")
    
    return predicted_category, subtopic_name
