# -*- coding: utf-8 -*-
# app.py

from flask import Flask, render_template, request
import os
from flask_caching import Cache
import pandas as pd
from config.config import CATEGORY_FILE, WATCH_HISTORY_FILE, PREPROCESSED_DATA_FILE
from src.data_processing import get_data
from src.model import ZeroShotCategoryClassifier, cluster_within_category, get_subtopic_names, predict_new_video_category_subtopic
from src.utils import extract_video_id  
from src.youtube_api import get_video_info  
from flask_socketio import SocketIO, emit
import logging

app = Flask(__name__)
socketio = SocketIO(app)

# 캐싱 설정
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,  # DEBUG로 변경 시 더 상세한 로그 가능
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# 카테고리 로드
with open(CATEGORY_FILE, 'r', encoding='utf-8') as f:
    categories = [line.strip() for line in f if line.strip()]

zs_classifier = ZeroShotCategoryClassifier(categories)
logging.info("ZeroShotCategoryClassifier loaded")

df_global = None
subtopic_map = None

@app.route('/', methods=['GET'])
def index():
    global df_global, subtopic_map
    # 서버 시작 시 자동으로 데이터 로드 및 처리
    if df_global is None:
        try:
            df = get_data()
            logging.info("Data loading and preprocessing completed successfully")
        except Exception as e:
            logging.error(f"Data loading and preprocessing failed: {e}")
            return "Preprocessing failed", 500

        # Zero-Shot Classification
        df['combined_text'] = df['title'] + ' ' + df['channel_name']
        df['category'] = df['combined_text'].apply(lambda text: zs_classifier.predict_category(text))
        logging.info("Zero-Shot Classification completed")

        # 카테고리별 클러스터링
        df = cluster_within_category(df, category_col='category', text_col='title', n_clusters=5)
        logging.info("Clustering completed")

        # 클러스터별 subtopic 이름 결정
        subtopic_names = get_subtopic_names(df, category_col='category', subtopic_col='subtopic_cluster', text_col='title')
        df['subtopic_name'] = df.apply(lambda row: subtopic_names.get((row['category'], row['subtopic_cluster']), "Misc"), axis=1)
        logging.info("Subtopic naming completed")

        # 결과 집계
        summary = {}
        for cat in df['category'].unique():
            cat_df = df[df['category'] == cat]
            for sname in cat_df['subtopic_name'].unique():
                count = len(cat_df[cat_df['subtopic_name'] == sname])
                if cat not in summary:
                    summary[cat] = {}
                summary[cat][sname] = count

        df_global = df
        subtopic_map = subtopic_names

        logging.info("Data processing and summarization completed successfully")

    return render_template('index.html', categories=summary, df_global_ready=True)

@app.route('/predict', methods=['POST'])
def predict():
    global df_global, subtopic_map
    video_url = request.form.get('video_url', '')
    if not video_url:
        logging.error("No video_url provided")
        return "No video_url provided", 400

    # video_id 추출 제거
    # vid = extract_video_id(video_url)
    # if not vid:
    #     logging.error("Invalid video URL")
    #     return "Invalid video URL", 400

    # 새로운 영상의 title과 channel_name 가져오기
    title, channel_name = get_video_info(video_url)  
    if not title:
        title = "UnknownVideo"
    if not channel_name:
        channel_name = "UnknownChannel"
    combined_text = title + " " + channel_name
    logging.info(f"Fetched video info: title='{title}', channel='{channel_name}'")

    # Zero-Shot Classification
    predicted_category = zs_classifier.predict_category(combined_text)
    logging.info(f"Predicted category: {predicted_category}")

    # 하위 주제 예측
    predicted_subtopic = "Misc"
    if df_global is not None and predicted_category in df_global['category'].unique():
        predicted_category, predicted_subtopic = predict_new_video_category_subtopic(
            title, channel_name, zs_classifier, df_global, subtopic_map
        )
        logging.info(f"Predicted subtopic: {predicted_subtopic}")
    else:
        logging.warning(f"No data available for predicted category: {predicted_category}")

    return render_template('index.html', predicted_category=predicted_category, predicted_subtopic=predicted_subtopic, df_global_ready=True)

def get_video_info_from_url(video_url: str):
    """
    주어진 YouTube 비디오 URL에서 title과 channel_name 가져오기
    """

    from src.youtube_api import get_video_info  
    
    title, channel_name = get_video_info(video_url)
    return title, channel_name

if __name__ == "__main__":
    socketio.run(app, debug=True)
