# -*- coding: utf-8 -*-
# src/data_processing.py

import json
import pandas as pd
from src.utils import clean_title
from config.config import MAX_DATA_SIZE, WATCH_HISTORY_FILE, PREPROCESSED_DATA_FILE
import logging
import os

def load_and_preprocess(json_path: str) -> pd.DataFrame:
    """
    시청 기록 JSON 파일을 전처리하여 DataFrame을 반환
    전처리 과정:
    1. header가 "YouTube"인 데이터만 필터링
    2. 필요한 컬럼(title, channel_name)만 남김
    3. 불필요한 데이터 제거
    4. 최대 8000개 데이터로 제한
    5. title 정제
    """
    logging.info(f"Loading JSON file from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
            raise e

    df = pd.DataFrame(data)
    logging.info(f"Total records loaded: {len(df)}")

    # 1. header가 YouTube인 것만 필터링
    df = df[df['header'] == 'YouTube'].copy()
    logging.info(f"Records after header filtering: {len(df)}")

    # 2. 필요한 컬럼만 남기기: title, channel_name
    def get_channel_info(subtitles):
        if isinstance(subtitles, list) and len(subtitles) > 0:
            first_sub = subtitles[0]
            name = first_sub.get('name', 'Unknown')
            return name
        else:
            return 'Unknown'

    df['channel_name'] = df['subtitles'].apply(get_channel_info)
    df = df[['title', 'channel_name']]
    logging.info("Selected required columns: title, channel_name")

    # 3. 불필요한 데이터 제거
    df = df[df['channel_name'] != 'Unknown']
    df = df[~df['title'].str.startswith("https://")]
    df = df[~df['channel_name'].str.contains("Topic")]
    logging.info(f"Records after filtering: {len(df)}")

    # 4. 데이터 개수 8000개로 제한
    if len(df) > MAX_DATA_SIZE:
        df = df.iloc[:MAX_DATA_SIZE]
        logging.info(f"Data limited to {MAX_DATA_SIZE} records")

    # 5. title 끝에 "을(를) 시청했습니다." 제거
    df['title'] = df['title'].apply(clean_title)
    logging.info("Titles cleaned")

    return df

def save_preprocessed_data(df: pd.DataFrame, filepath: str):
    """전처리된 데이터를 파일로 저장"""
    df.to_pickle(filepath)
    logging.info(f"Preprocessed data saved to {filepath}")

def load_preprocessed_data(filepath: str) -> pd.DataFrame:
    """저장된 전처리 데이터를 파일에서 불러오기"""
    if os.path.exists(filepath):
        try:
            df = pd.read_pickle(filepath)
            logging.info(f"Preprocessed data loaded from {filepath}")
            return df
        except Exception as e:
            logging.error(f"Failed to load preprocessed data: {e}")
            return None
    else:
        logging.warning(f"Preprocessed data file {filepath} does not exist.")
        return None

def get_data():
    """전처리된 데이터를 불러오거나, 없으면 전처리 후 저장하여 반환"""
    df = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    if df is not None:
        return df
    else:
        # 전처리 수행
        df = load_and_preprocess(WATCH_HISTORY_FILE)
        # 전처리된 데이터 저장
        save_preprocessed_data(df, PREPROCESSED_DATA_FILE)
        return df


'''
데이터에서 태그 추출하는 부분 제거


    # video_id 추출
    df['video_id'] = df['titleUrl'].apply(extract_video_id)
    logging.info("Video IDs extracted")


    # 멀티스레딩을 이용하여 tags 추가
    video_ids = df['video_id'].tolist()
    tags_list = []
    logging.info("Fetching tags via YouTube API using multithreading")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_vid = {executor.submit(fetch_video_tags, vid): vid for vid in video_ids if vid}
        total = len(future_to_vid)
        completed = 0
        for future in as_completed(future_to_vid):
            vid = future_to_vid[future]
            try:
                tags = future.result()
                tags_list.append(tags)
            except Exception as e:
                logging.error(f"Error processing video_id {vid}: {e}")
                tags_list.append([])
            completed += 1
            if completed % 100 == 0 or completed == total:
                logging.info(f"Fetched tags for {completed}/{total} videos")
    
    df['tags'] = tags_list
    logging.info("Tags fetching completed")
    '''



    