# -*- coding: utf-8 -*-
# src/youtube_api.py

import requests
from config.config import YOUTUBE_API_KEY
import logging

'''
태그를 저장하는 부분 삭제

def get_video_tags(video_id: str) -> list:
    """
    video_id를 입력받아 tags를 반환
    """
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            snippet = item.get("snippet", {})
            tags = snippet.get("tags", [])
            if tags:
                return tags
            else:
                
                return []  # 빈 리스트 반환 (빈 태그가 있을 수 있음을 처리)
    logging.error(f"Failed to fetch tags for video_id {video_id}, Status Code: {response.status_code}")
    return []
'''
def get_video_info(video_id: str):
    """
    video_id를 입력받아 title과 channel_name을 반환
    """
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={YOUTUBE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "items" in data and len(data["items"]) > 0:
            item = data["items"][0]
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            channel_name = snippet.get("channelTitle", "")
            return title, channel_name
    logging.error(f"Failed to fetch info for video_id {video_id}, Status Code: {response.status_code}")
    return None, None