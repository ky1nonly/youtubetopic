# -*- coding: utf-8 -*-
# utils.py
import re

def extract_video_id(title_url: str) -> str:
    # title_url에서 v= 이후부터 &나 문자열 끝까지가 video_id
    pattern = r"v=([^&]+)"
    match = re.search(pattern, title_url)
    if match:
        return match.group(1)
    return None

def clean_title(title: str) -> str:
    # "을(를) 시청했습니다." 제거
    return title.replace("을(를) 시청했습니다.", "").strip()
