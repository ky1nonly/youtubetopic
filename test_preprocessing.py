# test_preprocessing.py
# 작동 테스트 파일

from src.data_processing import load_and_preprocess, save_preprocessed_data, load_preprocessed_data

def main():
    json_path = "data/watch_history.json"
    preprocessed_path = "data/preprocessed_data.pkl"

    # 전처리 수행
    try:
        df = load_and_preprocess(json_path)
        print(f"Preprocessed DataFrame shape: {df.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    # 전처리된 데이터 저장
    try:
        save_preprocessed_data(df, preprocessed_path)
        print(f"Preprocessed data saved to {preprocessed_path}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")
        return

    # 저장된 데이터 불러오기
    try:
        df_loaded = load_preprocessed_data(preprocessed_path)
        if df_loaded is not None:
            print(f"Loaded DataFrame shape: {df_loaded.shape}")
        else:
            print("Failed to load preprocessed data.")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")

if __name__ == "__main__":
    main()
