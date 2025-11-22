import os
import shutil
import random
from tqdm import tqdm # 用於顯示進度條，可選

# --- 設定參數 ---
DATASET_DIR = "dataset"        # 原始資料集的根目錄
VALIDATION_SPLIT = 0.20        # 驗證集的比例，0.20 代表 20%
CLASS_NAMES = ["nothing", "hand", "cup"] 

# *** 新增: 處理模式設定 ***
# 預設使用 'copy' 複製檔案 (保留原始資料集)
# 可改為 'move' 移動檔案 (會刪除原始資料集的檔案)
ACTION_MODE = 'copy' 
# ------------------

def process_file(source_file, dest_file, mode):
    """根據模式 (copy 或 move) 處理檔案。"""
    if mode == 'copy':
        shutil.copy2(source_file, dest_file) # copy2 保留更多 metadata
    elif mode == 'move':
        shutil.move(source_file, dest_file)
    else:
        raise ValueError(f"不支援的處理模式: {mode}。必須是 'copy' 或 'move'。")


def split_dataset(base_dir, class_names, split_ratio, action_mode):
    """
    將指定目錄下的每個類別資料夾按比例分割為訓練集和驗證集。
    """
    print(f"--- 資料集分割程式啟動 ---")
    print(f"處理模式: **{action_mode.upper()}** (保留原始資料: {action_mode == 'copy'})")
    print(f"驗證集比例: {split_ratio * 100:.2f}%")
    
    # 建立一個新的根輸出目錄，與原始目錄平行
    OUTPUT_DIR = "data_split"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_files_processed = 0
    
    # 遍歷每個類別
    for class_name in class_names:
        source_dir = os.path.join(base_dir, class_name)
        
        # 定義新的輸出目錄名稱
        train_dir_name = f"{class_name}-train"
        validate_dir_name = f"{class_name}-validate"
        
        train_path = os.path.join(OUTPUT_DIR, train_dir_name)
        validate_path = os.path.join(OUTPUT_DIR, validate_dir_name)
        
        # 確保目標目錄存在 (如果已經存在會清空它們，防止重複複製)
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(validate_path):
            shutil.rmtree(validate_path)
            
        os.makedirs(train_path)
        os.makedirs(validate_path)
        
        print(f"\n處理類別: {class_name}")
        
        # 取得所有檔案列表 (只考慮 .jpg 檔案)
        all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        random.shuffle(all_files) # 隨機打亂檔案順序
        
        num_total = len(all_files)
        if num_total == 0:
            print(f"  --> 警告: {source_dir} 中沒有找到圖片，跳過。")
            continue
            
        # 計算分割點
        num_validate = int(num_total * split_ratio)
        num_train = num_total - num_validate
        
        # 分割列表
        validate_files = all_files[:num_validate]
        train_files = all_files[num_validate:]
        
        print(f"  總數: {num_total} | 訓練集: {num_train} | 驗證集: {num_validate}")
        
        # --- 處理訓練集檔案 ---
        print(f"  正在處理訓練集 ({num_train} 檔案)...")
        for file_name in tqdm(train_files, desc=train_dir_name, leave=False):
            source_file = os.path.join(source_dir, file_name)
            dest_file = os.path.join(train_path, file_name)
            process_file(source_file, dest_file, action_mode)
            total_files_processed += 1
            
        # --- 處理驗證集檔案 ---
        print(f"  正在處理驗證集 ({num_validate} 檔案)...")
        for file_name in tqdm(validate_files, desc=validate_dir_name, leave=False):
            source_file = os.path.join(source_dir, file_name)
            dest_file = os.path.join(validate_path, file_name)
            process_file(source_file, dest_file, action_mode)
            total_files_processed += 1

    print("\n--- 分割完成 ---")
    print(f"所有檔案已成功分割並複製到 {OUTPUT_DIR} 目錄下。")
    print(f"總共處理了 {total_files_processed} 個檔案。")

# 執行函式
try:
    # 將 ACTION_MODE 傳入函式中
    split_dataset(DATASET_DIR, CLASS_NAMES, VALIDATION_SPLIT, ACTION_MODE)
except FileNotFoundError as e:
    print(f"\n錯誤: 找不到資料集目錄或其中的某些檔案。請確認路徑是否正確：{e}")
except Exception as e:
    print(f"\n發生未知錯誤: {e}")
