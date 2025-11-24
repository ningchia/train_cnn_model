import torch
# import torch.nn as nn
# 移除冗餘的 class MobileNetTransfer 定義，改為導入：
from model_defs import MobileNetTransfer

import os
import sys

# 假設您的 MobileNetTransfer 類別在 2_train_cnn.py 中
# 為了簡單起見，我們在這裡重新定義它，或者將其導入
# 這裡我們需要一個完整的 MobileNetTransfer 結構定義
import torchvision.models as models

# --- 轉換流程 ---
def export_to_onnx(checkpoint_path, onnx_output_path, num_classes=3):
    
    # 1. 初始化模型結構
    # 這裡的 num_classes 必須與您訓練時的類別數 (hand, cup, nothing) 相符
    model = MobileNetTransfer(num_classes=num_classes)
    
    # 2. 載入訓練好的權重 (從檢查點載入 state_dict)
    if not os.path.exists(checkpoint_path):
        print(f"錯誤: 找不到檢查點檔案 {checkpoint_path}。請先運行訓練腳本。")
        sys.exit(1)
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 成功載入模型權重。")
    except Exception as e:
        print(f"錯誤: 載入模型權重失敗。請確認模型結構與檢查點是否匹配。\n錯誤訊息: {e}")
        sys.exit(1)

    model.eval() # 設定為評估模式

    # 3. 定義一個輸入 Tensor (模擬 224x224 RGB 圖片輸入)
    # 批次大小 (1) x 通道 (3) x 高度 (224) x 寬度 (224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. 執行 ONNX 轉換
    torch.onnx.export(
        model,               # 訓練好的模型
        dummy_input,         # 模型的虛擬輸入
        onnx_output_path,    # 輸出檔案名稱
        export_params=True,  # 導出訓練好的參數/權重
        opset_version=17,    # ONNX 標準版本
        do_constant_folding=True, # 優化常數折疊
        input_names = ['input'],   # 輸入名稱
        output_names = ['output'], # 輸出名稱
    )

    print(f"\n*** 轉換完成！ONNX 模型已儲存到：{onnx_output_path} ***")
    print("您現在可以將這個 ONNX 檔案上傳到 Netron 進行可視化。")


if __name__ == '__main__':
    CHECKPOINT_FILE = "latest_checkpoint_mobilenet.pth"
    MODEL_SAVE_PATH = "trained_model"
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    
    # 輸出檔案名稱
    onnx_output_path = "mobilenetv2_transfer_model.onnx"
    
    # 假設您有 3 個類別 (hand, cup, nothing)
    export_to_onnx(checkpoint_path, onnx_output_path, num_classes=3)
