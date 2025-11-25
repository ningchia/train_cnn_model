import torch.nn as nn
import torchvision.models as models

# --- 1. 模型定義：CleanCNN (已修正結構 - 移除冗餘 MaxPool) ---
class CleanCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CleanCNN, self).__init__()
        FINAL_CHANNELS = 64
        self.model = nn.Sequential(
            # 第一組 Conv + BN + ReLU
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            # 第二組 Conv + BN + ReLU
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            # 第三組 Conv + BN + ReLU
            nn.Conv2d(32, FINAL_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(FINAL_CHANNELS), 
            nn.ReLU(), 
            # nn.MaxPool2d(2), # <--- 已移除這個冗餘層
            
            nn.AdaptiveAvgPool2d(1), # <--- 直接連接到 Global Average Pool
            
            nn.Flatten(),
            nn.Linear(FINAL_CHANNELS, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- 2. 模型定義：MobileNetV2 遷移學習模型 ---
class MobileNetTransfer(nn.Module):
    def __init__(self, num_classes, use_pretrained=True): # <--- 新增參數
        super(MobileNetTransfer, self).__init__()
        
        # 根據參數決定是否載入預訓練權重
        if use_pretrained:
            # 載入預訓練的 MobileNetV2，使用 DEFAULT 權重 (推薦的最佳版本)
            weights_config = models.MobileNet_V2_Weights.DEFAULT
            print("MobileNetTransfer: 使用預訓練權重進行初始化。")
        else:
            # 只載入模型結構，權重將使用 PyTorch 的預設隨機初始化
            weights_config = None
            print("MobileNetTransfer: 只載入模型結構，權重將隨機初始化。")
        
        # 初始化基礎模型
        self.base_model = models.mobilenet_v2(weights=weights_config)
        
        # 凍結所有基礎特徵提取層的權重 (只有在載入預訓練權重時才做，否則凍結沒有意義)
        if use_pretrained:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # 替換分類器頭部 (Classifier Head)
        # MobileNetV2 的分類器是 self.base_model.classifier
        # 獲取原始分類器輸入的特徵數
        num_ftrs = self.base_model.classifier[-1].in_features
        
        # 建立新的分類器，只有這個分類器會被訓練
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2), 
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
