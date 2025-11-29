import torch
import torch.nn as nn
# 移除冗餘的 class MobileNetTransfer 定義，改為導入：
from model_defs import MobileNetTransfer

import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models # <--- 新增：導入 torchvision models
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
from PIL import Image
import warnings

# import psutil     # 用於系統資源監控

# 忽略 PIL/Image 庫可能發出的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 配置與參數設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_split"
MODEL_SAVE_PATH = "trained_model"
CHECKPOINT_FILE = "latest_checkpoint_mobilenet.pth" # 更改檢查點檔案名稱
NUM_EPOCHS = 300 
BATCH_SIZE = 32
# 遷移學習時，基礎層使用極小的學習率，或只訓練分類頭部
TRANSFER_LEARNING_LR = 0.0001 
FINE_TUNE_LR = 0.00001          # 較低的學習率用於解凍全部參數時

USE_PRETRAINED = True           # 是否使用預訓練權重. False : 只使用架構，隨機初始化權重.

WANT_REPRODUCEBILITY = False    # 是否要強化訓練結果的可重現性 (Reproducibility)
SEED = 42

# 嘗試導入 pynvml (NVIDIA Management Library Python Bindings)
# 這種紀錄法是在epoch結束的時候紀錄, 當時GPU與CPU都沒有在進行訓練工作所以不準.
'''
try:
    from pynvml import *
    NVML_AVAILABLE = True
except ImportError:
    # print("警告: 未安裝 pynvml 函式庫，無法監控 GPU 資訊。")
    NVML_AVAILABLE = False
    
# 如果使用 CUDA，初始化 NVML
if NVML_AVAILABLE and torch.cuda.is_available():
    try:
        nvmlInit()
    except Exception as e:
        # print(f"警告: NVML 初始化失敗: {e}")
        NVML_AVAILABLE = False

def get_resource_usage(device: torch.device):
    """
    獲取 CPU, 記憶體和 GPU 的使用率。
    """
    metrics = {}

    # --- 1. CPU 和 記憶體 ---
    process = psutil.Process(os.getpid())
    
    # 獲取進程的 CPU 使用率百分比
    # 注意: process.cpu_percent() 需要呼叫兩次來獲取非零值
    # 為簡化，我們只使用總體 CPU 使用率
    metrics['cpu_total_percent'] = psutil.cpu_percent(interval=None) 
    
    # 獲取進程使用的記憶體 (MB)
    metrics['mem_used_mb'] = process.memory_info().rss / (1024 * 1024) 
    
    # --- 2. GPU 監控 (僅限 CUDA) ---
    if device.type == 'cuda' and NVML_AVAILABLE:
        try:
            # 假設使用第一個 GPU (Index 0)
            handle = nvmlDeviceGetHandleByIndex(device.index if device.index is not None else 0)
            
            # 獲取 GPU 使用率 (Compute / Graphics Utilization)
            metrics['gpu_util_percent'] = nvmlDeviceGetUtilizationRates(handle).gpu
            
            # 獲取 GPU 記憶體使用情況
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            metrics['gpu_mem_used_mb'] = mem_info.used / (1024 * 1024)
            metrics['gpu_mem_total_mb'] = mem_info.total / (1024 * 1024)
            
        except Exception as e:
            metrics['gpu_error'] = str(e)
            
    return metrics
'''

# 如果要讓每次訓練結果盡可能一致（即提高可重現性），需要將所有涉及隨機性的組件都鎖定 (Lock Down)。
def set_seed(seed_value=42):
    """ 
    設定所有隨機性的種子，確保結果可重現。 
    """
    print("設定所有隨機性的種子，確保結果可重現。")
    random.seed(seed_value)         # Python 內建的 random 模組
    np.random.seed(seed_value)      # NumPy 模組
    torch.manual_seed(seed_value)   # PyTorch CPU 隨機數生成器
    
    # 僅在 CUDA 可用時設定 GPU 種子和 cuDNN 參數
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)      # 為所有 GPU 設定種子
        
        # 確保 cuDNN 運算是確定性的 (犧牲輕微性能換取精確復現)
        torch.backends.cudnn.deterministic = True   # 強制 cuDNN 使用確定性算法
        torch.backends.cudnn.benchmark = False      # 關閉自動尋優
    
    # 確保 DataLoader 的 Worker 也能有固定的種子
    # 這是 DataLoader 內多進程隨機性的補充設置
    def seed_worker(worker_id):
        worker_seed = seed_value + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
    g = torch.Generator()
    g.manual_seed(seed_value)
    
    return seed_worker, g # 返回 worker_init_fn 和 generator 給 DataLoader 使用

# --- 3. 數據加載：CustomSplitDataset (保持不變) ---
class CustomSplitDataset(Dataset):
    def __init__(self, root_dir, split_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        self.classes = [] 
        
        all_subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_names = set()
        
        # --- nothing-train 擴增邏輯 (優先使用 nothing-train-augmented) ---
        target_dirs = []
        
        if split_type == 'train':
            aug_nothing_dir = 'nothing-train-augmented'
            nothing_aug_path = os.path.join(root_dir, aug_nothing_dir)
            nothing_aug_exists = os.path.isdir(nothing_aug_path)
            
            processed_classes = set()
            
            if nothing_aug_exists:
                # 優先使用 augmented 訓練集
                target_dirs.append(aug_nothing_dir)
                class_names.add('nothing')
                processed_classes.add('nothing')
            
            suffix = f"-{split_type}" 
            for d in all_subdirs:
                if d.endswith(suffix):
                    class_name = d.rsplit('-', 1)[0]
                    
                    if class_name == 'nothing':
                        if 'nothing' not in processed_classes:
                            # nothing-train: 只有在 nothing-train-augmented 不存在時才使用
                            target_dirs.append(d)
                            class_names.add(class_name)
                            processed_classes.add('nothing')
                    else:
                        # hand-train, cup-train 等其他類別，直接加入
                        target_dirs.append(d)
                        class_names.add(class_name)
                        processed_classes.add(class_name)
                        
        else:
            # 處理非 train 集合 (例如 validate)
            suffix = f"-{split_type}" 
            for d in all_subdirs:
                 if d.endswith(suffix):
                      target_dirs.append(d)
                      class_name = d.rsplit('-', 1)[0]
                      class_names.add(class_name)

        # --- 邏輯結束 ---
        
        self.classes = sorted(list(class_names))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"[{split_type.upper()}] 檢測到的類別: {self.classes}")
        
        # 遍歷選定的資料夾來加載樣本
        for d in target_dirs:
            if d == 'nothing-train-augmented':
                 class_name = 'nothing'
            else:
                 class_name = d.rsplit('-', 1)[0]
                 
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                folder_path = os.path.join(root_dir, d)
                
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(folder_path, img_file), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"警告: 無法讀取檔案 {img_path}")
            return self.__getitem__((idx + 1) % len(self)) 

        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

# --- 4. 數據轉換 (Transform)：針對 MobileNetV2 進行調整 ---
# MobileNetV2 標準輸入尺寸為 224x224，並使用 ImageNet 標準化參數
# 訓練集專用轉換 (包含數據擴增)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # <--- 調整為 224x224
    # *** 關鍵新增：隨機水平翻轉 ***
    transforms.RandomHorizontalFlip(), 
    # ********************************
    transforms.ToTensor(), 
    # 使用 ImageNet 標準化的均值和標準差
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# 驗證集/測試集專用轉換 (不包含隨機擴增)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def get_loaders(data_dir, batch_size):
    train_dataset = CustomSplitDataset(data_dir, 'train', train_transform)
    val_dataset = CustomSplitDataset(data_dir, 'validate', val_transform)
   
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         raise ValueError(f"訓練集或驗證集為空。訓練集: {len(train_dataset)}, 驗證集: {len(val_dataset)}")
    
    # 保持 num_workers=4 以加快數據加載速度
    if not WANT_REPRODUCEBILITY:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        # 若要強化訓練結果的可重現性則用下面的設定
        worker_init_fn, generator = set_seed(SEED)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, len(train_dataset.classes)

def calculate_accuracy(loader, model):
    # ... (保持不變)
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# --- 5. 檢查點與凍結層函式 (針對 MobileNet 調整) ---

def save_checkpoint(epoch, model, optimizer, best_acc, path):
    """保存模型、優化器狀態、epoch 和最佳準確度"""
    # ... (保持不變)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    torch.save(checkpoint, path)
    print(f"\n[CHECKPOINT] 狀態已儲存到 {path} (Epoch: {epoch}, Acc: {best_acc:.2f}%)")

def load_checkpoint(path, model, optimizer):
    """載入檢查點並返回起始 epoch 和最佳準確度"""
    # ... (保持不變)
    if not os.path.exists(path):
        return 0, 0.0, False 

    checkpoint = torch.load(path, map_location=device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        print(f"\n[CHECKPOINT] 已載入檢查點，從 Epoch {start_epoch} 恢復訓練 (歷史最佳 Acc: {best_accuracy:.2f}%)")
        return start_epoch, best_accuracy, True
        
    except Exception as e:
        print(f"[警告] 檢查點載入失敗: {e}。將從頭開始訓練。")
        return 0, 0.0, False


def freeze_base_layers(model, freeze=True):
    """
    凍結或解凍 MobileNet 的特徵提取層 (除分類頭部外)。
    """
    for param in model.base_model.features.parameters():
        param.requires_grad = (not freeze)
    
    if freeze:
        print("*** [遷移學習] MobileNetV2 基礎層參數已凍結 (只訓練分類器)。 ***")
    else:
        print("*** [微調階段] MobileNetV2 基礎層參數已解凍 (進入微調)。 ***")


# --- 6. 訓練流程主函式 (優化器調整) ---
def train_model(train_loader, val_loader, model, total_epochs, start_epoch, initial_best_acc):
    
    if USE_PRETRAINED:
        # 決定優化器的學習率
        if start_epoch == 0:
            # 第一階段：只訓練分類頭部 (基礎層已在 __init__ 中凍結)
            lr = TRANSFER_LEARNING_LR
            print(f"初始化學習率 (僅分類頭): {lr}")
        else:
            # 斷點續訓，假設已經在微調階段
            lr = FINE_TUNE_LR
            print(f"續訓學習率 (微調模式): {lr}")
    else:
        lr = TRANSFER_LEARNING_LR
        print(f"從頭訓練, 學習率: {lr}")
        
    # 優化器只追蹤 requires_grad=True 的參數 (即未凍結的分類器)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = initial_best_acc 
    visual_best_acc = best_accuracy
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    
    # --- 儲存邏輯的起始 Epoch (保持不變) ---
    saving_start_epoch = total_epochs - 10 

    print(f"\n--- 開始訓練 (總目標 Epoch: {total_epochs}, 從 Epoch {start_epoch + 1} 開始) ---")
    
    # --- 調整微調邏輯 (可選) ---
    # 如果要引入微調 (Fine-Tuning) 階段，可以在達到一定 Epoch 後解凍所有層
    FINE_TUNE_EPOCH = 100 # 例如在第 100 個 Epoch 之後開始微調
    
    try:
        for epoch in range(start_epoch, total_epochs):
            current_epoch_num = epoch + 1
            
            # --- 微調階段切換 ---
            if current_epoch_num == FINE_TUNE_EPOCH + 1 and current_epoch_num < total_epochs:
                print(f"\n*** [切換] 達到 Epoch {FINE_TUNE_EPOCH}，進入微調 (Fine-Tuning) 階段 ***")
                freeze_base_layers(model, freeze=False) # 解凍基礎層
                # 重設優化器，使用更低的學習率和包含所有參數
                optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
                print(f"*** 優化器學習率已更新為 {FINE_TUNE_LR} ***")
            
            # --- 訓練與驗證階段 (保持不變) ---
            model.train() 
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {current_epoch_num}/{total_epochs}", leave=False)
            # ... (訓練邏輯不變)

            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})

            train_loss = running_loss / len(train_loader.dataset)
            
            model.eval() 
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = calculate_accuracy(val_loader, model)
            
            # --- 儲存檢查點邏輯 ---
            print_message = f"Epoch {current_epoch_num}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            
            if val_accuracy > visual_best_acc:
                visual_best_acc = val_accuracy
                print_message += f" (新歷史最高: {visual_best_acc:.2f}%)"
            
            is_saving_epoch = current_epoch_num >= saving_start_epoch
            
            if is_saving_epoch:
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
                    print_message += f" -> **模型狀態已更新儲存** (目前最佳 Acc: {best_accuracy:.2f}%)"
                else:
                    print_message += f" (儲存區間內，目前最佳: {best_accuracy:.2f}%)"
            
            else:
                print_message += f" (儲存功能關閉，剩餘 {saving_start_epoch - current_epoch_num} 個 Epoch 啟動)"

            print(print_message)

            # 這種紀錄法是在epoch結束的時候紀錄, 當時GPU與CPU都沒有在進行訓練工作所以不準.
            # 範例調用 (在訓練腳本內)
            '''
            if epoch % 5 == 0: # 每 5 個 Epoch 報告一次
                usage = get_resource_usage(device)
                
                print(f"[{time.strftime('%H:%M:%S')}] 資源使用情況:")
                print(f"  CPU 總體使用率: {usage.get('cpu_total_percent', 'N/A'):.2f}%")
                
                if 'gpu_util_percent' in usage:
                    print(f"  GPU 使用率: {usage['gpu_util_percent']}%")
                    print(f"  GPU 記憶體: {usage['gpu_mem_used_mb']:.2f}MB / {usage['gpu_mem_total_mb']:.2f}MB")
                print(f"  程式記憶體 (RSS): {usage['mem_used_mb']:.2f}MB")
            '''
            
    except KeyboardInterrupt:
        print("\n\n*** [使用者中斷] 偵測到 Ctrl+C，提前結束訓練。 ***")
        save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)

    except RuntimeError as e:
        if "DataLoader worker" in str(e):
             print("\n\n*** [DataLoader中斷] 偵測到 DataLoader worker 異常退出。 ***")
             try:
                 save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
                 print(f"-> 成功儲存檢查點，以防數據丟失。")
             except Exception as save_err:
                 print(f"-> 儲存檢查點時發生錯誤: {save_err}")
        else:
            raise e
            
    finally:
        print("-" * 50)
        print(f"訓練流程結束。")
        print(f"整體訓練過程中的最高準確度: {visual_best_acc:.2f}%\n")
        if best_accuracy > 0.0:
            print(f"最終儲存的最佳準確度: {best_accuracy:.2f}%")
            
# --- 7. 執行區塊 ---
if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

    try:
        train_loader, val_loader, num_classes_detected = get_loaders(DATA_DIR, BATCH_SIZE)
        
        print(f"總訓練樣本數: {len(train_loader.dataset)}")
        print(f"總驗證樣本數: {len(val_loader.dataset)}")
        print(f"偵測到類別數量: {num_classes_detected}")
        
        # 步驟 1: 初始化模型
        if USE_PRETRAINED:
            print("使用預訓練權重初始化模型。")
        else:
            print("不使用預訓練權重，模型權重將隨機初始化。")
        model = MobileNetTransfer(num_classes=num_classes_detected, use_pretrained=USE_PRETRAINED).to(device)
        
        # 初始化優化器（用於 load_checkpoint 載入狀態）
        initial_optimizer = optim.Adam(model.parameters(), lr=TRANSFER_LEARNING_LR) 

        # 步驟 2: 載入檢查點
        start_epoch, best_accuracy, is_resumed = load_checkpoint(checkpoint_path, model, initial_optimizer)
        
        # 步驟 3: 確保模型凍結狀態正確
        if USE_PRETRAINED:
            # 如果是從頭開始或剛載入檢查點，需要確保基礎層是凍結的，除非已進入微調階段
            if start_epoch <= 100:
                # 如果尚未進入微調階段，確保基礎層處於凍結狀態 (只訓練分類器)
                freeze_base_layers(model, freeze=True)
            else:
                # 如果從超過 100 Epoch 的地方續訓，預設進入微調
                freeze_base_layers(model, freeze=False)
             
        # 步驟 4: 開始訓練
        train_model(train_loader, val_loader, model, NUM_EPOCHS, start_epoch, best_accuracy)
        
    except ValueError as e:
        print(f"\n[資料錯誤] {e}\n請檢查 {DATA_DIR} 目錄下的檔案是否齊全且符合 `class-split` 命名格式。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 訓練過程中發生錯誤: {e}")
