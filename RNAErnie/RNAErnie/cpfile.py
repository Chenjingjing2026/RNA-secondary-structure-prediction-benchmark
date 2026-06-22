import os
import shutil

def merge_train_valid_for_each_subdir(base_dir, output_base_dir):
    """
    為 base_dir 下的每個子目錄，將其內部的 train 和 valid 合併到輸出目錄的對應位置
    
    Args:
        base_dir: 包含多個子目錄（如16s、5s等）的基礎目錄
        output_base_dir: 輸出目錄的基礎路徑
    """
    # 遍歷 base_dir 下的所有子目錄
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # 只處理目錄，跳過文件
        if not os.path.isdir(subdir_path):
            continue
        
        # 檢查該子目錄下是否有 train 和 valid 目錄
        train_path = os.path.join(subdir_path, 'train')
        valid_path = os.path.join(subdir_path, 'valid')
        
        if not (os.path.exists(train_path) and os.path.exists(valid_path)):
            print(f"跳過 {subdir}: 缺少 train 或 valid 目錄")
            continue
        
        # 創建輸出目錄
        output_subdir = os.path.join(output_base_dir, subdir)
        merged_dir = os.path.join(output_subdir, 'train+valid')
        
        # 如果目標文件夾已存在，先刪除
        if os.path.exists(merged_dir):
            print(f"刪除已存在的目錄: {merged_dir}")
            shutil.rmtree(merged_dir)
        
        # 創建新的目標文件夾
        os.makedirs(merged_dir)
        
        print(f"\n處理 {subdir}...")
        print(f"  源 train: {train_path}")
        print(f"  源 valid: {valid_path}")
        print(f"  目標: {merged_dir}")
        
        # 複製文件計數
        train_count = 0
        valid_count = 0
        conflict_count = 0
        
        # 複製 train 目錄中的文件
        for file in os.listdir(train_path):
            src_file = os.path.join(train_path, file)
            if os.path.isfile(src_file):
                dst_file = os.path.join(merged_dir, file)
                
                # 處理重名文件（添加 train_ 前綴）
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    dst_file = os.path.join(merged_dir, f"train_{name}{ext}")
                    conflict_count += 1
                
                shutil.copy2(src_file, dst_file)
                train_count += 1
        
        # 複製 valid 目錄中的文件
        for file in os.listdir(valid_path):
            src_file = os.path.join(valid_path, file)
            if os.path.isfile(src_file):
                dst_file = os.path.join(merged_dir, file)
                
                # 處理重名文件（添加 valid_ 前綴）
                if os.path.exists(dst_file):
                    name, ext = os.path.splitext(file)
                    dst_file = os.path.join(merged_dir, f"valid_{name}{ext}")
                    conflict_count += 1
                
                shutil.copy2(src_file, dst_file)
                valid_count += 1
        
        print(f"  完成! train: {train_count} 個文件, valid: {valid_count} 個文件, 衝突重命名: {conflict_count} 個")
    
    print("\n" + "="*60)
    print("所有目錄處理完成！")
    print(f"輸出基礎目錄: {output_base_dir}")

# 使用示例
if __name__ == "__main__":
    # 配置路徑
    base_dir = '/home/cjj/Model_Data/Crossfamily'
    output_base_dir = '/home/cjj/RNAErnie/TempData/Crossfamily'
    
    # 檢查輸入目錄是否存在
    if not os.path.exists(base_dir):
        print(f"錯誤：基礎目錄 {base_dir} 不存在")
        exit(1)
    
    # 創建輸出基礎目錄（如果不存在）
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"創建輸出目錄: {output_base_dir}")
    
    # 執行合併
    merge_train_valid_for_each_subdir(base_dir, output_base_dir)
