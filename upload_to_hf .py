"""
上傳訓練好的模型到 Hugging Face
優化版本：直接上傳資料夾（避免 API 限制）
修正版本：移除不兼容參數
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# 加載環境變數
load_dotenv('file.env')

def upload_entire_folder():
    """一次上傳整個 models 資料夾"""
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in file.env")
        return False
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    # 檢查是否有模型文件
    model_files = list(models_dir.glob('*.pt'))
    if not model_files:
        print("❌ No .pt model files found in models/")
        return False
    
    print("=" * 70)
    print("🚀 Hugging Face Folder Upload")
    print("=" * 70)
    print(f"")
    print(f"📦 Models directory: {models_dir}")
    print(f"📊 Model files found: {len(model_files)}")
    print(f"💾 Total size: {sum(f.stat().st_size for f in model_files) / (1024**2):.2f} MB")
    print("")
    
    # 獲取 repo 名稱
    hf_model_repo = os.getenv('HF_MODEL_REPO', 'your_username/trading-models')
    
    print(f"📤 Target repository: {hf_model_repo}")
    print(f"🔑 Using HF_TOKEN from file.env")
    print("")
    
    api = HfApi()
    
    # 建立 repo（如果不存在）
    try:
        print("📍 Creating/checking repository...")
        create_repo(
            repo_id=hf_model_repo,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=hf_token
        )
        print(f"✅ Repository ready: {hf_model_repo}")
    except Exception as e:
        print(f"❌ Failed to create/access repo: {e}")
        return False
    
    # 上傳整個資料夾
    try:
        print("")
        print("📤 Uploading entire models folder...")
        print("   (This may take a few minutes depending on folder size)")
        print("")
        
        api.upload_folder(
            folder_path=str(models_dir),
            repo_id=hf_model_repo,
            repo_type="model",
            token=hf_token,
            commit_message="Upload all trained models"
        )
        
        print("")
        print("=" * 70)
        print(f"✅ Upload complete!")
        print(f"📍 Models are at: https://huggingface.co/{hf_model_repo}")
        print("=" * 70)
        return True
    
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def upload_dataset_folder():
    """上傳 data 資料夾到 Dataset repo"""
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not found in file.env")
        return False
    
    data_dir = Path('backend/data')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # 檢查是否有 CSV 文件
    csv_files = list(data_dir.glob('**/*.csv'))
    if not csv_files:
        print("❌ No .csv data files found in backend/data/")
        return False
    
    print("=" * 70)
    print("🚀 Hugging Face Dataset Upload")
    print("=" * 70)
    print(f"")
    print(f"📦 Data directory: {data_dir}")
    print(f"📊 CSV files found: {len(csv_files)}")
    print(f"💾 Total size: {sum(f.stat().st_size for f in csv_files) / (1024**2):.2f} MB")
    print("")
    
    # 獲取 repo 名稱
    hf_dataset_repo = os.getenv('HF_DATASET_REPO', 'your_username/trading-data')
    
    print(f"📤 Target repository: {hf_dataset_repo}")
    print(f"🔑 Using HF_TOKEN from file.env")
    print("")
    
    api = HfApi()
    
    # 建立 repo（如果不存在）
    try:
        print("📍 Creating/checking repository...")
        create_repo(
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            private=False,
            exist_ok=True,
            token=hf_token
        )
        print(f"✅ Repository ready: {hf_dataset_repo}")
    except Exception as e:
        print(f"❌ Failed to create/access repo: {e}")
        return False
    
    # 上傳整個資料夾
    try:
        print("")
        print("📤 Uploading entire data folder...")
        print("   (This may take a few minutes depending on folder size)")
        print("")
        
        api.upload_folder(
            folder_path=str(data_dir),
            repo_id=hf_dataset_repo,
            repo_type="dataset",
            token=hf_token,
            commit_message="Upload training data"
        )
        
        print("")
        print("=" * 70)
        print(f"✅ Upload complete!")
        print(f"📍 Data is at: https://huggingface.co/datasets/{hf_dataset_repo}")
        print("=" * 70)
        return True
    
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def main():
    """主程式"""
    print("")
    print("🤖 Hugging Face Upload Tool")
    print("")
    print("Choose what to upload:")
    print("1. Upload models/ folder")
    print("2. Upload data/ folder")
    print("3. Upload both")
    print("")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    print("")
    
    results = {
        'models': False,
        'data': False
    }
    
    if choice in ['1', '3']:
        results['models'] = upload_entire_folder()
        print("")
    
    if choice in ['2', '3']:
        results['data'] = upload_dataset_folder()
        print("")
    
    # 總結
    if choice in ['1', '2', '3']:
        print("=" * 70)
        print("📊 Upload Summary")
        print("=" * 70)
        if choice in ['1', '3']:
            status = "✅ Success" if results['models'] else "❌ Failed"
            print(f"Models: {status}")
        if choice in ['2', '3']:
            status = "✅ Success" if results['data'] else "❌ Failed"
            print(f"Data: {status}")
        print("=" * 70)


if __name__ == '__main__':
    main()
