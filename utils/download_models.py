import os
import sys

# Try to import huggingface_hub
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ Error: 'huggingface_hub' is missing.")
    print("👉 Please run: pip install -r requirements.txt")
    sys.exit(1)

def download_qwen_model():
    """
    Downloads the Qwen2.5-VL-3B-Instruct model from Hugging Face.
    This model is used as the fallback intelligence layer for difficult handwriting.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)  
    LOCAL_DIR = os.path.join(root_dir, "models", "qwen2.5-vl-3b")
    # Configuration
    REPO_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    
    
    print(f"  Target Model: {REPO_ID}")
    print(f" Save Location: {LOCAL_DIR}")

    # Create directory if it doesn't exist
    os.makedirs(LOCAL_DIR, exist_ok=True)

    try:
        print(f" Starting download... (This may take a few minutes depending on internet speed)")
        
        # Download the model
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=LOCAL_DIR,
        )
        
        print("\n✅ SUCCESS: Qwen-2.5-VL model downloaded successfully!")
        print(f"   You can now run the pipeline with: python executable.py ... --use-vlm")
        
    except Exception as e:
        print(f"\n❌ ERROR: Download failed.")
        print(f"   Reason: {e}")
        print("   Tip: Check your internet connection or try running this script again.")

if __name__ == "__main__":
    download_qwen_model()