import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for download tracking"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split(os.sep)[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def ensure_models():
    """
    Download all required models for FaceReenact-Pro.
    This will automatically download models on first run.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("FaceReenact-Pro Model Downloader")
    print("=" * 60)
    
    all_success = True

    # 1. Main face swap model: inswapper_128.onnx (~554MB)
    model_path = os.path.join(model_dir, "inswapper_128.onnx")
    if not os.path.exists(model_path):
        print("\n[1/2] Downloading face swap model (~554MB)...")
        print("This will happen only once and may take a few minutes.")
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        try:
            download_url(url, model_path)
            print("✓ Face swap model downloaded successfully!")
        except urllib.error.HTTPError as e:
            print(f"✗ Error downloading model: {e}")
            print("\nManual download required:")
            print("URL: https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main")
            print(f"Save to: {model_path}")
            all_success = False
    else:
        print("\n[1/2] ✓ Face swap model already exists")

    # 2. GFPGAN for face restoration (~349MB)
    gfpgan_path = os.path.join(model_dir, "GFPGANv1.4.pth")
    if not os.path.exists(gfpgan_path):
        print("\n[2/2] Downloading face enhancer model (~349MB)...")
        print("This improves output quality significantly.")
        url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        try:
            download_url(url, gfpgan_path)
            print("✓ Face enhancer downloaded successfully!")
        except urllib.error.HTTPError as e:
            print(f"⚠ Warning: Could not download from GitHub: {e}")
            print("Trying alternative source...")
            # Fallback to Hugging Face mirror
            try:
                alt_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth"
                download_url(alt_url, gfpgan_path)
                print("✓ Face enhancer downloaded from alternative source!")
            except Exception as e2:
                print(f"✗ Alternative download also failed: {e2}")
                print("\nManual download (optional but recommended):")
                print("URL: https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.0")
                print(f"Save to: {gfpgan_path}")
                print("\nNote: You can continue without it, but face quality may be lower.")
    else:
        print("\n[2/2] ✓ Face enhancer model already exists")

    print("\n" + "=" * 60)
    if all_success:
        print("✓ All models are ready!")
        print("You can now run the face swap application.")
    else:
        print("⚠ Some models require manual download.")
        print("See instructions above.")
    print("=" * 60)
    
    return all_success

if __name__ == "__main__":
    ensure_models()










# model 1



# import os
# import urllib.request
# from tqdm import tqdm

# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)

# def download_url(url, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# def ensure_models():
#     model_dir = "models"
#     os.makedirs(model_dir, exist_ok=True)

#     # Main model: inswapper_128.onnx (standard version)
#     model_path = os.path.join(model_dir, "inswapper_128.onnx")
#     if not os.path.exists(model_path):
#         print("Downloading face swap model (~250MB)... This will happen only once.")
#         # Updated URL - using the main inswapper_128.onnx file
#         url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
#         try:
#             download_url(url, model_path)
#             print("Model downloaded successfully!")
#         except urllib.error.HTTPError as e:
#             print(f"Error downloading model: {e}")
#             print("\nAlternative: Please manually download the model from:")
#             print("https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main")
#             print(f"And save it to: {model_path}")
#             return False

#     # GFPGAN for face restoration (optional but recommended)
#     gfpgan_path = os.path.join(model_dir, "GFPGANv1.4.pth")
#     if not os.path.exists(gfpgan_path):
#         print("Downloading face enhancer model...")
#         url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.8/GFPGANv1.4.pth"
#         try:
#             download_url(url, gfpgan_path)
#             print("Face enhancer downloaded successfully!")
#         except urllib.error.HTTPError as e:
#             print(f"Warning: Could not download face enhancer: {e}")
#             print("You can continue without it, but face quality may be lower.")

#     return True

# if __name__ == "__main__":
#     ensure_models()


# model 2


# import os
# import urllib.request
# from tqdm import tqdm

# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)

# def download_url(url, output_path):
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# def ensure_models():
#     model_dir = "models"
#     os.makedirs(model_dir, exist_ok=True)

#     # Main model: inswapper_128.onnx (standard version)
#     model_path = os.path.join(model_dir, "inswapper_128.onnx")
#     if not os.path.exists(model_path):
#         print("Downloading face swap model (~250MB)... This will happen only once.")
#         url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
#         try:
#             download_url(url, model_path)
#             print("Model downloaded successfully!")
#         except urllib.error.HTTPError as e:
#             print(f"Error downloading model: {e}")
#             print("\nAlternative: Please manually download the model from:")
#             print("https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main")
#             print(f"And save it to: {model_path}")
#             return False

#     # GFPGAN for face restoration (optional but recommended)
#     gfpgan_path = os.path.join(model_dir, "GFPGANv1.4.pth")
#     if not os.path.exists(gfpgan_path):
#         print("Downloading face enhancer model...")
#         # Updated URL - the correct GitHub release link
#         url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
#         try:
#             download_url(url, gfpgan_path)
#             print("Face enhancer downloaded successfully!")
#         except urllib.error.HTTPError as e:
#             print(f"Warning: Could not download face enhancer: {e}")
#             print("Trying alternative source...")
#             # Alternative Hugging Face mirror
#             try:
#                 alt_url = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth"
#                 download_url(alt_url, gfpgan_path)
#                 print("Face enhancer downloaded successfully from alternative source!")
#             except Exception as e2:
#                 print(f"Alternative download also failed: {e2}")
#                 print("You can continue without it, but face quality may be lower.")
#                 print("\nManual download:")
#                 print("https://github.com/TencentARC/GFPGAN/releases")
#                 print(f"Save to: {gfpgan_path}")

#     return True

# if __name__ == "__main__":
#     if ensure_models():
#         print("\n✓ All models are ready!")
#     else:
#         print("\n⚠ Some models need manual download.")
