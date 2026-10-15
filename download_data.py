import os
import sys
import shutil
import zipfile
import urllib.request

# Đảm bảo UTF-8 output trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

DATA_DIR = 'data'
CAPTIONS_URL = 'http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip'
IMAGES_URL = 'https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip'

def download_file(url, target_path):
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        print(f"File {target_path} đã tồn tại ({os.path.getsize(target_path) / (1024*1024):.1f} MB), bỏ qua tải.")
        return

    print(f"Đang tải: {url} -> {target_path}")
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded * 100 / total_size
            sys.stdout.write(f"\rTiến độ: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rĐã tải: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, target_path, progress_hook)
    print("\nTải hoàn tất!")

def extract_zip(zip_path, extract_to):
    print(f"Đang giải nén: {zip_path} -> {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Giải nén hoàn tất!")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    captions_zip = os.path.join(DATA_DIR, 'caption_datasets.zip')
    images_zip = os.path.join(DATA_DIR, 'Flickr8k_Dataset.zip')

    download_file(CAPTIONS_URL, captions_zip)
    download_file(IMAGES_URL, images_zip)

    extract_zip(captions_zip, DATA_DIR)
    extract_zip(images_zip, DATA_DIR)

    # Dọn dẹp thư mục rác __MACOSX nếu có
    macosx_dir = os.path.join(DATA_DIR, '__MACOSX')
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir, ignore_errors=True)

    # Đảm bảo tên thư mục ảnh là Flicker8k_Dataset (có chữ e)
    alt_images_dir = os.path.join(DATA_DIR, 'Flickr8k_Dataset')
    target_images_dir = os.path.join(DATA_DIR, 'Flicker8k_Dataset')
    if os.path.exists(alt_images_dir) and not os.path.exists(target_images_dir):
        os.rename(alt_images_dir, target_images_dir)

    # Kiểm tra tính toàn vẹn
    json_path = os.path.join(DATA_DIR, 'dataset_flickr8k.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy file {json_path}")
    
    num_images = len([f for f in os.listdir(target_images_dir) if f.lower().endswith('.jpg')])
    print(f"\n[XÁC THỰC DỮ LIỆU]")
    print(f"- File chú thích: {json_path} (Tồn tại, {os.path.getsize(json_path)/(1024*1024):.2f} MB)")
    print(f"- Thư mục ảnh: {target_images_dir} (Số ảnh: {num_images})")
    if num_images < 8000:
        raise ValueError(f"Số lượng ảnh không đủ ({num_images} < 8000)")
    print(">>> Dữ liệu đã sẵn sàng cho bước tiền xử lý!")

if __name__ == '__main__':
    main()
