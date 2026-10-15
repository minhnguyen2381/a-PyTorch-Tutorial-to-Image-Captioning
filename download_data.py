import os
import sys
import time
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

def download_file(url, target_path, min_expected_size=1000000, max_retries=3):
    if os.path.exists(target_path) and os.path.getsize(target_path) >= min_expected_size:
        print(f"File {target_path} đã tồn tại ({os.path.getsize(target_path) / (1024*1024):.1f} MB), bỏ qua tải.")
        return

    part_path = target_path + ".part"
    print(f"Đang tải: {url} -> {target_path}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\rTiến độ: {percent:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\rĐã tải: {downloaded / (1024*1024):.1f} MB")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for attempt in range(1, max_retries + 1):
        try:
            if os.path.exists(part_path):
                os.remove(part_path)
            urllib.request.urlretrieve(url, part_path, progress_hook)
            sys.stdout.write("\n")
            if os.path.getsize(part_path) < min_expected_size:
                raise ValueError(f"Dung lượng tải về quá nhỏ ({os.path.getsize(part_path)} bytes)")
            if os.path.exists(target_path):
                os.remove(target_path)
            os.rename(part_path, target_path)
            print(f"Tải hoàn tất: {target_path}")
            return
        except Exception as e:
            print(f"\n[Lần thử {attempt}/{max_retries}] Lỗi tải file: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                raise e

def extract_zip(zip_path, extract_to):
    print(f"Đang giải nén: {zip_path} -> {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Giải nén hoàn tất!")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    captions_zip = os.path.join(DATA_DIR, 'caption_datasets.zip')
    images_zip = os.path.join(DATA_DIR, 'Flickr8k_Dataset.zip')

    json_path = os.path.join(DATA_DIR, 'dataset_flickr8k.json')
    target_images_dir = os.path.join(DATA_DIR, 'Flicker8k_Dataset')

    # 1. Tải và giải nén file captions nếu chưa có file json
    if not os.path.exists(json_path):
        download_file(CAPTIONS_URL, captions_zip, min_expected_size=30*1024*1024)
        extract_zip(captions_zip, DATA_DIR)
    else:
        print(f"File chú thích {json_path} đã tồn tại, bỏ qua giải nén caption_datasets.zip.")

    # 2. Tải và giải nén file ảnh nếu chưa có đủ ảnh
    num_existing_images = 0
    if os.path.exists(target_images_dir):
        num_existing_images = len([f for f in os.listdir(target_images_dir) if f.lower().endswith('.jpg')])

    if num_existing_images < 8000:
        download_file(IMAGES_URL, images_zip, min_expected_size=900*1024*1024)
        extract_zip(images_zip, DATA_DIR)

        # Đảm bảo tên thư mục ảnh là Flicker8k_Dataset (có chữ e)
        alt_images_dir = os.path.join(DATA_DIR, 'Flickr8k_Dataset')
        if os.path.exists(alt_images_dir) and not os.path.exists(target_images_dir):
            os.rename(alt_images_dir, target_images_dir)
    else:
        print(f"Thư mục ảnh {target_images_dir} đã có đủ {num_existing_images} ảnh, bỏ qua giải nén lại.")

    # Dọn dẹp thư mục rác __MACOSX nếu có
    macosx_dir = os.path.join(DATA_DIR, '__MACOSX')
    if os.path.exists(macosx_dir):
        shutil.rmtree(macosx_dir, ignore_errors=True)

    # 3. Kiểm tra tính toàn vẹn cuối cùng
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
