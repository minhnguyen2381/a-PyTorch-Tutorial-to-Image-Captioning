# Kế Hoạch Triển Khai: Chuẩn Bị Dữ Liệu và Huấn Luyện Mô Hình Image Captioning (Flickr8k)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng script tự động tải dữ liệu Flickr8k, giải nén và kiểm tra tính toàn vẹn, thực hiện tiền xử lý sinh file HDF5/JSON, sau đó khởi chạy huấn luyện mô hình Show, Attend and Tell trên GPU NVIDIA RTX 5060 Laptop và theo dõi quá trình huấn luyện qua file nhật ký.

**Architecture:** Tạo script `download_data.py` bằng Python tải trực tiếp `caption_datasets.zip` và `Flickr8k_Dataset.zip` vào `data/`, giải nén bằng `zipfile`. Chạy `create_input_files.py` để sinh word map và các file nhị phân HDF5. Khởi chạy `train.py` ghi log ra `training.log`, định kỳ giám sát checkpoint và điểm BLEU-4.

**Tech Stack:** Python 3.12, PyTorch 2.x, Torchvision, CUDA 12.x, h5py, tqdm/urllib, git.

## Global Constraints

- Chạy trên hệ điều hành Windows 11, kích hoạt Python từ môi trường ảo `.\.venv\Scripts\python.exe`.
- Tham số `workers = 0` trong `train.py` để tránh lỗi pickle của thư viện `h5py` trên Windows.
- Thư mục dữ liệu: `data/` nằm tại thư mục gốc của repo.
- Thư mục ảnh giải nén: `data/Flicker8k_Dataset` (lưu ý chữ `Flicker` có `e` để khớp với `create_input_files.py`).
- Mọi thông báo và nhật ký cập nhật cho người dùng đều sử dụng Tiếng Việt.
- Tuân thủ quy tắc: Không tự ý thực thi mã nguồn cho tới khi người dùng duyệt kế hoạch triển khai này.

---

### Task 1: Xây dựng Script Tải & Giải nén Dữ liệu Flickr8k (`download_data.py`)

**Files:**
- Create: `download_data.py`
- Output: `data/dataset_flickr8k.json`, `data/Flicker8k_Dataset/*.jpg`

**Interfaces:**
- Consumes: URL `http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip` và `https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip`.
- Produces: Thư mục `data/` với tệp `data/dataset_flickr8k.json` và thư mục ảnh `data/Flicker8k_Dataset` chứa đủ 8.091 ảnh JPG.

- [ ] **Step 1: Viết script `download_data.py`**
Tạo file `download_data.py` tại thư mục gốc với nội dung:
```python
import os
import sys
import shutil
import zipfile
import urllib.request

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
```

- [ ] **Step 2: Chạy script tải và giải nén dữ liệu**
Chạy lệnh:
```powershell
.\.venv\Scripts\python.exe download_data.py
```
Kỳ vọng: Cả 2 file zip được tải về thư mục `data/`, giải nén thành công, xuất hiện thông báo xác thực `Số ảnh: 8091`.

- [ ] **Step 3: Commit code script tải dữ liệu**
```powershell
git add download_data.py
git commit -m "feat: add download_data.py for automatic flickr8k ingestion"
```

---

### Task 2: Thực Hiện Tiền Xử Lý Dữ Liệu (`create_input_files.py`)

**Files:**
- Modify: `create_input_files.py` (nếu cần chỉnh sửa đường dẫn, mặc định đã khớp)
- Output: `data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json`, các file HDF5 và JSON trong `data/`

**Interfaces:**
- Consumes: `data/dataset_flickr8k.json`, `data/Flicker8k_Dataset`
- Produces:
  - `data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json`
  - `data/TRAIN_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5`
  - `data/VAL_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5`
  - `data/TEST_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5`
  - `data/TRAIN_CAPTIONS_...json`, `data/VAL_CAPTIONS_...json`, `data/TEST_CAPTIONS_...json`
  - `data/TRAIN_CAPLENS_...json`, `data/VAL_CAPLENS_...json`, `data/TEST_CAPLENS_...json`

- [ ] **Step 1: Kiểm tra các tham số trong `create_input_files.py`**
Đảm bảo các tham số khớp chính xác:
```python
create_input_files(dataset='flickr8k',
                   karpathy_json_path='data/dataset_flickr8k.json',
                   image_folder='data/Flicker8k_Dataset',
                   captions_per_image=5,
                   min_word_freq=5,
                   output_folder='data',
                   max_len=50)
```

- [ ] **Step 2: Thực thi tiền xử lý dữ liệu**
Chạy lệnh:
```powershell
.\.venv\Scripts\python.exe create_input_files.py
```
Kỳ vọng: Quá trình đọc ảnh, chuẩn hóa kích thước $256 \times 256$, tạo word map diễn ra trong 2-5 phút mà không gặp lỗi.

- [ ] **Step 3: Xác minh các file đầu ra**
Kiểm tra bằng lệnh PowerShell:
```powershell
Get-ChildItem -Path data -Filter "*flickr8k*" | Select-Object Name, Length
```
Kỳ vọng: Đầy đủ các file HDF5 (Train ~1.2GB, Val ~200MB, Test ~200MB) và file `WORDMAP_...json`.

---

### Task 3: Khởi Chạy & Giám Sát Huấn Luyện Trên GPU (`train.py`)

**Files:**
- Verify: `train.py`
- Output: `training.log`, `checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar`, `BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar`

**Interfaces:**
- Consumes: Các file HDF5 và JSON trong `data/`.
- Produces: Log tiến trình thời gian thực và file trọng số Checkpoint.

- [ ] **Step 1: Kiểm tra cấu hình trong `train.py`**
Xác nhận trong `train.py`:
- `data_folder = 'data'`
- `data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'`
- `batch_size = 32`
- `workers = 0`
- `epochs = 120`
- `fine_tune_encoder = False`
- `checkpoint = None`

- [ ] **Step 2: Khởi chạy huấn luyện dưới dạng tác vụ nền (Background Task)**
Chạy lệnh PowerShell chuyển hướng toàn bộ output stdout & stderr vào `training.log`:
```powershell
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "train.py" -RedirectStandardOutput "training.log" -RedirectStandardError "training_error.log"
```

- [ ] **Step 3: Giám sát log khởi động và xác thực chạy trên GPU**
Đợi 30-60 giây, sau đó đọc các dòng log đầu tiên:
```powershell
Get-Content -Path "training.log" -Tail 30
```
Kỳ vọng:
- ResNet-101 và DecoderWithAttention khởi tạo thành công trên GPU CUDA.
- In ra log training từng batch: `Epoch: [0][0/938]`, `Epoch: [0][100/938]...` với loss giảm dần.
- File `training_error.log` rỗng hoặc không có ngoại lệ nghiêm trọng.

- [ ] **Step 4: Cập nhật và báo cáo trạng thái hoàn thành khởi chạy huấn luyện**
Thông báo tóm tắt cho người dùng về trạng thái huấn luyện đang chạy nền, cách thức theo dõi và hướng dẫn sử dụng mô hình sau khi train hoàn tất.
