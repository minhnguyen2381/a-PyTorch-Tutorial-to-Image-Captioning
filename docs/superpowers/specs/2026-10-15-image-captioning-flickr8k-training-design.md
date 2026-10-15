# Thiết Kế Hệ Thống: Chuẩn Bị Dữ Liệu và Huấn Luyện Mô Hình Image Captioning (Flickr8k)

- **Ngày tạo**: 2026-10-15
- **Trạng thái**: Đã phê duyệt (Approved)
- **Tác vụ**: Tải dữ liệu Flickr8k, tiền xử lý và huấn luyện mô hình Show, Attend and Tell với PyTorch trên GPU NVIDIA RTX 5060 Laptop

---

## 1. Tổng quan & Mục tiêu

### 1.1. Bối cảnh
Dự án triển khai mô hình sinh chú thích ảnh tự động (*Image Captioning*) theo kiến trúc **Show, Attend and Tell (Xu et al., 2015)**:
- **Encoder**: Mạng tích chập ResNet-101 trích xuất đặc trưng không gian (spatial feature map $14 \times 14 \times 2048$).
- **Attention**: Cơ chế Bahdanau Attention chấm điểm trọng số cho 196 vùng ảnh tại mỗi bước giải mã.
- **Decoder**: Mạng hồi quy LSTM sinh chuỗi từ ngữ kết hợp với trọng số chú ý và cơ chế Doubly Stochastic Attention Regularization.

Hiện tại mã nguồn đã được cập nhật tương thích PyTorch 2.x, môi trường ảo `.venv` đã cài đặt đầy đủ thư viện và nhận diện GPU **NVIDIA GeForce RTX 5060 Laptop (8GB VRAM)**. Tuy nhiên, thư mục `data/` chưa có dữ liệu thô (ảnh và chú thích Flickr8k) cũng như chưa có các tệp tiền xử lý HDF5/JSON.

### 1.2. Mục tiêu (Goals)
1. Xây dựng quy trình tự động hóa tải 2 bộ dữ liệu: `caption_datasets.zip` (~40MB) và `Flickr8k_Dataset.zip` (~1GB).
2. Tự động giải nén và kiểm tra tính toàn vẹn của dữ liệu (file chú thích JSON và đủ 8.091 ảnh JPG).
3. Thực hiện tiền xử lý dữ liệu qua `create_input_files.py`, xuất ra các file HDF5 và JSON word map.
4. Thiết lập và khởi chạy quá trình huấn luyện mô hình qua `train.py` dưới dạng tác vụ nền (background process), ghi nhật ký đầy đủ ra file log.
5. Theo dõi điểm số BLEU-4 và lưu trữ checkpoint định kỳ cùng checkpoint tốt nhất (`BEST_checkpoint_...`).

### 1.3. Ngoài phạm vi (Non-goals)
- Không fine-tune Encoder (ResNet-101) ngay từ đầu (giữ `fine_tune_encoder = False` theo cấu hình chuẩn để tối ưu tốc độ và tránh overfitting sớm).
- Không sửa đổi kiến trúc lõi trong `models.py`.

---

## 2. Kiến trúc & Luồng Xử lý Dữ liệu

```
[Internet Sources]
       │
       ▼ (download_data.py)
[data/caption_datasets.zip] & [data/Flickr8k_Dataset.zip]
       │
       ▼ (ZipFile extraction & Validation)
[data/dataset_flickr8k.json] & [data/Flicker8k_Dataset/*.jpg (8,091 images)]
       │
       ▼ (create_input_files.py)
┌──────────────────────────────────────────────────────────────────┐
│  data/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json        │
│  data/TRAIN_IMAGES_...hdf5, VAL_IMAGES_...hdf5, TEST_IMAGES...   │
│  data/TRAIN_CAPTIONS_...json, VAL_CAPTIONS... TEST_CAPTIONS...   │
└──────────────────────────────────────────────────────────────────┘
       │
       ▼ (train.py on GPU RTX 5060)
┌──────────────────────────────────────────────────────────────────┐
│  Epoch Iteration (Forward -> Attention Loss -> Backprop -> Step) │
│  Validation BLEU-4 Calculation at each Epoch                     │
│  Save: checkpoint_...pth.tar & BEST_checkpoint_...pth.tar        │
│  Log: training.log                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Đặc tả Chi tiết Các Thành Phần

### 3.1. Module Tải Dữ liệu & Giải nén (`download_data.py`)
- **Vị trí**: Tạo script tiện ích tạm `scripts/download_data.py` (hoặc chạy trong root).
- **URLs nguồn**:
  - Captions: `http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip`
  - Images: `https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip`
- **Quy tắc thực thi**:
  - Kiểm tra xem file zip đã tồn tại đủ kích thước chưa. Nếu đã có thì bỏ qua bước tải.
  - Tải dạng streaming với chunk 1MB, hiển thị tiến độ % và dung lượng tải về.
  - Giải nén bằng `zipfile.ZipFile` vào thư mục `data/`.
  - Tự động dọn dẹp thư mục rác `__MACOSX` nếu có.
- **Tiêu chuẩn nghiệm thu dữ liệu thô**:
  - Tồn tại tệp `data/dataset_flickr8k.json` (kích thước ~4MB).
  - Tồn tại thư mục `data/Flicker8k_Dataset` chứa đúng hoặc tối thiểu 8.091 tệp `.jpg`.

### 3.2. Module Tiền Xử lý Dữ liệu (`create_input_files.py`)
- **Cấu hình**:
  - `dataset = 'flickr8k'`
  - `karpathy_json_path = 'data/dataset_flickr8k.json'`
  - `image_folder = 'data/Flicker8k_Dataset'`
  - `captions_per_image = 5`
  - `min_word_freq = 5`
  - `output_folder = 'data'`
  - `max_len = 50`
- **Tiêu chuẩn nghiệm thu**:
  - `WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json` được tạo thành công.
  - `TRAIN_IMAGES_flickr8k_5_cap_per_img_5_min_word_freq.hdf5` (~1.2GB) và các file VAL/TEST images hdf5 được tạo không lỗi.
  - Các file captions và caplens dạng json đầy đủ.

### 3.3. Module Huấn Luyện (`train.py`)
- **Môi trường**: Kích hoạt `.venv`, kiểm tra `torch.cuda.is_available() == True`.
- **Tham số cấu hình**:
  - `device`: `cuda` (NVIDIA GeForce RTX 5060 Laptop GPU)
  - `batch_size`: `32` (chiếm ~3GB VRAM)
  - `workers`: `0` (bắt buộc cho môi trường Windows do hạn chế của multiprocessing với h5py)
  - `epochs`: `120`
  - `decoder_lr`: `4e-4`
  - `fine_tune_encoder`: `False`
  - `checkpoint`: `None` (khởi tạo từ đầu)
  - `grad_clip`: `5.0`
  - `alpha_c`: `1.0`
- **Quy trình lưu trữ checkpoint**:
  - Sau mỗi epoch: lưu `checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar`.
  - Khi `val_bleu4 > best_bleu4`: lưu đè `BEST_checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar`.
- **Nhật ký huấn luyện**:
  - Điều hướng output stdout/stderr vào file `training.log`.
  - Giám sát định kỳ các epoch ban đầu (Epoch 0, 1, ...) để xác nhận quá trình tính toán loss hội tụ bình thường.

---

## 4. Xử Lý Lỗi & Khả Năng Phục Hồi (Error Handling & Resilience)

| Tình huống rủi ro | Biện pháp xử lý |
|---|---|
| Mạng bị ngắt trong lúc tải file zip | Cơ chế tự động thử lại (retry 3 lần), kiểm tra kích thước file trước khi giải nén. |
| File zip bị lỗi / giải nén hỏng | Bắt ngoại lệ `zipfile.BadZipFile`, thông báo xóa file zip lỗi và tải lại. |
| Thư mục ảnh bị sai tên (`Flickr8k` vs `Flicker8k`) | Script chuẩn bị dữ liệu kiểm tra và chuẩn hóa tên thư mục thành `data/Flicker8k_Dataset` đúng với mã nguồn của `create_input_files.py`. |
| Tràn bộ nhớ GPU (CUDA OOM) | Mức chiếm dụng ước tính là ~3GB trên tổng số 8GB VRAM (an toàn). Nếu xảy ra OOM, giảm `batch_size` xuống 16. |
| Tiến trình train bị ngắt đột ngột | Chỉ cần trỏ biến `checkpoint = 'checkpoint_flickr8k_5_cap_per_img_5_min_word_freq.pth.tar'` trong `train.py` để tiếp tục từ epoch gần nhất mà không mất công huấn luyện lại. |

---

## 5. Kế Hoạch Xác Thực & Nghiệm Thu (Verification Plan)

1. **Kiểm tra tải dữ liệu**: Chạy lệnh kiểm tra file `data/dataset_flickr8k.json` và đếm số lượng ảnh trong `data/Flicker8k_Dataset`.
2. **Kiểm tra tiền xử lý**: Xác nhận file HDF5 và JSON word map tồn tại trong `data/` với kích thước hợp lệ.
3. **Kiểm tra khởi động huấn luyện**:
   - Khởi động `train.py`.
   - Đọc 50-100 dòng đầu của `training.log` để kiểm tra: nạp mô hình ResNet-101 thành công, nạp word map thành công, hoàn thành batch 0 và batch 100 với loss giảm dần, không có lỗi runtime CUDA.
   - Xác nhận checkpoint đầu tiên được lưu sau epoch 0.
