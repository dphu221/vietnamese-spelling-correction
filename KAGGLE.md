# Running on Kaggle GPU

Hướng dẫn chi tiết triển khai và huấn luyện mô hình **Vietnamese Spelling Correction** (~1M parameters Compact-1M) trên hạ tầng **Kaggle GPU Notebooks / Kaggle Scripts**.

---

## 🚀 Quick Start (Chạy Nhanh)

### Cách 1: Sử dụng Kaggle Notebook (`vietnamese_spelling_kaggle.ipynb`)

1. Tạo một Kaggle Notebook mới hoặc upload file [`vietnamese_spelling_kaggle.ipynb`](vietnamese_spelling_kaggle.ipynb) lên Kaggle.
2. Bật **GPU Accelerator** trong phần **Notebook Settings** (Khuyên dùng **GPU T4 x2** hoặc **GPU P100**).
3. Chạy từng ô lệnh (cell) theo thứ tự để:
   - Kiểm tra GPU & tự động chọn kiểu Mixed Precision (FP16 / BF16).
   - Tải dataset public `Sanng1112/vietnamese-spelling-synthetic-1gb` từ Hugging Face Hub.
   - Huấn luyện mô hình streaming theo từng epoch.
   - Vẽ biểu đồ Loss, F1 score và Correction Recall.
   - Thử nghiệm sửa lỗi trực tiếp trên các câu tiếng Việt.

---

### Cách 2: Chạy bằng Lệnh Terminal / Kaggle Script

Nếu bạn muốn chạy script trực tiếp trên Kaggle Notebook cell hoặc Kaggle Script:

```bash
# Cài đặt thư viện phụ thuộc
pip install -q huggingface_hub

# Huấn luyện 3 epochs với tự động chọn GPU precision & tải dataset
python kaggle_train.py --epochs 3 --batch-size 64 --amp-dtype auto
```

Hoặc dùng file shell script đi kèm:

```bash
bash scripts/kaggle_train.sh
```

---

## ⚙️ Cấu Hình Hardware & Precision (Mixed Precision)

Kaggle cung cấp các loại GPU phổ biến:
- **NVIDIA T4 (16GB VRAM)**: Không có phần cứng bfloat16 tensor cores native. Hệ thống sẽ tự chọn `--amp-dtype fp16` (kết hợp `GradScaler`) giúp tốc độ huấn luyện tối ưu nhất.
- **NVIDIA P100 (16GB VRAM)**: Tương tự T4, tự động chọn `--amp-dtype fp16`.
- **NVIDIA L4 / A100**: Hỗ trợ bfloat16 native, hệ thống tự động chọn `--amp-dtype bf16`.

Flag `--amp-dtype auto` (mặc định) sẽ tự động phát hiện GPU và chọn kiểu precision phù hợp nhất mà không gây lỗi hoặc giảm tốc độ.

---

## 📦 Quản Lý Dataset Trên Kaggle

Có 2 phương án sử dụng dataset trên Kaggle:

### Phương án 1: Auto Download từ HuggingFace (Khuyên dùng)
Đặt `--auto-download` (hoặc biến môi trường `AUTO_DOWNLOAD=1`), trainer sẽ tự động tải các file sau từ repo `Sanng1112/vietnamese-spelling-synthetic-1gb` vào thư mục làm việc (`/kaggle/working/data`):
- `train_full.jsonl` (4.56M mẫu)
- `validation_full.jsonl` (94.8K mẫu)
- `word_vocab_full.json` (9.500 từ)
- `char_vocab_full.json` (400 ký tự)

### Phương án 2: Thêm Kaggle Dataset Input
Nếu bạn đã tạo Kaggle Dataset từ trước:
1. Nhấp **Add Data** ở góc phải Kaggle Notebook.
2. Chọn dataset `vietnamese-spelling-synthetic-1gb` của bạn.
3. Trainer sẽ tự động tìm thấy data trong `/kaggle/input/vietnamese-spelling-synthetic-1gb/` mà không cần tải lại.

---

## 🔍 Thử Nghiệm Sửa Lỗi (Inference)

Sau khi huấn luyện xong, bạn có thể chạy mô hình để kiểm tra trực tiếp bằng `infer.py`:

### Dùng Lệnh CLI:
```bash
python infer.py --checkpoint /kaggle/working/checkpoints/full_1gb_bf16_3_epochs/best.pt \
                --text "Tôi là sinh viên trường dại học bách khoa"
```

### Dùng Python Code:
```python
from infer import VietnameseSpellingCorrectorPipeline

pipeline = VietnameseSpellingCorrectorPipeline.from_checkpoint(
    checkpoint_path="/kaggle/working/checkpoints/full_1gb_bf16_3_epochs/best.pt"
)

result = pipeline.correct_text("Tôi là sinh viên trường dại học bách khoa")
print("Gốc  :", result["original_text"])
print("Đã sửa:", result["corrected_text"])
```

---

## 💾 Lưu Và Tải Checkpoint Từ Kaggle

Sau khi hoàn thành huấn luyện, tất cả kết quả được ghi tại `/kaggle/working/checkpoints/full_1gb_bf16_3_epochs/`:
- `best.pt`: Checkpoint có `val_loss` tốt nhất.
- `last.pt`: Checkpoint ở epoch cuối cùng.
- `history.json`: Lịch sử loss, F1 score và recall theo từng epoch.

Để tải `best.pt` về máy cá nhân:
1. Mở panel **Output** ở góc phải giao diện Kaggle.
2. Tìm thư mục `checkpoints` -> bấm biểu tượng **Download** bên cạnh `best.pt`.

---

## 🛠 Troubleshooting (Xử Lý Lỗi Phổ Biến)

### `ModuleNotFoundError: No module named 'train_stream'`
Lỗi này xảy ra khi Jupyter Notebook được tạo ở thư mục gốc `/kaggle/working` trong khi các file `.py` nằm trong subfolder `vietnamese-spelling-correction`.

**Giải pháp**:
Thêm đoạn mã sau vào cell đầu tiên trước khi import:
```python
import os, sys
from pathlib import Path

repo_dir = Path("vietnamese-spelling-correction").resolve() if Path("vietnamese-spelling-correction").exists() else Path(".").resolve()
os.chdir(repo_dir)
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))
```
*(Lưu ý: File [`vietnamese_spelling_kaggle.ipynb`](vietnamese_spelling_kaggle.ipynb) đi kèm đã tích hợp sẵn đoạn mã xử lý đường dẫn tự động này ở Cell 2).*

