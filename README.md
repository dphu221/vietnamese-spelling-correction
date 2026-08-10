# Vietnamese Spelling Correction

Triển khai PyTorch gọn (~1M tham số) của kiến trúc Hierarchical Transformer
Encoders for Vietnamese Spelling Correction. Character encoder tạo vector cho
mỗi token, sau đó word encoder dùng ngữ cảnh câu để phát hiện và sửa lỗi.

## Thành phần

- `models.py`: Hierarchical Transformer và cấu hình Compact-1M.
- `train_stream.py`: trainer mixed precision (BF16/FP16) đọc JSONL theo stream, không nạp 5 GB vào RAM.
- `infer.py`: module và CLI suy luận (inference) sửa lỗi chính tả trực tiếp trên câu.
- `kaggle_train.py` & `vietnamese_spelling_kaggle.ipynb`: script và Jupyter notebook dành cho Kaggle GPU.
- `datasets/build_dataset.py`: trích xuất, tạo lỗi tổng hợp, vocabulary và verify.
- `Dockerfile`: image CUDA/BF16 để train trên RunPod.
- `KAGGLE.md`: hướng dẫn chi tiết chạy trên Kaggle.
- `RUNPOD.md`: hướng dẫn chạy trên RunPod.

## Dataset

Dataset huấn luyện public: [Sanng1112/vietnamese-spelling-synthetic-1gb](https://huggingface.co/datasets/Sanng1112/vietnamese-spelling-synthetic-1gb).

| Split | Mẫu | Token | Lỗi tổng hợp |
|---|---:|---:|---:|
| Train | 4.564.618 | 117.414.000 | 9.730.826 |
| Validation | 94.836 | 2.426.854 | 202.379 |

Nguồn là Vietnamese Wikipedia đã lọc. Train/validation tách theo `page_id`,
không có overlap trang. Benchmark Viwiki-Spelling được tách riêng khỏi train.

## Train local

```bash
conda activate deeplearning_env
python -u train_stream.py --epochs 3 --warmup-epochs 2 --amp-dtype auto \
  --output-dir checkpoints/full_1gb_bf16_3_epochs
```

## Train trên Kaggle

Bạn có thể chạy trực tiếp bằng Kaggle Notebook [`vietnamese_spelling_kaggle.ipynb`](vietnamese_spelling_kaggle.ipynb) hoặc dùng script `kaggle_train.py`:

```bash
python kaggle_train.py --epochs 3 --batch-size 64 --amp-dtype auto
```

Xem chi tiết hướng dẫn thiết lập GPU, tự động tải dataset và xuất checkpoint trong [KAGGLE.md](KAGGLE.md).

## Train trên RunPod

Build/push image theo Dockerfile, tạo Pod với volume mount tại `/workspace`,
và đặt `AUTO_DOWNLOAD=1`. Entry point sẽ tải dataset vào volume một lần rồi
bắt đầu train BF16. Xem chi tiết trong [RUNPOD.md](RUNPOD.md).

## Inference (Sửa lỗi chính tả)

Sau khi có checkpoint, bạn có thể kiểm tra trực tiếp bằng CLI:

```bash
python infer.py --checkpoint checkpoints/full_1gb_bf16_3_epochs/best.pt \
  --text "Tôi là sinh viên trường dại học bách khoa"
```

## Giới hạn hiện tại

Corrector hiện là classifier một-token-sang-một-token với vocabulary 9.500 từ.
Do đó chưa hỗ trợ lỗi dính/tách token; khoảng 9,6% target lỗi tổng hợp nằm ngoài
vocabulary. Đây là hướng ưu tiên cho phiên bản dùng subword/character decoder.

## Ứng dụng web cục bộ

Repository có thêm giao diện React và API FastAPI để thử toàn bộ luồng nhập văn
bản, xem thay đổi, độ tin cậy, gợi ý gần nhất, tải tệp `.txt` và lưu lịch sử trên
trình duyệt. Mặc định ứng dụng chạy ở **chế độ minh họa**, dùng một bộ sửa xác
định nhỏ để kiểm tra giao diện trước khi checkpoint thật sẵn sàng. Kết quả ở chế
độ này luôn được ghi rõ là không phải đầu ra của mô hình.

### Cài đặt

Yêu cầu Python 3.11+ và Node.js 20+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
npm.cmd --prefix frontend install
```

### Chạy để phát triển

Mở hai terminal tại thư mục repository. Terminal thứ nhất chạy API:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal thứ hai chạy giao diện:

```powershell
npm.cmd --prefix frontend run dev
```

Mở `http://127.0.0.1:5173`. Giao diện giới hạn mỗi lần xử lý ở 5.000 ký tự;
backend tự chia đoạn dài thành các chunk không quá 192 token và giữ nguyên
khoảng trắng, dấu câu cùng xuống dòng của văn bản gốc.

Để chạy giao diện đã build chung với API:

```powershell
npm.cmd --prefix frontend run build
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Sau đó mở `http://127.0.0.1:8000`.

### Gắn checkpoint thật

Mô hình local phải có đúng bốn tệp trong cùng một thư mục:

```text
best.pt
word_vocab_full.json
char_vocab_full.json
model_manifest.json
```

Checkpoint hiện tại đã được đóng gói đủ bốn tệp tại
`checkpoints/full_1gb_bf16_3_epochs`. Tệp `.env` cục bộ đã bật cấu hình sau:

```dotenv
MODEL_SOURCE=local
MODEL_LOCAL_DIR=checkpoints/full_1gb_bf16_3_epochs
MODEL_DEVICE=auto
```

Khi chuyển sang checkpoint khác, sao chép `model.example.json` thành
`model_manifest.json`, đặt nó cạnh checkpoint và đúng hai vocabulary đã dùng
khi huấn luyện, rồi cập nhật `MODEL_LOCAL_DIR`.

Hugging Face Hub cũng được hỗ trợ:

```dotenv
MODEL_SOURCE=huggingface
HF_MODEL_REPO=ten-tai-khoan/ten-model
HF_MODEL_REVISION=main
MODEL_DEVICE=auto
```

`HF_TOKEN` chỉ cần cho repository private. Backend dùng `snapshot_download`,
kiểm tra manifest, kích thước vocabulary và cấu hình trong checkpoint trước khi
đưa mô hình vào phục vụ. Nếu artifact thiếu hoặc không tương thích, endpoint
`/api/health` báo trạng thái không sẵn sàng thay vì âm thầm dùng kết quả giả.

### API

```http
GET /api/health
POST /api/correct
```

Ví dụ request:

```json
{
  "text": "Hom nay tui ko đi học.",
  "mode": "balanced"
}
```

Ba mode `conservative`, `balanced`, `aggressive` lần lượt dùng ngưỡng phát hiện
0,80; 0,50; 0,30. Trường loại lỗi trong response chỉ là giải thích suy đoán từ
cặp token trước/sau, không phải nhãn được mô hình dự đoán.

### Kiểm thử

```powershell
python -m pytest backend\tests
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
```

