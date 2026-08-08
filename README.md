# Vietnamese Spelling Correction

Triển khai PyTorch gọn (~1M tham số) của kiến trúc Hierarchical Transformer
Encoders for Vietnamese Spelling Correction. Character encoder tạo vector cho
mỗi token, sau đó word encoder dùng ngữ cảnh câu để phát hiện và sửa lỗi.

## Thành phần

- `models.py`: Hierarchical Transformer và cấu hình Compact-1M.
- `train_stream.py`: trainer BF16 đọc JSONL theo stream, không nạp 5 GB vào RAM.
- `datasets/build_dataset.py`: trích xuất, tạo lỗi tổng hợp, vocabulary và verify.
- `Dockerfile`: image CUDA/BF16 để train trên RunPod.
- `RUNPOD.md`: hướng dẫn cho thành viên nhóm.

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
python -u train_stream.py --epochs 3 --warmup-epochs 2 --amp-dtype bf16 \
  --output-dir checkpoints/full_1gb_bf16_3_epochs
```

## Train trên RunPod

Build/push image theo Dockerfile, tạo Pod với volume mount tại `/workspace`,
và đặt `AUTO_DOWNLOAD=1`. Entry point sẽ tải dataset vào volume một lần rồi
bắt đầu train BF16. Xem chi tiết trong [RUNPOD.md](RUNPOD.md).

## Giới hạn hiện tại

Corrector hiện là classifier một-token-sang-một-token với vocabulary 9.500 từ.
Do đó chưa hỗ trợ lỗi dính/tách token; khoảng 9,6% target lỗi tổng hợp nằm ngoài
vocabulary. Đây là hướng ưu tiên cho phiên bản dùng subword/character decoder.
