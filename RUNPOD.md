# RunPod cho thành viên nhóm

Mỗi thành viên dùng image Docker giống nhau và một persistent volume riêng,
mount tại `/workspace`. Data và checkpoint không nằm trong image nên Pod có thể
restart mà không mất tiến trình đã ghi.

## Tạo và publish image

```bash
docker build -t DOCKERHUB_USER/vietnamese-spell:bf16 .
docker push DOCKERHUB_USER/vietnamese-spell:bf16
```

Image pin `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, cài `hf` CLI, và
mặc định chạy trainer BF16.

## Thiết lập Pod

1. Chọn image `DOCKERHUB_USER/vietnamese-spell:bf16`.
2. Gắn persistent/network volume tại `/workspace`.
3. Không cần expose port và để trống Start Command.
4. Đặt environment variables sau:

```text
AUTO_DOWNLOAD=1
EPOCHS=3
BATCH_SIZE=64
BUCKET_SIZE=4096
WARMUP_EPOCHS=2
AMP_DTYPE=bf16
OUTPUT_DIR=/workspace/checkpoints/member_NAME_bf16_3_epochs
```

Ở lần chạy đầu, `AUTO_DOWNLOAD=1` tải public dataset
`Sanng1112/vietnamese-spelling-synthetic-1gb` vào `/workspace/data`. Các lần
sau dùng lại volume, không tải lại. Nếu muốn chuẩn bị data thủ công, để
`AUTO_DOWNLOAD=0` rồi chạy:

```bash
hf download Sanng1112/vietnamese-spelling-synthetic-1gb \
  --repo-type dataset --local-dir /workspace/data
```

## Đầu ra cần gửi lại

Sau mỗi epoch, thư mục `OUTPUT_DIR` có `history.jsonl`, `history.json`,
`last.pt` và `best.pt`. Thành viên gửi lại `history.json` cùng `best.pt`; giữ
`last.pt` trên volume để tránh mất checkpoint. Run hiện chưa tự resume từ
`last.pt`, do đó không xoá volume khi chưa sao lưu checkpoint.
