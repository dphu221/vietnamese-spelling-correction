# Datasets: Vietnamese spelling correction

Pipeline này tạo dữ liệu huấn luyện synthetic, căn thẳng một-token-sang-một-token cho `../models.py`. Nó chỉ dùng nguồn có giấy phép rõ ràng, giữ provenance của Wikipedia, và giữ Viwiki-Spelling hoàn toàn ngoài train/validation.

## Nguồn dữ liệu

| Vai trò | Nguồn | License | Cách dùng |
|---|---|---|---|
| Test cuối cùng | [Viwiki-Spelling](https://github.com/heraclex12/Viwiki-spelling) | CC BY 4.0 | Chỉ evaluation; 107 document có lỗi gán nhãn thủ công |
| Corpus sạch | [Vietnamese Wikipedia dump](https://dumps.wikimedia.org/viwiki/latest/) | CC BY-SA | Train/validation sau khi lọc markup và loại page id trùng test |

Không scrape báo chí hoặc phụ đề có bản quyền. Có thể bổ sung corpus do bạn có quyền sử dụng; record đầu vào phải có trường `text` và `source_id` ổn định.

## Chạy pipeline

```bash
python3 datasets/build_dataset.py download-test
python3 datasets/build_dataset.py prepare-benchmark
python3 datasets/build_dataset.py download-wikipedia
# Dừng sau khi tạo ít nhất 2 GiB JSONL corpus sạch (không tính dump nén).
python3 datasets/build_dataset.py extract-wikipedia --max-output-gib 2
python3 datasets/build_dataset.py generate
python3 datasets/build_dataset.py build-vocab --size 9500
python3 datasets/build_dataset.py verify
```

Để kiểm tra nhanh trước khi xử lý toàn bộ dump:

```bash
# Corpus seed trực tiếp từ MediaWiki API chính thức, không cần chờ dump 1.14 GB.
python3 datasets/build_dataset.py collect-wikipedia-api --pages 1000
python3 datasets/build_dataset.py generate \
  --input datasets/interim/clean_viwiki_api.jsonl --max-examples 200000
```

API seed phù hợp để kiểm tra/train thử. Với dataset train chính thức, dùng XML dump vì nó tái lập được và bao phủ corpus rộng hơn.

`generate` ghi đè các file output cùng tên. Viwiki-Spelling không được dùng để train, validation hoặc build vocabulary.

`prepare-benchmark` chỉ giữ các lỗi sửa được bằng đúng một token, vì corrector hiện tại là classifier 1-token-sang-1-token. Với bản benchmark đang public, bước này tạo 917 chunks với 1.486 vị trí lỗi đánh giá được; các lỗi dính/tách từ hoặc suggestion rỗng được giữ trong raw benchmark nhưng không thể chấm bằng kiến trúc hiện tại.

## Định dạng output

```json
{
  "noisy_text": "Cơn bảo dag đổ bôj vào đất lền .",
  "clean_text": "Cơn bão đang đổ bộ vào đất liền .",
  "noisy_tokens": ["Cơn", "bảo", "dag", "đổ", "bôj", "vào", "đất", "lền", "."],
  "clean_tokens": ["Cơn", "bão", "đang", "đổ", "bộ", "vào", "đất", "liền", "."],
  "detection_labels": [0, 1, 1, 0, 1, 0, 0, 1, 0],
  "error_types": ["none", "drop_tone", "keyboard", "none", "telex", "none", "none", "drop_tone", "none"]
}
```

Các lỗi nhân tạo gồm Telex, VNI, mất thanh, mất toàn bộ dấu, thay phím kề, xóa ký tự, đảo ký tự, và lẫn phụ âm đầu (`tr/ch`, `s/x`, `d/gi`, `n/l`). Tất cả giữ nguyên số token, tương thích corrector phân loại của paper.

## Kiểm soát chất lượng

- Chỉ lấy namespace bài viết chính của Wikipedia và bỏ redirect.
- Xóa HTML, ref, table, link, template; bỏ câu còn markup/ký hiệu dày đặc.
- Lọc câu 5–192 token, có tín hiệu tiếng Việt, và deduplicate theo hash.
- Split theo `page_id`, không để câu cùng một bài ở cả train và validation.
- Loại toàn bộ `page_id` của Viwiki-Spelling trước khi extraction.
- Giữ `source`, `page_id`, `title`; chạy `verify` để kiểm token alignment và label.
