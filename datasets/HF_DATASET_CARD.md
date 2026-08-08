---
license: cc-by-sa-4.0
language:
- vi
tags:
- vietnamese
- spelling-correction
- synthetic-data
---

# Vietnamese Spelling Correction Synthetic 1GB

Corpus tổng hợp cho bài toán phát hiện và sửa lỗi chính tả tiếng Việt, sinh từ
Vietnamese Wikipedia đã làm sạch.

## Nội dung

| File | Nội dung |
|---|---|
| `train_full.jsonl` | 4.564.618 mẫu huấn luyện |
| `validation_full.jsonl` | 94.836 mẫu validation, tách theo `page_id` |
| `word_vocab_full.json` | Vocabulary 9.500 token từ |
| `char_vocab_full.json` | Vocabulary 400 ký tự |

Mỗi record giữ căn chỉnh một-token-sang-một-token và có `noisy_tokens`,
`clean_tokens`, `detection_labels`, cùng `error_types`.

## Nhiễu tổng hợp

Telex, VNI, mất thanh, mất toàn bộ dấu, phím kề, xóa ký tự, đảo ký tự và nhầm
phụ âm đầu. Corpus không chứa benchmark Viwiki-Spelling; các trang benchmark
đã bị loại trước khi trích xuất, tránh leakage.

## Nguồn và license

Nguồn câu sạch là Vietnamese Wikipedia. Dataset dẫn xuất này phân phối theo
CC BY-SA 4.0. Benchmark Viwiki-Spelling có license khác và không nằm trong repo
này.
