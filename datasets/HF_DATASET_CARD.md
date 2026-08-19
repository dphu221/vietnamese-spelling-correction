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

Corpus tổng hợp cho bài toán phát hiện và sửa lỗi chính tả tiếng Việt, trích xuất và làm sạch từ Vietnamese Wikipedia Dump mới nhất, hỗ trợ mô hình phân loại token 1-đối-1.

## 📊 Nội dung Dataset

| File | Số lượng mẫu (Rows) | Dung lượng thô | Mô tả |
|---|---:|---:|---|
| `train_full.jsonl` | 4.564.360 | 6.04 GB | Tập huấn luyện chính thức |
| `validation_full.jsonl` | 95.094 | 124 MB | Tập validation held-out (tách theo `page_id`) |
| `word_vocab_full.json` | 9.500 entries | 174 KB | Từ vựng từ chuẩn cho mô hình |
| `char_vocab_full.json` | 400 entries | 5.5 KB | Từ vựng ký tự chuẩn |

Mỗi mẫu dữ liệu duy trì căn chỉnh **một-token-sang-một-token** giữa chuỗi bị làm nhiễu (`noisy_tokens`) và chuỗi chuẩn (`clean_tokens`).

---

## 📝 Định dạng dữ liệu (JSON Format)

```json
{
  "id": "1a2b3c4d5e6f7g8h9i0j",
  "noisy_text": "Hôm nay trời ko mưa , tui đi chs .",
  "clean_text": "Hôm nay trời không mưa , tôi đi chơi .",
  "noisy_tokens": ["Hôm", "nay", "trời", "ko", "mưa", ",", "tui", "đi", "chs", "."],
  "clean_tokens": ["Hôm", "nay", "trời", "không", "mưa", ",", "tôi", "đi", "chơi", "."],
  "detection_labels": [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
  "error_types": ["none", "none", "none", "abbreviation", "none", "none", "abbreviation", "none", "abbreviation", "none"]
}
```

---

## 🛠️ Thống kê 10 loại Nhiễu tổng hợp (Synthetic Noise Distribution)

Dữ liệu tổng hợp **11.029.010 vị trí lỗi** được chèn ngẫu nhiên và đã qua kiểm thực (`verify` passed):

| Loại lỗi (`error_types`) | Số lượng lỗi | Tỷ lệ % | Mô tả ví dụ |
|---|---:|---:|---|
| `keyboard` | 2.076.452 | 18.8% | Gõ nhầm phím kề QWERTY (ví dụ: `phòng` $\rightarrow$ `phongg`) |
| `telex` | 1.895.928 | 17.2% | Gõ dư/sai phím Telex (ví dụ: `bão` $\rightarrow$ `baox`, `bộ` $\rightarrow$ `booj`) |
| `delete` | 1.635.363 | 14.8% | Xóa ký tự ngẫu nhiên trong từ (ví dụ: `ngôn` $\rightarrow$ `ngn`) |
| `transpose` | 1.403.313 | 12.7% | Tráo đổi 2 ký tự liền kề (ví dụ: `phòng` $\rightarrow$ `póhng`) |
| `drop_diacritic` | 1.182.701 | 10.7% | Mất toàn bộ dấu thanh và dấu phụ (ví dụ: `đường` $\rightarrow$ `duong`) |
| `drop_tone` | 1.151.346 | 10.4% | Mất dấu thanh (ví dụ: `bão` $\rightarrow$ `bao`) |
| `vni` | 1.007.055 | 9.1% | Lỗi gõ phím VNI (ví dụ: `bão` $\rightarrow$ `bao4`) |
| `consonant` | 460.189 | 4.2% | Lẫn phụ âm đầu (`tr/ch`, `s/x`, `d/gi`, `r/gi`, `n/l`) |
| `swap_tone` | 173.215 | 1.6% | Tráo dấu hỏi và dấu ngã (`bão` $\leftrightarrow$ `bảo`, `nghĩ` $\leftrightarrow$ `nghỉ`) |
| `abbreviation` | 43.448 | 0.4% | Teencode / Viết tắt 1-1 (`k`/`ko` $\rightarrow$ `không`, `chs` $\rightarrow$ `chơi`, `nx` $\rightarrow$ `nữa`) |

---

## 🛡️ Kiểm soát chất lượng & Chống rò rỉ (Data Leakage)

- **Ngăn chặn rò rỉ dữ liệu test**: Toàn bộ `page_id` của bộ benchmark công khai **Viwiki-Spelling** đã được loại bỏ trước khi trích xuất câu từ Wikipedia dump.
- **Lọc chất lượng câu**: Chỉ giữ câu có độ dài 5–192 token (24–900 ký tự), tỷ lệ chữ cái $\ge 52\%$, chứa ký tự tiếng Việt, tỷ lệ ký tự đặc biệt $< 1.5\%$, deduplicate SHA-256.
- **Phân chia Train/Validation theo Bài viết**: Tách câu dựa trên hash SHA-256 của `page_id` (đảm bảo 0% overlap bài viết giữa Train và Validation).

---

## 📜 Nguồn và Giấy phép

Nguồn văn bản sạch là Vietnamese Wikipedia Dump. Dataset dẫn xuất này được phân phối theo giấy phép **CC BY-SA 4.0**.
