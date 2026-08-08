# Kiến trúc mô hình: từ ký tự sang token cấp từ

Tài liệu này diễn giải Figure 1 và phần 4 của paper *Hierarchical Transformer
Encoders for Vietnamese Spelling Correction*. Trọng tâm là cách thông tin ký tự
đi vào mô hình cấp từ; đây **không** phải là quá trình sửa token từng bước.

## Ý tưởng cốt lõi

Mỗi token có hai nguồn thông tin:

1. **Word embedding**: token này là gì trong vocabulary.
2. **Character representation**: token này được viết bằng những ký tự nào.

Hai vector được ghép lại để tạo một vector duy nhất cho token. Chỉ sau đó,
Word-level Transformer mới dùng ngữ cảnh của toàn câu để quyết định token có
sai hay không và nếu sai thì thay bằng token nào.

```text
ký tự trong một token ──> Character-level Transformer ──> 1 vector ký tự
                                                                   │
word embedding của token ─────────────────────────────────────────┼─> ghép
                                                                   │
                                                                   v
                                                        vector của 1 token
                                                                   │
               các vector token của cả câu ──> Word-level Transformer
```

## Ví dụ chạy qua token `bôj`

Giả sử câu đầu vào là:

```text
Cơn bảo dag đổ bôj vào đất lền .
```

Token thứ năm là `bôj`. Nó được đưa vào hai nhánh **song song**.

### Nhánh word

```text
`bôj` ──> bảng word embedding ──> E_bôj
```

`E_bôj` là vector tra từ bảng vocabulary. Nếu `bôj` là token ngoài vocabulary,
paper dùng `UNK` ở nhánh này. Vì thế nhánh word có thể không biết chi tiết
`bôj` gồm những ký tự nào.

### Nhánh character

```text
`bôj` ──> [b] [ô] [j]
              │   │   │
              v   v   v
       character embeddings + vị trí ký tự
              │
              v
       Character-level Transformer (4 layer, hidden size 256)
              │
              v
       O_bôj = một vector tóm tắt cho toàn bộ `b-ô-j`
```

Trực giác: `O_bôj` có thể mang tín hiệu rằng token này có hình thức gần với
`bộ`, hoặc giống một lỗi Telex. Nó **không** tự xuất ra từ `bộ`; đây mới chỉ là
một biểu diễn số của chuỗi ký tự.

## Bước chuyển từ ký tự sang token

Sau hai nhánh trên, paper ghép hai vector ở cùng một token:

```text
R_bôj = [ E_bôj ; O_bôj ]
```

Trong đó `[ ; ]` nghĩa là nối hai vector, không phải cộng hai từ hay thay token
`bôj` thành một token mới. Có thể coi `R_bôj` là một “thẻ thông tin” phong phú
hơn cho token `bôj`.

```text
E_bôj : “token này là UNK / bôj theo vocabulary”
O_bôj : “nó được viết bằng b - ô - j”
------------------------------------------------
R_bôj : “một token có dạng b-ô-j, có thể là lỗi chính tả”
```

Paper minh họa điều này trong Figure 1 bằng cặp `E_bôj` và `O_c5`: `E_bôj` là
word embedding, còn `O_c5` là output cấp ký tự ứng với token thứ năm.

> Paper không nêu cụ thể cách nén nhiều output ký tự thành đúng một vector
> `O_bôj` (ví dụ mean pooling, hidden state cuối, hay token `[CLS]`). Ta chỉ
> biết từ Figure 1 và mô tả rằng phải có một vector output cấp ký tự tương ứng
> với mỗi token. Đây là chi tiết cần làm rõ nếu tái lập mô hình.

## Đưa các token đã ghép vào Word-level Transformer

Toàn bộ câu giờ là một chuỗi vector, mỗi token đúng một vector:

```text
Cơn   -> R_Cơn   = [E_Cơn   ; O_Cơn]
bảo   -> R_bảo   = [E_bảo   ; O_bảo]
dag   -> R_dag   = [E_dag   ; O_dag]
đổ    -> R_đổ    = [E_đổ    ; O_đổ]
bôj   -> R_bôj   = [E_bôj   ; O_bôj]
vào   -> R_vào   = [E_vào   ; O_vào]
đất   -> R_đất   = [E_đất   ; O_đất]
lền   -> R_lền   = [E_lền   ; O_lền]
```

Chuỗi này đi qua Word-level Transformer (12 self-attention layers). Tại đây,
`R_bôj` có thể attention đến `R_đổ`, `R_vào`, và các token khác. Context
`đổ ... vào` khiến ứng viên `bộ` hợp lý hơn nhiều ứng viên khác.

```text
R_bôj + context("Cơn ... đổ ... vào ...")
                  │
                  v
contextual representation của token thứ năm
```

Tương tự, token `bảo` nhìn được `Cơn`, `dag`, `đổ`, `bôj` và toàn bộ câu,
nên mô hình có thể học mẫu `Cơn bão đang đổ bộ`.

## Dự đoán sau cùng

Với mỗi token, output Word-level Transformer đi vào hai head:

```text
contextual vector của `bôj`
         ├── Detector  -> sai / không sai
         └── Corrector -> phân phối trên vocabulary 60.000 token
```

Ví dụ:

```text
`bôj` -> Detector: sai
      -> Corrector: `bộ`
```

Corrector chỉ chọn một token thay thế từ vocabulary, do đó kiến trúc phù hợp
nhất với sửa lỗi 1 token sang 1 token:

```text
bôj -> bộ
dag -> đang
```

Nó không tự nhiên xử lý các trường hợp đổi số token, chẳng hạn:

```text
khongbiet -> không biết
tp hcm    -> Thành phố Hồ Chí Minh
```

## Không có vòng phản hồi sau khi sửa

Các token không được sửa tuần tự theo dạng:

```text
sửa `bảo` -> đưa `bão` lại vào mô hình -> sửa `dag`
```

Thay vào đó, một lần forward pass sử dụng toàn bộ **câu lỗi gốc**. Self-attention
cho phép mọi token nhìn các token khác theo hai chiều; sau đó Detector và
Corrector dự đoán cho mọi vị trí gần như đồng thời.

```text
Cơn <--> bảo <--> dag <--> đổ <--> bôj <--> vào <--> đất <--> lền
```

Mũi tên trên chỉ là trực giác: attention thực tế cho phép một token kết nối trực
tiếp với mọi token trong câu, không chỉ hai hàng xóm.

## Input: mô hình thực sự nhận gì?

Model không nhận trực tiếp một Python string. Trước khi gọi `models.py`, cần có
hai vocabulary do dữ liệu train tạo ra:

```text
word_vocab: ký tự/chuỗi token -> word id
char_vocab: ký tự đơn        -> char id
```

Ví dụ minh họa (các id chỉ để dễ nhìn, không phải id thật):

```text
word_vocab
  Cơn -> 41       bôj -> UNK -> 1       vào -> 78

char_vocab
  b -> 12         ô -> 19               j -> 15
```

Với câu:

```text
Cơn bảo dag đổ bôj vào đất lền .
```

bước tiền xử lý là:

```text
1. Tách trắng:
   [Cơn] [bảo] [dag] [đổ] [bôj] [vào] [đất] [lền] [.]

2. Đổi token thành word id:
   word_ids = [41, 1, 1, 63, 1, 78, 90, 1, 4]
                    ^     ^      ^          ^
                    token ngoài vocabulary dùng UNK (= 1)

3. Tách từng token thành ký tự rồi đổi thành character id:
   char_ids[4] cho `bôj` = [12, 19, 15, 0, 0, ...]
                               ^  ^  ^  ^
                               b  ô  j  PAD
```

Trong code, batch input có bốn tensor chính:

| Tensor | Shape | Nội dung |
|---|---|---|
| `word_ids` | `(B, T)` | id của token; `B` là số câu/batch, `T` là số token tối đa mỗi câu |
| `char_ids` | `(B, T, C)` | id ký tự trong từng token; `C` là số ký tự tối đa/token |
| `attention_mask` | `(B, T)` | `True` tại token thật, `False` tại phần PAD cuối câu |
| `char_attention_mask` | `(B, T, C)` | `True` tại ký tự thật, `False` tại PAD ký tự |

Với cấu hình mặc định của paper/implementation:

```text
T <= 192 token/câu
C <= 32 ký tự/token (lựa chọn triển khai; paper không nêu giới hạn này)
```

`PAD` chỉ dùng để đưa các câu/token có độ dài khác nhau vào cùng một batch.
Mask đảm bảo attention và loss không coi padding là dữ liệu thật.

## Cấu hình Compact-1M

Hàm `compact_1m_config()` trong `models.py` tạo một cấu hình khoảng **1.0 triệu
tham số**, phù hợp để bắt đầu train trên RTX 4050 Laptop 6 GB VRAM. Nó giữ ý
tưởng chính của paper, nhưng không cố sao chép kích thước BERT-base rất lớn.

| Thành phần | Paper | Compact-1M | Lý do |
|---|---:|---:|---|
| Word vocabulary | 60.000 | 9.500 | Paper nêu 7.184 âm tiết phổ biến; phần dư chứa PAD/UNK, dấu câu và token phổ biến |
| Word embedding | không nêu rõ | 64 | Vocabulary là nguồn tham số lớn nhất; corrector dùng chung trọng số này |
| Character encoder | 4 layer × 256 | 2 layer × 64 | Vẫn giữ tín hiệu typo/Telex/dấu với chi phí nhỏ |
| Word encoder | 12 layer × 768 | 4 lượt × 128 | Giảm kích thước để train laptop |
| Word-layer sharing | dùng ALBERT, chi tiết không rõ | có | Một block 128-d được chạy 4 lần với cùng trọng số, giữ độ sâu nhưng giảm tham số |
| Detector hidden size | không nêu rõ | 64 | Đủ cho nhiệm vụ nhị phân đúng/sai |

Điểm quan trọng về sharing:

```text
Không sharing:  Block A -> Block B -> Block C -> Block D
                 (4 bộ trọng số khác nhau)

Compact-1M:    Block A -> Block A -> Block A -> Block A
                 (1 bộ trọng số, chạy 4 lần)
```

Như vậy model vẫn có bốn bước trộn ngữ cảnh giữa các token, nhưng chỉ lưu một
Word Transformer block. Đây là cách phù hợp với việc paper nói implementation
dựa trên ALBERT.

Phân rã tham số ước tính của preset này:

```text
word embedding (9.500 × 64, được weight-tie)    ~608k
shared word Transformer block (128 hidden)      ~198k
character encoder + char embeddings              ~128k
fusion, detector, projection, position, bias      ~67k
------------------------------------------------------
tổng                                             ~1,001k
```

Khởi tạo preset:

```python
from models import HierarchicalSpellingCorrector, compact_1m_config

config = compact_1m_config()
model = HierarchicalSpellingCorrector(config)
```

Đây là mô hình để thử nghiệm/học pipeline, không nên kỳ vọng tự động đạt F1
68.88% như cấu hình paper lớn hơn khoảng hai bậc độ lớn. Vì vocabulary nhỏ,
mọi token lỗi hiếm sẽ thường đi qua nhánh `UNK` ở word level; Character
Transformer vẫn giữ chuỗi ký tự gốc. Token mục tiêu cần sửa phải nằm trong
vocabulary 9.500 từ, nếu không corrector không thể sinh nó.

## Target khi train

Mỗi câu lỗi phải đi kèm câu đúng, đã căn thẳng theo token. Ví dụ:

```text
input : Cơn bảo dag đổ bôj vào đất lền .
target: Cơn bão  đang đổ bộ vào đất liền .
```

Từ đó tạo nhãn:

```text
detection_labels = [0, 1, 1, 0, 1, 0, 0, 1, 0]
                    0 = token đúng; 1 = token sai

correction_labels = [IGNORE, id(bão), id(đang), IGNORE, id(bộ),
                     IGNORE, IGNORE, id(liền), IGNORE]
```

`models.py` tính:

```text
detector loss  : áp dụng cho toàn bộ token thật
corrector loss : chỉ áp dụng tại nơi detection_labels = 1
total loss     = detector loss + corrector loss
```

Điều kiện căn thẳng theo token rất quan trọng. Nếu lỗi làm thay đổi số token,
ví dụ `khongbiet -> không biết`, cách gán nhãn trên không còn phù hợp; đây là
hạn chế trực tiếp của kiến trúc 1-token-sang-1-token.

## Output: mô hình trả gì?

Sau một forward pass, `models.py` trả về hai tensor logits:

| Output | Shape | Cách đọc |
|---|---|---|
| `detection_logits` | `(B, T, 2)` | hai điểm số cho `[đúng, sai]` tại mỗi token |
| `correction_logits` | `(B, T, V)` | điểm số cho toàn bộ `V` token trong word vocabulary |

Ví dụ tại vị trí `bôj`:

```text
detection_logits[0, 4] = [ -1.2, 3.7 ]
                         đúng   sai
                         => argmax là `sai`

correction_logits[0, 4] = ...
                         id(bộ): 8.5
                         id(bơi): 2.1
                         id(bốn): 0.9
                         ...
                         => argmax là id(bộ)
```

Hàm `correct()` làm ba việc:

```text
1. detector chọn vị trí bị lỗi;
2. corrector chọn word id thay thế cho từng vị trí;
3. chỉ thay thế nơi detector dự đoán lỗi, rồi đổi id về chuỗi bằng word_vocab.
```

```text
input ids     : [id(Cơn), id(bảo), id(dag), id(đổ), id(bôj), ...]
error flags   : [false,   true,    true,    false,   true,       ...]
suggestions   : [id(...), id(bão), id(đang), id(...), id(bộ),    ...]
corrected ids : [id(Cơn), id(bão), id(đang), id(đổ), id(bộ),    ...]
```

Cuối cùng, chương trình bên ngoài model phải map `corrected_ids` ngược về
token và nối chúng bằng khoảng trắng. `models.py` cố ý không tự làm vocabulary
hay tokenize tiếng Việt vì hai thành phần đó phụ thuộc dữ liệu train của bạn.

## Luồng hoạt động đầy đủ

```text
string lỗi
  │
  ├─ tokenizer theo khoảng trắng
  ├─ word_vocab: token -> word_ids
  └─ char_vocab: từng token -> char_ids
                           │
                           v
                  Character Transformer, độc lập trong từng token
                           │
                           v
                       O_1, O_2, ..., O_T
                           │
word embeddings E_1, ..., E_T
                           │
                           v
            ghép [E_i ; O_i] cho mọi token i
                           │
                           v
              Word Transformer, attention trên cả câu
                           │
                           ├─ Detector: đúng / sai
                           └─ Corrector: token thay thế trong vocabulary
                                       │
                                       v
                              các token câu đã sửa
```

Mọi token được dự đoán trong **cùng một lượt**. Không có bước sửa `bảo` thành
`bão`, đưa `bão` trở lại input, rồi mới sửa `dag`; vì vậy câu gốc bị lỗi là ngữ
cảnh duy nhất cho toàn bộ quyết định.

## Hiệu năng thực tế: chất lượng

Theo Table 2 của paper, trên Wiki spelling test set (khoảng 1.500 lỗi thật),
model hierarchical Transformer đạt:

| Chỉ số | Kết quả | Ý nghĩa |
|---|---:|---|
| Detector precision | 66.96% | Trong các token bị báo sai, khoảng 2/3 thật sự sai |
| Detector recall | 70.92% | Bắt được khoảng 71% lỗi thật |
| Detector F1 | 68.88% | Cân bằng precision và recall khi phát hiện lỗi |
| Corrector, trong lỗi đã phát hiện | 96.01% | Khi detector đã báo đúng vị trí lỗi, đề xuất thay thế thường chính xác |

So với Transformer chỉ dùng word-level, F1 detector tăng từ 45.97% lên 68.88%.
Đây là bằng chứng rõ nhất rằng nhánh ký tự hữu ích cho lỗi chính tả tiếng Việt.

Paper gọi 64.29% là `Correction accuracy in total`, nhưng cần đọc cẩn thận:

```text
64.29% = 66.96% detector precision × 96.01% corrector accuracy khi đã phát hiện
```

Nó không phải tỷ lệ sửa đúng trên tất cả lỗi thật vì không tính các lỗi bị bỏ
sót trong mẫu số. Một ước lượng recall sửa lỗi đầu-cuối hợp lý hơn là:

```text
70.92% detector recall × 96.01% corrector accuracy = 68.09%
```

Nghĩa là trong điều kiện benchmark này, model có thể sửa đúng khoảng 68 trên
100 lỗi thật; con số này là suy luận từ các metric paper công bố, không phải
một metric được tác giả báo cáo trực tiếp.

Trên subtitle bị xóa toàn bộ dấu, paper báo F1 detector 99.75% và accuracy sửa
trong các lỗi phát hiện là 98.50%. Đây chủ yếu đo bài toán khôi phục dấu, vốn
đồng nhất và dễ hơn lỗi tự nhiên hỗn hợp; không nên coi nó là chất lượng chung
cho mọi loại spelling error trong chat thực tế.

## Hiệu năng thực tế: tốc độ và giới hạn

Kiến trúc này thường nhanh hơn một seq2seq autoregressive vì nó dự đoán tất cả
token song song trong một forward pass. Tuy nhiên paper **không công bố** số
ms/câu, throughput, GPU/CPU, batch size inference, hay số tham số; vì thế không
thể kết luận một độ trễ triển khai cụ thể từ paper.

Về độ phức tạp, mô hình có hai phần chính:

```text
Character Transformer: xử lý từng token riêng; chi phí attention xấp xỉ O(T × C²)
Word Transformer:      xử lý cả câu;      chi phí attention xấp xỉ O(T²)
```

với `T` là số token trong câu và `C` là số ký tự/token. Ở giới hạn `T = 192`,
Word-level Transformer 12 layer, hidden size 768 thường là phần nặng nhất.
Hiệu năng latency thực tế phụ thuộc mạnh vào GPU/CPU, độ dài câu, batch,
framework và vocabulary size 60.000 của corrector.
