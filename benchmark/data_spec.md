# 결함 이미지 메타데이터 스펙

## 배경

이 벤치마크는 CraigslistBargain 스타일 중고거래 협상에서, agent가 **이미지 속 실제
결함을 얼마나 정확히 찾아내고 협상에 활용하는지**를 측정합니다. 이를 위해 아이템마다
"결함이 합성된 이미지 + 그 결함의 ground-truth 메타데이터"가 필요합니다. 이 문서는
그 메타데이터를 어떤 형식으로 준비해주면 되는지 정리한 스펙입니다.

## 파일 형식

아이템 하나당 아래 스키마를 따르는 JSON 객체 하나. 여러 개면 JSON 배열로 묶거나,
한 줄에 객체 하나씩(JSONL)으로 주시면 됩니다.

## 필드 스펙

### Item (아이템 최상위)

| 필드                 | 타입          | 필수 | 설명                                                                         |
| -------------------- | ------------- | ---- | ---------------------------------------------------------------------------- |
| `item_id`            | string        | ✅   | 고유 식별자                                                                  |
| `category`           | string        | ✅   | 예: `"furniture"`                                                            |
| `title`              | string        | ✅   | 리스팅 제목                                                                  |
| `description`        | string        | ✅   | 리스팅 설명 — **결함을 절대 언급하지 말 것** (아래 "지켜야 할 것" 참고)      |
| `listing_price`      | number        | ✅   | 원 판매 희망가                                                               |
| `image_ref`          | string        | ✅   | 결함 합성 **후** 이미지 파일 경로                                            |
| `image_ref_original` | string        | 권장 | 결함 합성 **전** 원본 이미지 경로 (검증/비교용, 나중에 필요해질 가능성 높음) |
| `defects`            | array[Defect] | ✅   | 아래 Defect 스키마의 배열. 0개(결함 없는 아이템)도 허용                      |

### Defect (`defects` 배열의 원소)

| 필드           | 타입   | 필수 | 설명                                                                   |
| -------------- | ------ | ---- | ---------------------------------------------------------------------- |
| `id`           | string | ✅   | 이 아이템 안에서만 고유하면 됨 (전역 고유 불필요)                      |
| `description`  | string | ✅   | 결함의 사실관계 (예: `"3cm tear on the left armrest fabric"`)          |
| `price_impact` | number | ✅   | 이 결함이 공정가에서 깎아야 하는 금액 (listing_price와 같은 통화 단위) |

> `salience`(육안 발견 난이도) 필드는 **이번 8주 학술제 스코프에서 뺐습니다**
> (2026-07-11 결정) — 지금 당장은 안 보내주셔도 됩니다. 아래 "향후 계획" 참고.

## 향후 계획 (스코프 밖) — `salience`

원래 "합성 품질이 낮아서 VLM이 못 찾은 건데 벤치마크가 감점하는" confound를
막으려고 `salience`(0~1, 육안 발견 난이도) 필드를 넣으려 했는데, 제대로 재려면:

- 결함을 직접 합성한 사람이 아닌 **제3의 독립 코더 3인 이상**이
- 결함이 어디 있는지/어떤 종류인지 **모르는 블라인드 상태**로 이미지를 보고
  스스로 찾아내야 하고
- 그 결과의 신뢰도를 ICC(intraclass correlation)로 확인해야 함

이건 이번 8주 스코프 밖의 별도 작업이라 지금은 보류하는게 좋을 것 같습니다. (연구자 본인이나
합성한 사람이 "이 정도면 잘 보이겠지"라고 혼자 매기면, 이미 결함 위치를 아는
상태라 판단이 편향돼서 애초에 confound를 막으려던 목적 자체가 무의미해짐 —
그래서 "일단 대충 혼자 매기기"조차 지금 안 하는 쪽으로 결정.)

## 지켜야 할 것 (중요)

1. **`description`(리스팅 설명)에 결함을 절대 언급하지 말 것.** 이미지를 봐야만
   결함을 알 수 있어야 "시각 증거 활용 역량을 측정한다"는 벤치마크 전제가
   성립합니다. 만약에 실제 CraigslistBargain description 중 결함을 언급하는 것이 있다면 자연스럽게 제거하는 것이 필요할 것 같습니다.
2. **결함 합성 전/후 이미지를 둘 다 보관**해주세요. 합성 후 이미지(`image_ref`)만
   있으면 나중에 "결함이 실제로 잘 보이게 합성됐는지" 검증할 방법이 없습니다.
3. `defects`가 빈 배열(결함 없는 정상 아이템)인 경우도 섞어주시면 좋습니다 —
   "없는 결함을 지어내는지"(hallucination) 테스트에 필요합니다.

## 예시

```json
{
  "item_id": "sofa_001",
  "category": "furniture",
  "title": "Grey mid-century sofa",
  "description": "Comfortable 3-seat sofa, gently used, pickup only.",
  "listing_price": 250.0,
  "image_ref": "images/sofa_001_defect.jpg",
  "image_ref_original": "images/sofa_001_original.jpg",
  "defects": [
    {
      "id": "tear_0",
      "description": "3cm tear on the left armrest fabric",
      "price_impact": 15.0
    },
    {
      "id": "stain_1",
      "description": "Coffee stain on the seat cushion",
      "price_impact": 8.0
    }
  ]
}
```

## 참고

- 스키마 대응 코드: `benchmark/env.py`의 `Item`/`Defect` dataclass
  (`benchmark/env.py:34-66` 근처)
- 이 데이터를 실제로 연동하는 loader 함수는 아직 미구현
