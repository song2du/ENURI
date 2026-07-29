"""results.jsonl(JSON 배열 또는 JSONL)을 읽어 env.py의 Item/Defect로 변환.

실 데이터 레코드 형식:
- Item: item_id/category/title/description/listing_price/image_ref/image_ref_original/defects
- Defect(defects 배열 원소, 0~1개): id/description/defect_type/visibility/alignment/severity

여기서 하는 변환:
- `visibility`+`alignment` -> `quadrant` 문자열 (예: "obvious"+"aligned" -> "obvious_aligned")
- `defect_type`+`listing_price`로 `price_impact_for()` 호출 -> `price_impact` 계산
  (레코드엔 price_impact이 없음, defect_type 기반으로 여기서 계산)
- `severity`는 그대로 보관하되 계산엔 안 씀 (env.py 참고)
- `item_id`/`image_ref_original`은 Item에 대응 필드가 없어 파싱만 하고 버림
"""

from __future__ import annotations

import json
from pathlib import Path

from env import Defect, Item, price_impact_for


def _defect_from_dict(d: dict, listing_price: float, mapping: dict[str, float] | None) -> Defect:
    quadrant = f"{d['visibility']}_{d['alignment']}"
    return Defect(
        id=d["id"],
        description=d["description"],
        price_impact=price_impact_for(listing_price, d["defect_type"], mapping),
        quadrant=quadrant,
        severity=d.get("severity"),  # 정보 보관용, 계산엔 미사용
        defect_type=d["defect_type"],
    )


def _item_from_dict(d: dict, base_dir: Path, mapping: dict[str, float] | None) -> Item:
    listing_price = float(d["listing_price"])
    defects = tuple(
        _defect_from_dict(defect, listing_price, mapping) for defect in d.get("defects", [])
    )
    return Item(
        category=d["category"],
        title=d["title"],
        description=d["description"],
        listing_price=listing_price,
        image_ref=str((base_dir / d["image_ref"]).resolve()),  # jsonl 기준 상대경로 -> 절대경로
        ground_truth_defects=defects,
        # item_id/image_ref_original: Item에 대응 필드 없음, 파싱만 하고 버림
    )


def load_items(path: str | Path, mapping: dict[str, float] | None = None) -> list[Item]:
    """JSON 배열 또는 JSONL 파일을 읽어 Item 리스트로 변환.

    형식 자동 감지: 첫 non-whitespace 문자가 '['면 배열, 아니면 줄 단위 JSONL.
    mapping: env.py의 PRICE_IMPACT_MAPPINGS 중 하나. None이면 기본값(B_moderate) 사용 --
    매핑 로버스트니스 실험에서 mapping만 바꿔가며 재호출.
    """
    path = Path(path)
    base_dir = path.parent
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [_item_from_dict(r, base_dir, mapping) for r in records]
