"""data_spec.md 스펙(JSON 배열 또는 JSONL)을 읽어 env.py의 Item/Defect로 변환.

benchmark/CLAUDE.md 2026-07-12 항목에서 미구현으로 남겨뒀던 로더 -- 팀원이 data_spec.md
포맷으로 결함-합성 이미지 메타데이터를 넘겨주면, 이 파일로 실제 Item 리스트를 만든다.

data_spec.md의 item_id/image_ref_original 필드는 env.py의 Item dataclass에 대응하는
슬롯이 없다 (지금 아무 코드도 이 두 값을 읽지 않음) -- 로더가 파싱은 하되 Item에는 안
옮긴다. data_spec.md 자체가 "나중에 필요해질 가능성 높음"이라고 명시했으니, 실제로
필요해지면 Item에 필드를 추가하고 여기서 이어붙이면 된다 (지금 미리 추가하지 않는 이유:
아무도 안 쓰는 필드를 미리 만들지 않는다는 프로젝트 원칙).
"""

from __future__ import annotations

import json
from pathlib import Path

from env import Defect, Item


def _defect_from_dict(d: dict) -> Defect:
    return Defect(
        id=d["id"],
        description=d["description"],
        price_impact=float(d["price_impact"]),
        # quadrant/salience: data_spec.md 스펙에 아직 없는 필드 -- 팀원 데이터에
        # 나중에 추가되면 여기서 d.get(...)으로 채우면 됨. 지금은 항상 기본값(None).
    )


def _item_from_dict(d: dict) -> Item:
    defects = tuple(_defect_from_dict(defect) for defect in d.get("defects", []))
    return Item(
        category=d["category"],
        title=d["title"],
        description=d["description"],
        listing_price=float(d["listing_price"]),
        image_ref=d["image_ref"],
        ground_truth_defects=defects,
        # item_id/image_ref_original: 위 모듈 docstring 참고, 지금은 파싱만 하고 버림.
    )


def load_items(path: str | Path) -> list[Item]:
    """data_spec.md 스펙 파일 하나(JSON 배열 또는 JSONL)를 읽어 Item 리스트로 변환.

    형식은 자동 감지: 파일의 첫 non-whitespace 문자가 '['면 JSON 배열 전체를 한 번에
    파싱하고, 아니면 한 줄에 객체 하나씩(JSONL)이라고 보고 줄 단위로 파싱한다
    (data_spec.md "여러 개면 JSON 배열로 묶거나, 한 줄에 객체 하나씩(JSONL)으로 주시면
    됩니다" 규칙 그대로).
    """
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [_item_from_dict(r) for r in records]
