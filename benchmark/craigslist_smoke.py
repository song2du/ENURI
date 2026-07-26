"""CraigslistBargain 실제 데이터로 VLM 배관(이미지+forced tool call) 스모크테스트.

benchmark/CLAUDE.md 2026-07-23 결정: 팀원의 결함-합성 이미지가 아직 없어서, 실제
grounding(citation_precision 등)은 검증 불가 -- 대신 "이미지를 실제로 보내면 GPT가
받아서 처리하는가"만 CraigslistBargain의 진짜 텍스트+이미지로 먼저 확인한다.

데이터 출처 (둘 다 직접 curl로 검증 완료, 2026-07-23):
- 텍스트(title/description/price/category/post_id): stanfordnlp/cocoa GitHub repo의
  scraper JSON (craigslistbargain/scraper/data/negotiation/*.json)에서 그대로 가져옴.
- 이미지: 원본 scraper JSON의 image_urls(images.craigslist.org CDN)는 죽어있음 --
  2018년 스크래핑이라 만료됨, 30개 샘플 전부 404 확인. 대신 cocoa README가 링크한
  Codalab 아카이브(bundle 0xb93730d80e1c4d4cb4c6bf7c9ebef12f, "images accompanying the
  original Craigslist posts", 286MB)에서 {category}/{post_id}_0.jpg 경로로 실제
  이미지를 받을 수 있음을 확인함 (furniture/phone/electronics 각 1개씩 HTTP 200 확인).

Defect ground truth는 전혀 없음(ground_truth_defects=()) -- CraigslistBargain은 결함
데이터셋이 아니라 그냥 실제 중고거래 게시물이라서. 그래서 이 스크립트로는
citation_precision 등 grounding metric을 검증할 수 없고, "이미지가 실제로 전달되고
GPT가 그걸 보고 반응하는지"만 확인하는 용도다. reservation/price bounds는 실제
listing_price 비율로 대충 구성한 것 -- 협상 역학 자체를 정밀 검증하려는 게 아니라
이미지 배관만 보려는 것이라 정교할 필요 없음.
"""

from __future__ import annotations

import random

from env import Episode, Item, Role, TypeB, run_episode
from kernel import make_counterpart_policy
from llm_agent import make_llm_agent_policy
from voice import add_voice

_IMAGE_BASE = "https://worksheets.codalab.org/rest/bundles/0xb93730d80e1c4d4cb4c6bf7c9ebef12f/contents/blob"

# (category, post_id, title, description, price) -- 전부 실제 cocoa scraper JSON 값 그대로
# (2026-07-23 확인). image_urls[0]은 죽어있어서 대신 Codalab 아카이브 경로를 조합해 쓴다.
_REAL_ITEMS = [
    (
        "furniture",
        "6122156120",
        "Mid-century Modern Blonde Wood Dining Table - Delivery Avail",
        "Very good condition light blonde wood mid century modern dining table with a unique pair of "
        "curved five-pole legs, also comes with two leaf extensions which are made intentionally lighter "
        "in color. Stylish, sturdy and quite heavy, this is a strong very well made table that will last "
        "for years to come. Ready for pickup in San Leandro or I can deliver locally (gas $ helps!) Thanks!",
        180.0,
    ),
    (
        "electronics",
        "6152016631",
        "Acoustic Response Stereo Speakers",
        "Pair of unused Acoustic Response 707 series home stereo system speakers. Great sound for the "
        "price. New in boxes. Pair for $200 obo.",
        200.0,
    ),
]


def make_real_item(category: str, post_id: str, title: str, description: str, price: float) -> Item:
    return Item(
        category=category,
        title=title,
        description=description,
        listing_price=price,
        image_ref=f"{_IMAGE_BASE}/{category}/{post_id}_0.jpg",
        ground_truth_defects=(),  # CraigslistBargain엔 결함 ground truth 자체가 없음
    )


def main() -> None:
    rng = random.Random(0)
    for category, post_id, title, description, price in _REAL_ITEMS:
        item = make_real_item(category, post_id, title, description, price)
        episode = Episode(
            regime="overlap",
            p_min=price * 0.5,
            p_max=price * 1.1,
            role_A=Role.BUYER,
            r_A=price * 0.75,
            t_B=TypeB(r=price * 0.6, urgency=0.5, stance="neutral"),
            opener="AgentOpens",
            K=6,
            harshness=0.5,
            item=item,
        )
        agent = make_llm_agent_policy(model="gpt-4o")
        counterpart = add_voice(make_counterpart_policy("Candid"))

        print(f"=== {item.title} (${item.listing_price:.2f}) ===")
        print(f"image: {item.image_ref}")
        result = run_episode(episode, agent, counterpart, rng)
        for k, side, action in result.history:
            price_str = f"${action.price:.2f}" if action.price is not None else ""
            print(f"{k:2d} [{side:11s}] {action.decision.name:7s} {price_str}  msg={action.message!r}")
        print("outcome:", result.outcome)
        print()


if __name__ == "__main__":
    main()
