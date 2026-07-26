"""craigslist_smoke.py의 협상 transcript만으론 "이미지를 진짜로 봤는지"가 불명확해서
(모델이 협상 중엔 시각적 디테일을 굳이 말 안 할 수 있음) -- 협상과 분리해서 딱 "이 사진에
뭐가 보이냐"고 직접 물어보는 최소 스크립트. 2026-07-23, 일회성 검증용."""

from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

from craigslist_smoke import _REAL_ITEMS, make_real_item

load_dotenv()
client = OpenAI()

for category, post_id, title, description, price in _REAL_ITEMS:
    item = make_real_item(category, post_id, title, description, price)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what you see in this photo in one or two sentences. Be specific about color, material, and condition."},
                    {"type": "image_url", "image_url": {"url": item.image_ref}},
                ],
            }
        ],
    )
    print(f"=== {item.title} ===")
    print(response.choices[0].message.content)
    print()
