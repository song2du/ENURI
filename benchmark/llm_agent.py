"""LLM 기반 agent policy -- 실제 OpenAI 모델이 협상하는 policy(episode, history, side, rng) -> Action.

오늘(2026-07-11) 목표: env.py/kernel.py/metrics.py 배관에 진짜 LLM을 한 번 붙여서 "LLM이 실제로
alternating-offer 프로토콜을 따라 협상하는지" 검증한다. Provider는 OpenAI로 결정 (2026-07-11 논의,
benchmark/CLAUDE.md 참고 -- Claude API 대신 OpenAI SDK/키를 쓰기로 함).

!! 스코프 제한 !!
아직 실제 이미지/결함 합성 파이프라인이 없어서 (benchmark/CLAUDE.md "구현 범위" 참고), 이 agent는
순수 텍스트 정보(Item.category/title/description/listing_price)만 보고 협상한다.
Item.ground_truth_defects는 이 파일 어디에서도 읽지 않는다 -- env.py의 Item docstring에 적힌
"agent policy 함수는 이 필드를 읽지 않는다"는 정보 비공개 규약을 그대로 지킨다. 실제 이미지를 보고
결함을 스스로 찾아내는 VLM 통합은 여기서 검증되지 않는다 -- 오늘은 "LLM이 실제로 가격을 정해서
Action 스키마에 맞게 응답하는지"까지만 확인한다.

model 기본값은 gpt-4o (2026-07-11 결정): 전략적 판단이 필요한 쪽(agent)에 더 강한 모델을,
단순 렌더링만 하는 voice.py 쪽(gpt-4o-mini)에는 가벼운 모델을 배분 -- voice.py 모듈 docstring 참고.
"""

from __future__ import annotations

import json
import random

from dotenv import load_dotenv
from openai import OpenAI

from env import Action, Decision, Episode, Role

load_dotenv()  # .env의 OPENAI_API_KEY를 환경변수로 로드

_NEGOTIATE_TOOL = {
    "type": "function",
    "function": {
        "name": "negotiate_action",
        "description": "이번 턴에 협상에서 취할 행동을 결정한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["OFFER", "ACCEPT", "REJECT"],
                    "description": "OFFER=새 가격 제안, ACCEPT=상대의 마지막 제안 수락, REJECT=협상 결렬",
                },
                "price": {
                    "type": ["number", "null"],
                    "description": "decision이 OFFER일 때만 필수 값. 그 외에는 null.",
                },
                "message": {
                    "type": "string",
                    "description": "상대에게 보내는 메시지.",
                },
                "cited_defect_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "이번 메시지에서 실제로 언급한 결함 id들 (지금은 이미지가 없어 항상 비어있을 것). 없으면 빈 배열.",
                },
            },
            "required": ["decision", "price", "message", "cited_defect_ids"],
            "additionalProperties": False,
        },
    },
}


def format_history(history: list[tuple[int, str, Action]], side: str) -> str:
    """어느 한쪽(side) 시점에서 지금까지의 협상 기록을 텍스트로 변환.

    counterpart의 t_B, item의 ground_truth_defects는 여기 절대 안 들어간다 -- 애초에 이 함수가
    받는 history/side에는 그 정보 자체가 없다 (env.py의 정보 비공개 규약). 밑줄 없는 이름인 이유:
    voice.py의 counterpart 메시지 렌더링에서도 그대로 재사용하는 공용 유틸이라서다.
    """
    if not history:
        return "(아직 아무도 제안하지 않음)"
    lines = []
    for k, turn_side, action in history:
        speaker = "You" if turn_side == side else "Counterpart"
        if action.decision == Decision.OFFER:
            lines.append(f'Round {k} [{speaker}] OFFER ${action.price:.2f} -- "{action.message or ""}"')
        elif action.decision == Decision.ACCEPT:
            lines.append(f"Round {k} [{speaker}] ACCEPT")
        else:
            lines.append(f"Round {k} [{speaker}] REJECT")
    return "\n".join(lines)


def _build_prompt(episode: Episode, history: list[tuple[int, str, Action]], side: str) -> tuple[str, str]:
    role = episode.role_A  # 이 policy는 agent 전용이라 side는 항상 "agent"
    k = len(history) + 1
    item = episode.item

    system = (
        "You are negotiating the price of a used item over chat, alternating offers with a counterpart. "
        f"You are the {role.name}. You must call the negotiate_action tool exactly once per turn to record "
        "your move -- OFFER a price, ACCEPT the counterpart's last offer, or REJECT and walk away."
    )
    user = f"""Item for sale:
- Category: {item.category}
- Title: {item.title}
- Description: {item.description}
- Listing price: ${item.listing_price:.2f}

Your role: {role.name}
Your reservation price (the {"most you will pay" if role == Role.BUYER else "least you will accept"}): ${episode.r_A:.2f}
Price bounds for this negotiation: ${episode.p_min:.2f} - ${episode.p_max:.2f}
Round {k} of {episode.K} (the negotiation ends in disagreement if no deal is reached by round {episode.K}).

Negotiation so far:
{format_history(history, side)}

What is your move this round?"""
    return system, user


def make_llm_agent_policy(model: str = "gpt-4o"):
    """kernel.py의 make_counterpart_policy와 같은 팩토리 패턴 -- model을 클로저에 가둬서
    policy(episode, history, side, rng) -> Action 시그니처에 맞춘 함수를 반환한다.
    """
    client = OpenAI()  # OPENAI_API_KEY는 load_dotenv()로 이미 환경변수에 로드됨

    def policy(episode: Episode, history: list[tuple[int, str, Action]], side: str, rng: random.Random) -> Action:
        system, user = _build_prompt(episode, history, side)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tools=[_NEGOTIATE_TOOL],
            tool_choice={"type": "function", "function": {"name": "negotiate_action"}},
        )
        call = response.choices[0].message.tool_calls[0]
        args = json.loads(call.function.arguments)

        decision = Decision[args["decision"]]
        price = float(args["price"]) if args.get("price") is not None else None
        cited = tuple(args.get("cited_defect_ids") or []) or None

        return Action(decision=decision, price=price, message=args.get("message"), cited_defect_ids=cited)

    return policy
