"""counterpart의 "voice" 레이어 -- 이미 확정된 economic decision을 자연어 메시지로 렌더링.

논문 Appendix C.5.4 "Language Realization"과 대응:
    "Natural-language realizations are generated only after the economic kernel has already
    committed to (d^B_k, p^B_k, s~k, c~k). The voice layer therefore cannot alter economic
    outcomes."
즉 kernel.py가 이미 결정한 decision/price/sentiment/posture를 그대로 받아서 문장으로 포장만
한다 -- 이 레이어가 협상 결과 자체를 바꾸지 않는다는 게 논문/우리 설계 둘 다의 핵심 불변식.

kernel.py는 이 파일을 import하지 않는다 (그 반대). kernel.py는 OpenAI 의존성이 전혀 없는 순수
경제 시뮬레이터로 남겨두고(그래야 API 키 없이도 기존 sanity check가 다 돌아감), 이 파일이
`add_voice()`로 그 위에 얹는 선택적 레이어 역할을 한다.

VOICE_MODEL이 llm_agent.py의 agent 모델과 완전히 분리·고정된 이유 (2026-07-11 결정):
논문은 "The counterpart voice model is fixed to GPT-5.2"라고 명시 -- 평가 대상 agent 모델이
무엇으로 바뀌든 voice는 절대 안 바뀐다. 만약 voice가 agent와 같은 모델이면, agent가 사실상
"자기 자신과 대화"하는 셈이 되어 자기 편향(self-bias) 오염이 생긴다. 그래서 VOICE_MODEL은
run_negotiation.py의 --model 플래그(agent용)와 무관하게 여기서 상수로 고정한다.

모델 배분(2026-07-11): agent는 gpt-4o(전략적 판단 필요), voice는 gpt-4o-mini(경제적
결정 없이 이미 확정된 내용을 문장으로 포장만 하는 좁은 렌더링 작업이라 가벼운 모델로 충분).
"""

from __future__ import annotations

import random
from dataclasses import replace

from dotenv import load_dotenv
from openai import OpenAI

from env import Action, Decision, Episode, Role
from llm_agent import format_history

load_dotenv()

VOICE_MODEL = "gpt-4o-mini"  # agent 기본값(gpt-4o)과 다른, 고정된 모델. run_negotiation.py --model과 무관.


def _build_voice_prompt(role_B: Role, action: Action, history_text: str) -> tuple[str, str]:
    """논문 C.5.4가 명시한 입력 그대로: 역할, 확정된 decision/price, sentiment/posture cue, history 요약."""
    posture_style = {
        "Concede": "compromise-oriented -- signal willingness to move toward agreement",
        "Hold": "firm but non-escalatory -- hold your position without ramping up conflict",
        "Pressure": "urgency- or deadline-oriented -- convey time pressure or impatience",
    }[action.posture]
    sentiment_style = {
        "positive": "polite and constructive",
        "neutral": "matter-of-fact",
        "negative": "tense",
    }[action.sentiment]

    system = (
        "You write short chat messages for one side of a price negotiation. "
        "The economic decision (accept/reject/offer and price) has ALREADY been made and is fixed -- "
        "you only write the message that accompanies it. Do not mention a different price or decision "
        "than the one given. Keep the message to 1-2 sentences."
    )
    price_line = f"Price you are offering: ${action.price:.2f}" if action.price is not None else "(no price -- this is not an offer)"
    user = f"""You are the {role_B.name} in this negotiation.
Your fixed decision this turn: {action.decision.name}
{price_line}
Tone to express: {sentiment_style}
Posture to express: {posture_style}

Negotiation so far:
{history_text}

Write your message now (1-2 sentences, no price/decision other than what was given above)."""
    return system, user


def add_voice(counterpart_policy, model: str = VOICE_MODEL):
    """counterpart_policy(예: make_counterpart_policy(family)의 반환값)를 받아서, 같은
    시그니처의 policy(episode, history, side, rng) -> Action를 반환하되 Action.message를
    실제 LLM이 쓴 문장으로 채워준다. decision/price는 절대 안 건드림 -- economic kernel의
    출력을 그대로 감싸기만 한다.
    """
    client = OpenAI()

    def policy(episode: Episode, history: list[tuple[int, str, Action]], side: str, rng: random.Random) -> Action:
        action = counterpart_policy(episode, history, side, rng)  # 먼저 economic decision 확정
        role_B = Role.SELLER if episode.role_A == Role.BUYER else Role.BUYER
        history_text = format_history(history, side)

        system, user = _build_voice_prompt(role_B, action, history_text)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        message = response.choices[0].message.content

        return replace(action, message=message)  # Action은 frozen dataclass라 replace로 message만 교체

    return policy
