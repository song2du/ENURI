"""LLM 기반 agent policy -- 실제 OpenAI 모델이 협상하는 policy(episode, history, side, rng) -> Action.

오늘(2026-07-11) 목표: env.py/kernel.py/metrics.py 배관에 진짜 LLM을 한 번 붙여서 "LLM이 실제로
alternating-offer 프로토콜을 따라 협상하는지" 검증한다. Provider는 OpenAI로 결정 (2026-07-11 논의,
benchmark/CLAUDE.md 참고 -- Claude API 대신 OpenAI SDK/키를 쓰기로 함).

Item.ground_truth_defects는 이 파일 어디에서도 읽지 않는다 -- env.py의 Item docstring에 적힌
"agent policy 함수는 이 필드를 읽지 않는다"는 정보 비공개 규약을 그대로 지킨다.

!! 이미지 지원 (2026-07-23 추가) !!
`item.image_ref`가 실제 http(s) URL이면(placeholder 문자열이 아니면) OpenAI 비전 API의
`image_url` 컨텐츠 블록으로 함께 전송한다 (`_has_real_image`/`_user_content` 참고). 아직은
진짜 결함-합성 이미지 파이프라인이 없어서(benchmark/CLAUDE.md "구현 범위" 참고), 이 경로는
CraigslistBargain 실제 이미지로 "이미지+forced tool call이 같이 동작하는가"만 스모크테스트하는
용도다 -- 실제 결함 grounding 검증(citation_precision 등)은 팀원의 합성 이미지가 와야 가능.

!! 로컬 이미지 + 태그 어휘 인용 (2026-07-28 추가) !! 실 데이터(결함-합성 이미지) 도착 후 두
가지 보강: (1) `image_ref`가 로컬 파일 경로(http(s) 아님)여도 실제 이미지로 인식해서
base64 data URI로 전송한다. (2) agent는 ground-truth Defect.id를 절대 못 보므로(정보
비공개 규약), 결함을 인용하려면 "무슨 문자열을 써야 하는지" 알 방법이 없었다 -- 2026-07-25
결정(logs/2026-07-25.md)대로, id 대신 **닫힌 태그 어휘**(defect_type 7종)를 프롬프트에
미리 공개하고 그 단어 자체로 인용하게 한다. 이 어휘 공개는 "이 아이템에 실제로 어떤 결함이
있는지"를 새는 게 아니라 "결함 카테고리가 이렇게 나뉜다"만 공개하는 것이라 정보 비공개
규약과 안 충돌한다.

!! id -> defect_type 매칭으로 단순화 (2026-07-28, 사용자 결정) !! 처음엔 "<type>_0"
형식(Defect.id 그대로)으로 인용하게 시켰는데, 아이템당 결함이 0~1개뿐이라 "_0" 접미사는
항상 고정값이라 아무 정보가 없다 -- 그런데 agent가 그 접미사를 빼먹거나 대소문자를
틀리면 실제로 결함을 맞게 봤는데도 citation_precision 등에서 "실패"로 잡혔다. 측정
목표는 "형식을 맞추는가"가 아니라 "인용해서 써먹는가"이므로, 접미사를 아예 없애고
defect_type 단어 자체로만 비교한다 (대소문자/공백은 여기서 정규화). kernel.py/metrics.py도
Defect.id 대신 Defect.defect_type으로 매칭하도록 같이 수정됨 (decisions_log.md 참고).

model 기본값은 gpt-4o (2026-07-11 결정): 전략적 판단이 필요한 쪽(agent)에 더 강한 모델을,
단순 렌더링만 하는 voice.py 쪽(gpt-4o-mini)에는 가벼운 모델을 배분 -- voice.py 모듈 docstring 참고.

!! multi-provider 지원 (2026-07-26 추가) !! `provider="openrouter"`를 넘기면 OpenAI SDK
클라이언트를 OpenRouter(https://openrouter.ai)로 향하게 한다 -- 별도 SDK 없이 base_url만
바꾸는 이유: TERMS-Bench 논문 자체가 OpenRouter로 13개 LLM을 평가했음(terms-bench.txt:457
"LLMs are called via OpenRouter"), OpenRouter가 OpenAI 호환 chat.completions 형식(tools/
tool_choice/image_url 컨텐츠 블록 포함)으로 여러 provider를 통일해서 프록시해주므로
`_NEGOTIATE_TOOL`/`_build_prompt`/`_user_content`/응답 파싱 전부 그대로 재사용 가능
(benchmark/decisions_log.md 2026-07-26 참고). `provider="openai"`(기본값)는 기존 동작과
100% 동일 -- OPENROUTER_API_KEY가 없어도 안 깨짐. OpenRouter 쪽 모델 문자열은
"anthropic/claude-opus-4.6" 같은 "provider/model" 네이밍을 씀 (OpenAI 직결의 "gpt-4o"와
다름). OPENROUTER_API_KEY는 아직 미발급 상태(교수님께 연구실 결제 요청 예정, benchmark/
CLAUDE.md "다음 작업"의 "평가 대상 agent 리스트 제출"과 연결) -- 그래서 이 경로는 구조만
준비, 실제 호출 스모크테스트는 키 도착 후.

!! google provider 추가 (2026-08-03) !! Gemini/Gemma는 OpenRouter를 거치지 않고
`provider="google"`로 Google 자체 API(OpenAI 호환 엔드포인트, base_url=
".../v1beta/openai/")에 직결한다 -- 사용자가 Google 계정에 크레딧을 충전해둔 상태라
OpenRouter 마진 없이 직접 호출하는 게 비용상 유리하다는 2026-08-03 판단. 같은 OpenAI
SDK/엔드포인트 호환 방식이라 `_NEGOTIATE_TOOL`/`_build_prompt`/`_user_content`/응답
파싱은 openrouter 때와 동일하게 그대로 재사용. 모델 문자열은 OpenRouter 슬러그
("google/gemini-3.1-pro")가 아니라 Google 자체 모델 ID(예: "gemini-3.1-pro-preview")를
써야 함 -- 논문 로스터(Gemini 3.1 Pro Preview / Gemma 4 31B IT)와 정확히 맞는 ID인지는
아직 스모크테스트 전이라 AI Studio 콘솔에서 재확인 필요. `.env`의 `GOOGLE_API_KEY`를 읽음.
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

from env import Action, Decision, Episode, PRICE_IMPACT_MAPPINGS, Role
from kernel import favorability

load_dotenv()  # .env의 OPENAI_API_KEY(및 있다면 OPENROUTER_API_KEY)를 환경변수로 로드

# 일시적 API 장애 재시도 (2026-08-04 추가) -- 400-episode 정식 실행 중 OpenAI가
# `InternalServerError`(500, "server_error")를 던지며 몇 시간짜리 배치 전체가 죽는 걸
# 실측(decisions_log.md 참고). openai SDK 자체도 기본 재시도(max_retries=2)가 있지만
# 그걸로 부족한 경우가 실제로 있었으므로, policy() 레벨에서 한 번 더 감싼다. 재시도
# 대상은 "다시 시도하면 성공할 수 있는" 상태(5xx/429/네트워크 연결 문제)로만 한정 --
# BadRequestError(400, 우리 쪽 요청 자체가 잘못됨)나 AuthenticationError(401, 키 문제)
# 처럼 재시도해도 절대 안 풀리는 에러까지 여기서 삼키면 진짜 버그를 숨기게 되므로 제외.
# 2026-08-05 추가: `json.JSONDecodeError`도 포함 -- C_aggressive 정식 실행 중 응답 바디가
# 중간에 잘려서(네트워크 문제로 추정) `response.json()`이 그대로 실패하며 크래시하는 걸
# 실측. 이건 openai SDK 예외로 안 감싸이고 httpx가 raw json 파싱에서 바로 터뜨리는
# 케이스라 별도로 추가 -- 응답 바디가 잘리는 것도 "다시 요청하면 온전히 올 수 있는"
# 일시적 네트워크 문제로 분류.
_RETRYABLE_API_ERRORS = (
    openai.InternalServerError, openai.RateLimitError, openai.APIConnectionError, json.JSONDecodeError,
)
_MAX_API_RETRIES = 4
_RETRY_BACKOFF_SECONDS = 5

# 닫힌 태그 어휘 (2026-07-28, logs/2026-07-25.md 결정) -- PRICE_IMPACT_MAPPINGS의 키를 그대로
# 재사용해서 env.py와 어휘가 갈라지지 않게 한다 (어느 mapping preset이든 키는 동일 7종).
_DEFECT_TAG_VOCAB = sorted(PRICE_IMPACT_MAPPINGS["B_moderate"].keys())

_NEGOTIATE_TOOL = {
    "type": "function",
    "function": {
        "name": "negotiate_action",
        "description": "Decide the action to take in this negotiation turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["OFFER", "ACCEPT", "REJECT"],
                    "description": "OFFER=propose a new price, ACCEPT=accept the counterpart's last offer, REJECT=walk away from the negotiation",
                },
                "price": {
                    "type": ["number", "null"],
                    "description": "Required only when decision is OFFER. Null otherwise.",
                },
                "message": {
                    "type": "string",
                    "description": "The message sent to the counterpart.",
                },
                "cited_defect_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Defect type(s) you actually observed in the photo and are citing this "
                    "message, using the defect type list given in the prompt (e.g. 'scratch'). Empty array "
                    "if you don't cite any defect this turn.",
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
        return "(No one has made an offer yet)"
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


def _has_real_image(item) -> bool:
    """item.image_ref가 mock placeholder("placeholder://...")가 아니라 실제 이미지인지 --
    http(s) URL이거나(craigslist_smoke.py 경로), 실제 존재하는 로컬 파일 경로면(data_loader.py
    경로, 2026-07-28 추가) True. env.py의 sample_item은 항상 placeholder를 쓰므로 mock
    경로는 계속 False."""
    if item.image_ref.startswith("http://") or item.image_ref.startswith("https://"):
        return True
    return os.path.isfile(item.image_ref)


def _sniff_image_mime(data: bytes) -> str:
    """확장자 대신 매직 바이트로 실제 이미지 포맷을 판별한다 -- 실 데이터의 synth/*.jpg는
    파일명은 .jpg지만 실제로는 PNG(나노바나나 결함합성 API 출력이 항상 PNG라 확인됨,
    2026-07-28), originals/*.jpg는 실제 JPEG라 확장자만으로는 못 믿는다."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    return "jpeg"  # JPEG(\xff\xd8\xff)가 기본, 이 데이터셋의 나머지 전부


def _image_url(item) -> str:
    """item.image_ref -> OpenAI 비전 API의 image_url 값. http(s)는 그대로 URL로, 로컬 파일은
    base64 data URI로 인코딩한다 (2026-07-28 추가 -- data_loader.py가 넘기는 image_ref는
    항상 로컬 절대경로). MIME 타입은 확장자가 아니라 _sniff_image_mime()으로 판별."""
    if item.image_ref.startswith("http://") or item.image_ref.startswith("https://"):
        return item.image_ref
    raw = Path(item.image_ref).read_bytes()
    data = base64.b64encode(raw).decode("ascii")
    return f"data:image/{_sniff_image_mime(raw)};base64,{data}"


def _build_prompt(
    episode: Episode, history: list[tuple[int, str, Action]], side: str, *, include_image: bool
) -> tuple[str, str]:
    role = episode.role_A  # 이 policy는 agent 전용이라 side는 항상 "agent"
    k = len(history) + 1
    item = episode.item

    system = (
        "You are negotiating the price of a used item over chat, alternating offers with a counterpart. "
        f"You are the {role.name}. You must call the negotiate_action tool exactly once per turn to record "
        "your move -- OFFER a price, ACCEPT the counterpart's last offer, or REJECT and walk away."
    )
    photo_line = (
        "\nA photo of the item is attached above -- look it over before deciding your move. "
        "If you notice a defect, cite it in cited_defect_ids using its type, one of: "
        f"{', '.join(_DEFECT_TAG_VOCAB)}."
    ) if include_image else ""
    user = f"""Item for sale:
- Category: {item.category}
- Title: {item.title}
- Description: {item.description}
- Listing price: ${item.listing_price:.2f}
{photo_line}
Your role: {role.name}
Your reservation price (the {"most you will pay" if role == Role.BUYER else "least you will accept"}): ${episode.r_A:.2f}
Price bounds for this negotiation: ${episode.p_min:.2f} - ${episode.p_max:.2f}
Round {k} of {episode.K} (the negotiation ends in disagreement if no deal is reached by round {episode.K}).

Negotiation so far:
{format_history(history, side)}

What is your move this round?"""
    return system, user


def _user_content(episode: Episode, user_text: str, *, include_image: bool) -> str | list[dict]:
    """include_image가 참이면 OpenAI 비전 API 형식(텍스트+image_url 블록 리스트)으로,
    아니면 기존처럼 평범한 문자열로 반환 -- 이미지 없는 기존 mock 경로는 동작이 안 바뀐다."""
    if not include_image:
        return user_text
    return [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": _image_url(episode.item)}},
    ]


def make_llm_agent_policy(
    model: str = "gpt-4o", provider: str = "openai", use_image: bool = True, cost_log: list[float] | None = None
):
    """kernel.py의 make_counterpart_policy와 같은 팩토리 패턴 -- model/provider/use_image를
    클로저에 가둬서 policy(episode, history, side, rng) -> Action 시그니처에 맞춘 함수를 반환한다.

    provider="openai"(기본값): 기존 동작 그대로, OPENAI_API_KEY로 OpenAI에 직결.
    provider="openrouter": 같은 OpenAI SDK 클라이언트를 OpenRouter의 base_url로 돌려서
    Claude/Gemini/Qwen 등 다른 provider 모델도 같은 코드 경로로 호출 (모듈 docstring의
    "multi-provider 지원" 절 참고). model 문자열은 이때 "anthropic/claude-opus-4.6"처럼
    OpenRouter 네이밍을 써야 한다.
    provider="google"(2026-08-03 추가): Gemini/Gemma를 OpenRouter 대신 Google 자체 API로
    직결 (모듈 docstring의 "google provider 추가" 절 참고). model 문자열은 Google 자체
    모델 ID(예: "gemini-3.1-pro-preview")를 써야 한다 -- OpenRouter 슬러그와 다름.

    use_image=False(2026-07-29 추가, pulse.pptx 슬라이드6 "visual 유/무 baseline"): 실 이미지가
    있어도 일부러 안 보낸다 -- item.image_ref가 실제 이미지인지(_has_real_image, 데이터 사실)와
    이번 실행이 그걸 실제로 쓸지(use_image, 실험 조건)를 분리해서, "이미지를 줬을 때와 안 줬을
    때 협상 결과가 달라지는가"라는 gap을 같은 아이템으로 비교할 수 있게 한다.

    cost_log(2026-08-03 추가, 실비용 측정용): 리스트를 넘기면 provider="openrouter"일 때
    매 호출마다 실제 청구 비용(USD)을 append한다. OpenRouter가 요청에 `usage: {"include":
    true}`를 실으면 응답 `usage.cost`에 그 요청의 실제 과금액(토큰 단가뿐 아니라 프롬프트
    캐싱 할인 등까지 반영된 최종값)을 얹어준다는 걸 실측 확인(2026-08-03) -- 토큰 수 x
    공개 단가로 수동 환산하는 것보다 정확. 이전(2026-08-02) 비용 파일럿은 이 기능 없이
    수동으로 환산했는데, 이번부터는 이 값을 그대로 합산하면 됨. openai/google
    provider는 OpenRouter 전용 파라미터라 굳이 안 건드림(비용 측정이 필요해지면 그때
    각자 방식으로 추가).
    """
    if provider == "openai":
        client = OpenAI()  # OPENAI_API_KEY는 load_dotenv()로 이미 환경변수에 로드됨
    elif provider == "openrouter":
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    elif provider == "google":
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ["GOOGLE_API_KEY"],
        )
    else:
        raise ValueError(f"unknown provider: {provider!r} (expected 'openai', 'openrouter', or 'google')")

    def policy(episode: Episode, history: list[tuple[int, str, Action]], side: str, rng: random.Random) -> Action:
        include_image = use_image and _has_real_image(episode.item)
        system, user = _build_prompt(episode, history, side, include_image=include_image)
        extra_kwargs = {}
        if cost_log is not None and provider == "openrouter":
            extra_kwargs["extra_body"] = {"usage": {"include": True}}
        for attempt in range(_MAX_API_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": _user_content(episode, user, include_image=include_image)},
                    ],
                    tools=[_NEGOTIATE_TOOL],
                    tool_choice={"type": "function", "function": {"name": "negotiate_action"}},
                    **extra_kwargs,
                )
                break
            except _RETRYABLE_API_ERRORS:
                if attempt == _MAX_API_RETRIES:
                    raise
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))  # 5s, 10s, 15s, 20s
        if cost_log is not None and provider == "openrouter":
            cost = getattr(response.usage, "cost", None)
            if cost is not None:
                cost_log.append(cost)
        # 응답 파싱 전체를 하나의 방어막으로 묶는다 (2026-08-02, qwen3.6-plus의 price=""와
        # kimi-k3의 tool_calls=None을 실측하며 확장). 처음엔 price 파싱만 개별로 감쌌는데,
        # `tool_choice`로 강제 호출을 요구해도 OpenRouter로 붙는 provider가 이걸 100% 안
        # 지킬 수 있다는 게 확인된 이상, "이 응답을 신뢰할 수 있는 구조로 못 읽는다"는 카테고리
        # 전체(빈 tool_calls, JSON 파싱 실패, decision 값 이상, price 파싱 실패 등)를 한 곳에서
        # 같은 방식으로 처리하는 게 더 견고하다 -- env.py의 "LLM이 이상한 값을 내도 벤치마크가
        # 안 죽어야 한다" 철학(run_episode 참고)의 연장.
        try:
            call = response.choices[0].message.tool_calls[0]
            args = json.loads(call.function.arguments)
            decision = Decision[args["decision"]]
            raw_price = args.get("price")
            price = float(raw_price) if raw_price not in (None, "") else None
            if decision == Decision.OFFER and price is None:
                raise ValueError("OFFER without a usable price")
            # 대소문자/공백만 정규화 -- defect_type 자체가 틀렸으면(할루시네이션) 그대로 안
            # 걸러지고 kernel.py/metrics.py의 defect_type 매칭에서 "실제 없는 결함"으로
            # 정상적으로 잡혀야 함.
            cited = tuple(c.strip().lower() for c in (args.get("cited_defect_ids") or [])) or None
            return Action(decision=decision, price=price, message=args.get("message"), cited_defect_ids=cited)
        except (KeyError, ValueError, TypeError, AttributeError, IndexError):
            # 논문 H.1.5의 JSON-파싱-실패 폴백과 동일한 규칙("상대 offer가 걸어나가기보다
            # 나으면 수락, 아니면 거절") -- fixed_concession_agent.py가 이미 쓰는 걸 재사용,
            # 새 규칙을 만들지 않는다.
            last_action = history[-1][2] if history else None
            if (
                last_action is not None
                and last_action.decision == Decision.OFFER
                and favorability(last_action.price, episode.r_A, episode.role_A, episode.p_max - episode.p_min) >= 0
            ):
                return Action(decision=Decision.ACCEPT, price=None, message=None, cited_defect_ids=None)
            return Action(decision=Decision.REJECT, price=None, message=None, cited_defect_ids=None)

    return policy
