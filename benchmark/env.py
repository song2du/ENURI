"""Evidence-Based Negotiation 환경 (Gamma) + protocol 루프.

implementation/env.py의 TERMS-BENCH 환경(Gamma = (F, T_A, T_B, u_A, u_B, mu),
alternating-offer protocol)을 뼈대로 그대로 가져오되, 우리 벤치마크만의 새 축인
"시각 증거(visual evidence)"를 Item/Defect로 추가한다.

설계 논의 전체 기록: benchmark/CLAUDE.md의 "구현 범위" 절 참고. 이 파일은
implementation/env.py를 import하지 않고 독립적으로 복사+확장했다 — implementation/은
학습용 스크래치 공간으로 스코프가 잡혀 있어, 실제 결과물인 benchmark/가 거기
의존하면 나중에 학습용 코드가 바뀔 때 같이 깨질 위험이 있기 때문 (2026-07-11 결정).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto

DISAGREEMENT = object()  # f = perp (bottom / no-deal outcome). implementation/env.py와 동일.


class Role(Enum):
    BUYER = auto()
    SELLER = auto()


class Decision(Enum):
    OFFER = auto()
    ACCEPT = auto()
    REJECT = auto()


# ---------------------------------------------------------------------------
# Evidence 데이터 모델 (benchmark/CLAUDE.md "환경 -- evidence 데이터 모델" 절, 2026-07-11 결정)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Defect:
    """아이템에 실제로 존재하는 결함 하나 -- ground truth.

    counterpart kernel(전용, ground truth를 직접 읽음)과 metrics(rule matcher가
    agent의 cited_defect_ids와 비교)만 이 객체를 직접 참조한다. VLM agent는 이
    객체를 절대 못 보고, Item.image_ref(이미지)를 직접 보고 결함을 스스로
    찾아내야 한다 -- 이게 "시각 증거 활용 역량"을 재는 핵심 정보 비대칭이다.
    """

    id: str  # rule matcher(citation_precision 등)가 참조하는 안정적 식별자
    description: str  # 결함의 사실 관계 (예: "왼쪽 팔걸이에 3cm 찢어짐")
    price_impact: float  # 이 결함 하나가 "공정가"에서 깎아야 하는 금액
    salience: float | None = None  # 0~1, 육안으로 얼마나 두드러지는지. !! 2026-07-11 결정: 8주 학술제 스코프에서 제외 !! -- 제대로 측정하려면 연구자 본인이 아닌 3인 이상 독립 코더가 결함 위치를 모르는 채로(블라인드) 각자 판단하고 ICC(intraclass correlation)로 신뢰도를 확인해야 하는데, 이건 이번 스코프 밖 future work다. 지금은 항상 None -- 어떤 코드도 이 필드를 실제로 참조하지 않는다 (benchmark/CLAUDE.md, benchmark/data_spec.md 참고).


@dataclass(frozen=True)
class Item:
    """협상 대상 물건 -- buyer/seller(와 VLM agent) 모두에게 category/title/
    description/listing_price/image_ref는 동일하게 노출된다 (paper/CLAUDE.md
    시나리오: "판매자·구매자 모두 시각 증거 + 텍스트 메타데이터에 동일하게 접근
    가능"). ground_truth_defects만 예외 -- 커널 전용, agent에게는 비공개.
    """

    category: str
    title: str
    description: str  # CraigslistBargain 스타일 리스팅 설명. 결함을 자진 공개하지 않는다는 전제 -- 그래야 이미지를 봐야 하는 이유가 생김.
    listing_price: float
    image_ref: str  # 이미지 파일 경로/식별자. 실제 결함-합성 이미지 파이프라인은 스코프 밖 (benchmark/CLAUDE.md 참고) -- 지금은 자리표시자 문자열.
    ground_truth_defects: tuple[Defect, ...]  # 커널/metrics 전용. agent policy 함수는 이 필드를 읽지 않는다는 규약으로 정보 비공개를 지킨다 (t_B.r을 agent policy가 안 읽는 것과 동일한 관례).


# ---------------------------------------------------------------------------
# TERMS-BENCH 원본 데이터 모델 (implementation/env.py와 동일, item 필드만 추가)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeB:
    """t_B = (r_B, delta_B, sigma_B). implementation/env.py와 동일."""

    r: float
    urgency: float
    stance: str  # "conciliatory" | "neutral" | "aggressive"


@dataclass(frozen=True)
class Episode:
    regime: str  # "overlap" | "urgency_shift" | "no_deal"
    p_min: float
    p_max: float
    role_A: Role
    r_A: float
    t_B: TypeB
    opener: str  # "AgentOpens" | "CounterpartOpens"
    K: int
    harshness: float  # d_{0,e}, opening-offer harshness. implementation/env.py와 동일.
    item: Item  # NEW: 이 episode에서 협상 중인 물건 + ground truth 결함


@dataclass(frozen=True)
class Action:
    decision: Decision
    price: float | None = None  # required iff decision == OFFER
    message: str | None = None  # l_k, placeholder -- 언어 레이어는 아직 없음 (VLM agent가 실제 텍스트를 생성하게 되면 이 필드에 채워짐)
    sentiment: str | None = None  # s_k_tilde, counterpart-only
    posture: str | None = None  # c_k_tilde, counterpart-only
    cited_defect_ids: tuple[str, ...] | None = None  # NEW: 이번 액션에서 agent가 언급한 결함 id들. 자유 텍스트 message만으로는 rule matcher가 채점 못하므로, agent 출력 스키마에 이 구조화된 필드를 별도로 요구한다 (implementation/agent.py의 reported_belief와 같은 패턴).


@dataclass
class EpisodeResult:
    outcome: object  # a price (float) or DISAGREEMENT
    history: list[tuple[int, str, Action]] = field(default_factory=list)
    violations: list[tuple[int, str, str]] = field(default_factory=list)


def _other(side: str) -> str:
    return "counterpart" if side == "agent" else "agent"


def utility(role: Role, outcome: object, r: float) -> float:
    """u_buyer(p) = r_buyer - p, u_seller(p) = p - r_seller, both 0 on DISAGREEMENT."""
    if outcome is DISAGREEMENT:
        return 0.0
    price = outcome
    return (r - price) if role == Role.BUYER else (price - r)


def sample_urgency(rng: random.Random, shifted: bool, shift: float, alpha: float = 2.0, beta: float = 2.0) -> float:
    """Baseline delta_B ~ Beta(alpha, beta) on [0,1]. implementation/env.py와 동일 (Simplification도 동일하게 유지)."""
    u = rng.betavariate(alpha, beta)
    if shifted:
        u = min(1.0, max(0.0, u + shift))
    return u


# ---------------------------------------------------------------------------
# Evidence(Item/Defect) mock 생성기 -- 프로토타입 전용
# ---------------------------------------------------------------------------

_DEFECT_CATALOG = [
    # (kind, description, price_impact) -- salience는 2026-07-11 결정으로 스코프 밖이라 여기 없음 (Defect.salience 필드 주석 참고)
    ("scratch", "Scratch on the surface", 5.0),
    ("tear", "Tear in the fabric", 15.0),
    ("stain", "Stain", 8.0),
    ("dent", "Dent", 12.0),
    ("missing_part", "Missing accessory/part", 20.0),
]


def sample_item(rng: random.Random, listing_price: float, num_defects: int = 2) -> Item:
    """프로토타입용 mock item 생성기.

    실제 CraigslistBargain 이미지 + 결함 합성 파이프라인 연동은 스코프 밖
    (benchmark/CLAUDE.md "구현 범위" 참고) -- 지금은 grounding/utilization
    metric의 데이터 배관(파이프라인)이 실제로 작동하는지 검증하기 위한
    자리표시자 데이터를 만든다. image_ref도 실제 파일이 아니라 문자열
    placeholder.

    num_defects는 고정값으로 단순화 (원래는 0~N개로 다양화해야 하나, 오늘은
    "결함이 최소 1개는 있는 정상 케이스"부터 배관을 검증하는 게 우선).
    """
    picks = rng.sample(_DEFECT_CATALOG, k=min(num_defects, len(_DEFECT_CATALOG)))
    defects = tuple(
        Defect(
            id=f"{kind}_{i}",
            description=desc,
            price_impact=price_impact,
            # salience는 지정 안 함 -> 기본값 None (스코프 밖, Defect.salience 필드 주석 참고)
        )
        for i, (kind, desc, price_impact) in enumerate(picks)
    )
    return Item(
        category="furniture",
        title="Used item for sale",
        description="Gently used, see photos for details.",
        listing_price=listing_price,
        image_ref="placeholder://item.jpg",
        ground_truth_defects=defects,
    )


def sample_episode(
    rng: random.Random,
    p_min: float = 0.0,
    p_max: float = 100.0,
    K: int = 10,
    regimes: tuple[str, ...] = ("overlap", "urgency_shift", "no_deal"),
    regime_weights: tuple[float, ...] | None = None,
    z_range: tuple[float, float] = (5.0, 40.0),
    q_range: tuple[float, float] = (5.0, 40.0),
    urgency_shift: float = 0.2,
    stance_prior: tuple[str, ...] = ("conciliatory", "neutral", "aggressive"),
    harshness_range: tuple[float, float] = (0.20, 0.80),
    item: Item | None = None,
) -> Episode:
    """mu: episode의 regime, reservation, urgency, stance + item(evidence)을 샘플링.

    가격 geometry(z/q/regime 샘플링)는 implementation/env.py와 완전히 동일 --
    프로토타입 단계에서는 손대지 않기로 결정했다.

    !! Simplification (2026-07-11, benchmark/CLAUDE.md "구현 범위" 참고) !!
    결함이 심각한 아이템인데 ZOPA(z/q)는 결함과 무관하게 랜덤으로 나올 수 있다 --
    가격 geometry와 결함 ground truth가 서로 독립적으로 샘플링되는 상태다. 원래는
    r_buyer/r_seller(또는 listing_price)가 Defect.price_impact 합산에서 논리적으로
    유도돼야 한다. 오늘은 마감(프로토타입) 때문에 범위를 좁혔지만, 이 갭은
    grounding/utilization metric의 타당성 자체에 영향을 줄 수 있는 리스크라
    방치하면 안 된다 -- 나중에 반드시 재검토할 것.
    """
    weights = list(regime_weights) if regime_weights else [1.0] * len(regimes)
    regime = rng.choices(regimes, weights=weights)[0]

    if regime in ("overlap", "urgency_shift"):
        z = rng.uniform(*z_range)
        m = rng.uniform(p_min + z / 2, p_max - z / 2)
        r_buyer, r_seller = m + z / 2, m - z / 2
    else:  # no_deal
        q = rng.uniform(*q_range)
        m = rng.uniform(p_min + q / 2, p_max - q / 2)
        r_buyer, r_seller = m - q / 2, m + q / 2

    urgency_B = sample_urgency(rng, shifted=(regime == "urgency_shift"), shift=urgency_shift)
    stance_B = rng.choice(stance_prior)

    role_A = rng.choice([Role.BUYER, Role.SELLER])
    opener = rng.choice(["AgentOpens", "CounterpartOpens"])
    harshness = rng.uniform(*harshness_range)

    r_A = r_buyer if role_A == Role.BUYER else r_seller
    r_B = r_seller if role_A == Role.BUYER else r_buyer

    episode_item = item if item is not None else sample_item(rng, listing_price=(p_min + p_max) / 2)

    return Episode(
        regime=regime,
        p_min=p_min,
        p_max=p_max,
        role_A=role_A,
        r_A=r_A,
        t_B=TypeB(r=r_B, urgency=urgency_B, stance=stance_B),
        opener=opener,
        K=K,
        harshness=harshness,
        item=episode_item,
    )


def run_episode(episode: Episode, agent_policy, counterpart_policy, rng: random.Random) -> EpisodeResult:
    """Alternating-offer protocol 루프. implementation/env.py의 run_episode와 완전히 동일 --
    cited_defect_ids는 Action의 필드일 뿐이라 이 루프 자체는 변경이 필요 없다
    (history에 Action 객체를 그대로 저장하므로 인용 정보도 같이 보존된다).
    """
    history: list[tuple[int, str, Action]] = []
    violations: list[tuple[int, str, str]] = []
    last_offer: dict[str, float | None] = {"agent": None, "counterpart": None}
    turn = "agent" if episode.opener == "AgentOpens" else "counterpart"

    for k in range(1, episode.K + 1):
        policy = agent_policy if turn == "agent" else counterpart_policy
        action: Action = policy(episode, history, side=turn, rng=rng)

        if action.decision == Decision.ACCEPT and last_offer[_other(turn)] is None:
            violations.append((k, turn, "accept_before_any_offer"))
            return EpisodeResult(outcome=DISAGREEMENT, history=history, violations=violations)

        history.append((k, turn, action))

        if action.decision == Decision.ACCEPT:
            return EpisodeResult(outcome=last_offer[_other(turn)], history=history, violations=violations)
        if action.decision == Decision.REJECT:
            return EpisodeResult(outcome=DISAGREEMENT, history=history, violations=violations)

        # OFFER: record and pass the turn
        last_offer[turn] = action.price
        turn = _other(turn)

    return EpisodeResult(outcome=DISAGREEMENT, history=history, violations=violations)  # round-limit disagreement
