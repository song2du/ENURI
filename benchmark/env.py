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
    quadrant: str | None = None  # NEW (2026-07-23): "obvious_aligned"|"subtle_aligned"|"obvious_misaligned"|"subtle_misaligned" -- pulse.pptx 슬라이드2/4의 가시성x정합성 4분면 태그. salience와 다르게 이건 "측정해서 검증해야 하는 연속값"이 아니라 "결함을 합성할 때 제작자가 의도적으로 부여하는 설계 레이블"이다 (counterpart의 FAMILIES가 커널 설계 레이블인 것과 같은 성격) -- 그래서 salience 스코프아웃 결정과 충돌하지 않는다. metrics.py의 subtle_misaligned_gap이 이 필드로 "안 보이지만 가치 영향 큰" 결함을 식별한다. 실제 이미지 파이프라인 연동 전까지는 _DEFECT_IDENTITIES(2026-07-26, 구 _DEFECT_CATALOG)의 mock 태그만 존재 -- 팀원의 실제 분류로 교체 예정.
    severity: str | None = None  # NEW (2026-07-28, 실 데이터 도착 후 추가): "mild"|"moderate"|"severe", 실 데이터가 갖고 있는 필드라 파싱은 하되 price_impact 계산엔 안 쓴다 -- 아래 PRICE_IMPACT_MAPPINGS 주석의 2026-07-28 경위 참고 (한때 severity 축으로 가격을 계산하려 했다가, pulse.pptx 원본 슬라이드 확인 후 defect_type 축이 맞다는 게 드러나 되돌림). 순수 정보 보관용 -- quadrant(visibility x alignment)와 100% 상관되는 값이라(실 데이터 270개 결함 전수 검증, 예외 0건) 나중에 참고할 순 있어도 지금 아무 계산도 이 필드를 읽지 않는다.
    defect_type: str | None = None  # NEW (2026-07-28): "scratch"/"dent"/"rust"/"crack"/"tear"/"stain"/"missing_part" 결함 종류 태그. price_impact를 실제로 계산하는 입력이 이 필드다 (아래 PRICE_IMPACT_MAPPINGS 참고) -- pulse.pptx "구체적 절차" 슬라이드 원표의 열 이름이 문자 그대로 결함 종류(스크래치/화면균열/부품누락)였다.


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


def _other_role(role: Role) -> Role:
    return Role.SELLER if role == Role.BUYER else Role.BUYER


def _price_out_of_bounds(price: float, episode: Episode) -> bool:
    """Appendix B.3 (i): p_min <= p_k <= p_max 위반."""
    return price < episode.p_min or price > episode.p_max


def _worse_than_own_reservation(turn: str, price: float, episode: Episode) -> bool:
    """Appendix B.3 (ii): 행위자 본인 reservation보다 엄격히 나쁜 가격을 offer/accept.

    "r_A"라는 논문 표기는 두 플레이어(A/B)를 일반화한 것 -- 실제로 행위자가
    agent 턴이면 role_A/r_A, counterpart 턴이면 그 반대 role/t_B.r을 쓴다.
    counterpart는 kernel.py의 클리핑(opening offer를 [r_B,p_max]/[p_min,r_B]로
    제한 + accept 전 favorability>=0 게이트)으로 구조적으로 이 조건에 걸릴 일이
    없다 -- 그래도 "누구 턴인지"에 무관하게 항상 성립해야 하는 일반 유틸리티로
    작성해, 나중에 kernel이 바뀌어도 안전하게 잡히도록 함.
    """
    role = episode.role_A if turn == "agent" else _other_role(episode.role_A)
    r = episode.r_A if turn == "agent" else episode.t_B.r
    return utility(role, price, r) < 0


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

# NEW (2026-07-26, pulse.pptx 슬라이드3 "결함->가치 매핑 로버스트니스" 실험 준비): 결함의
# "정체성"(종류/설명/quadrant)과 "가치"(listing_price 대비 몇 %를 깎는지)를 분리한다. 원래
# _DEFECT_CATALOG는 이 둘이 한 튜플에 섞여 있어서 price_impact가 절대금액 하나로 고정돼
# 있었는데, 교수님이 요구한 로버스트니스 체크(매핑을 보수적/중간/공격적/비선형 등 여러
# 버전으로 만들어서 agent 순위가 안 흔들리는지 확인)를 하려면 "같은 결함 정체성에 다른
# 가치 매핑"을 갈아끼울 수 있어야 한다. rng.sample()이 리스트의 개수/순서에만 의존하고
# 각 원소 안의 값 크기에는 의존하지 않으므로, 정체성 리스트(_DEFECT_IDENTITIES)만 매핑
# 간 고정해두면 같은 seed에서 "어떤 결함이 뽑히는지"는 매핑이 달라도 항상 동일하고
# price_impact 숫자만 달라진다 -- 이게 매핑 로버스트니스 실험에서 seed를 고정해 confound
# (매핑에 따른 협상 난이도 변화)를 결함 가치 변화 하나로만 좁히는 핵심 전제
# (benchmark/CLAUDE.md 2026-07-26 논의 참고).
_DEFECT_IDENTITIES = [
    # (kind, description, quadrant) -- pulse.pptx 슬라이드3의 매핑 표(스크래치/화면균열/
    # 부품누락, 2026-07-26)에 stain/rust 2종을 추가해 4개 사분면을 전부 커버한다 (2026-07-26
    # 결정, "quadrant 기반 utilization 지표"(quadrant_detection_rate/quadrant_calibration,
    # metrics.py) 설계 중 발견 -- 3종만으로는 obvious_misaligned/subtle_aligned 사분면
    # 결함이 생성 자체가 안 돼서 그 두 지표가 항상 None으로 죽는 문제가 있었음,
    # decisions_log.md 참고).
    # kind 문자열 (2026-07-28 수정): mock 전용 이름(screen_crack/hairline_crack)을 쓰지
    # 않고 **실 데이터의 defect_type 어휘 그대로** 쓴다 -- agent가 인용한 결함을 나중에
    # defect_type 단어 매칭으로 대조할 일이 생길 때 mock/실 데이터 어휘가 갈라져 있으면
    # 안 되므로. screen_crack -> crack, hairline_crack -> rust로 교체(quadrant 배치는
    # 그대로 유지: obvious_aligned 두 번째 슬롯, subtle_aligned 슬롯).
    # quadrant: scratch/crack/missing_part는 pulse.pptx 표 그대로. rust/stain은 슬라이드2
    # 원문 예시를 그대로 옮김: subtle_aligned 예시("모서리에 녹이 슬기 시작해서 자세히
    # 봐야 보이지만, 실제로 값을 떨어뜨림"), obvious_misaligned 예시("눈에 확 띄지만 실제
    # 가치엔 별 영향 없는 하자 -- 예: 쉽게 닦이는 얼룩").
    ("scratch", "Scratch on the surface", "obvious_aligned"),
    ("crack", "Crack on the surface", "obvious_aligned"),
    ("rust", "Rust starting to form in a hard-to-notice spot", "subtle_aligned"),
    ("stain", "Stain that wipes off easily", "obvious_misaligned"),
    ("missing_part", "Missing accessory/part", "subtle_misaligned"),
]

# 결함->가치 매핑 로버스트니스 실험용 프리셋 (2026-07-26 최초 작성, 2026-07-28 두 번 수정
# -- 아래 경위).
#
# !! 2026-07-28 경위 (두 번 뒤집힌 축) !! ① 최초(2026-07-26)엔 defect_type(kind) 축으로
# 올바르게 짰었다. ② 실 데이터 도착 후 팀원의 실제 생성 코드(summary_v1_0726.md)를 보고,
# "pulse.pdf 표 열을 severity(mild/moderate/severe) 예시로 재해석해서 구현했다"는 팀원
# 문서 설명을 그대로 믿고 축을 severity로 바꿨다. ③ 그런데 사용자가 pulse.pptx 원본
# 슬라이드("구체적 절차") 스크린샷을 직접 확인시켜줌 -- 표의 열 이름이 문자 그대로
# "스크래치/화면균열/부품누락"(defect_type)이었다. 즉 ①이 원래 맞았고, 팀원의 severity
# 재해석은 팀원 본인 생성 파이프라인만의 임의 변형이었음(교수님 원본과 다름). 그래서
# defect_type 축으로 다시 되돌린다 -- Defect.severity 필드는 그대로 남기되(실 데이터가
# 갖고 있는 정보라 보관 가치는 있음) price_impact 계산엔 이제 안 쓴다.
#
# 실 데이터의 defect_type 카탈로그는 7종(scratch/dent/rust/crack/stain/tear/missing_part)
# 인데 pulse.pptx 표는 3종(scratch/crack/missing_part -- "화면균열"은 실 데이터에서 전부
# electronics 카테고리에만 나오는 "crack"과 사실상 동일 개념으로 판단)만 커버한다.
# 나머지 4종(dent/rust/stain/tear, 결함 270개 중 140개로 절반 이상)은 교수님이 준 숫자가
# 없어서, 2026-07-28 사용자 결정("매핑을 늘리면 되잖아")에 따라 이미 있는 3개 퍼센트
# 티어에 유추 배정한다 -- **새 숫자를 지어낸 게 아니라 교수님이 승인한 퍼센트 3개를 더
# 많은 키에 재사용**하는 것: stain은 scratch와 같은 "외관/경미" 티어(둘 다 표면적이고
# 보통 큰 구조적 문제는 아님), dent/rust/tear는 crack과 같은 "중간/구조적" 티어(외관보다는
# 심하지만 부품 손실만큼 심각하진 않음)로 묶었다. missing_part 티어(가장 심각)에 준하는
# 결함 종류는 실 데이터에 따로 없어서 그대로 둠. 임의성은 "어느 티어에 배정하느냐"에만
# 있고 퍼센트 숫자 자체는 전부 pulse.pptx 원표 그대로다.
# 키는 딱 실 데이터의 defect_type 어휘 7종(scratch/dent/rust/crack/stain/tear/
# missing_part)만 쓴다 -- mock 전용 별칭(구 screen_crack/hairline_crack)을 안 두는 이유는
# 위 _DEFECT_IDENTITIES 주석 참고(2026-07-28, 사용자 결정): mock도 이제 이 7종 어휘를
# 그대로 쓰도록 바꿨으므로 별칭이 필요 없다.
PRICE_IMPACT_MAPPINGS: dict[str, dict[str, float]] = {
    "A_conservative": {"scratch": 0.05, "stain": 0.05, "crack": 0.15, "dent": 0.15, "rust": 0.15, "tear": 0.15, "missing_part": 0.20},
    "B_moderate":     {"scratch": 0.10, "stain": 0.10, "crack": 0.25, "dent": 0.25, "rust": 0.25, "tear": 0.25, "missing_part": 0.35},
    "C_aggressive":   {"scratch": 0.15, "stain": 0.15, "crack": 0.40, "dent": 0.40, "rust": 0.40, "tear": 0.40, "missing_part": 0.50},
    "D_nonlinear":    {"scratch": 0.08, "stain": 0.08, "crack": 0.30, "dent": 0.30, "rust": 0.30, "tear": 0.30, "missing_part": 0.45},
}


def price_impact_for(listing_price: float, defect_type: str, mapping: dict[str, float] | None = None) -> float:
    """defect_type("scratch"/"dent"/"rust"/"crack"/"stain"/"tear"/"missing_part") -> 실제
    할인 금액. mapping=None이면 PRICE_IMPACT_MAPPINGS["B_moderate"](2026-07-26 결정 근거와
    동일: 4개 후보 중 "중간"이라는 최소한의 근거가 있는 기본값). data_loader.py와
    sample_item() 양쪽에서 이 함수를 공유해서 실 데이터 경로와 mock 경로가 같은 계산을
    쓰도록 한다."""
    mapping = mapping if mapping is not None else PRICE_IMPACT_MAPPINGS["B_moderate"]
    return mapping[defect_type] * listing_price


def sample_item(
    rng: random.Random,
    listing_price: float,
    num_defects: int = 2,
    mapping: dict[str, float] | None = None,
) -> Item:
    """프로토타입용 mock item 생성기.

    실제 CraigslistBargain 이미지 + 결함 합성 파이프라인 연동은 스코프 밖
    (benchmark/CLAUDE.md "구현 범위" 참고) -- 지금은 grounding/utilization
    metric의 데이터 배관(파이프라인)이 실제로 작동하는지 검증하기 위한
    자리표시자 데이터를 만든다. image_ref도 실제 파일이 아니라 문자열
    placeholder.

    num_defects는 고정값으로 단순화 (원래는 0~N개로 다양화해야 하나, 오늘은
    "결함이 최소 1개는 있는 정상 케이스"부터 배관을 검증하는 게 우선).

    mapping: defect_type("scratch"/"dent"/"rust"/"crack"/"stain"/"tear"/"missing_part") ->
    listing_price 대비 퍼센트. None이면 PRICE_IMPACT_MAPPINGS["B_moderate"] (2026-07-26
    결정, 2026-07-28 재확인 -- 위 PRICE_IMPACT_MAPPINGS 주석 참고). data_loader.py가 실
    데이터의 defect_type 필드를 읽어 price_impact_for()에 넘기는 것과 정확히 같은 계산
    경로를 타므로, mock 경로와 실 데이터 경로가 서로 다른 축으로 갈라지지 않는다. 매핑
    로버스트니스 실험에서는 같은 rng 시퀀스에서 mapping 인자만 바꿔가며 여러 번 호출한다
    -- rng.sample이 _DEFECT_IDENTITIES의 개수/순서에만 의존하므로 "어떤 결함이 뽑히는지"는
    매핑이 달라도 동일하게 유지되고 price_impact 숫자만 달라진다 (위 _DEFECT_IDENTITIES/
    PRICE_IMPACT_MAPPINGS 주석 참고).
    """
    picks = rng.sample(_DEFECT_IDENTITIES, k=min(num_defects, len(_DEFECT_IDENTITIES)))
    defects = tuple(
        Defect(
            id=f"{kind}_{i}",
            description=desc,
            price_impact=price_impact_for(listing_price, kind, mapping),
            quadrant=quadrant,
            defect_type=kind,
            # severity는 지정 안 함 -> 기본값 None (mock 경로엔 실제 severity 라벨이 없음,
            # 위 Defect.severity 필드 주석 참고). salience도 지정 안 함 -> 기본값 None
            # (스코프 밖, Defect.salience 필드 주석 참고)
        )
        for i, (kind, desc, quadrant) in enumerate(picks)
    )
    return Item(
        category="furniture",
        title="Used item for sale",
        description="Gently used, see photos for details.",
        listing_price=listing_price,
        image_ref="placeholder://item.jpg",
        ground_truth_defects=defects,
    )


def fair_price(item: Item) -> float:
    """listing_price에서 실제 ground_truth_defects의 price_impact 합을 뺀 값 -- "결함까지
    반영한 진짜 공정가". TERMS-Bench §3.3(data-grounded extension)이 reference price
    주변에 reservation wedge를 앵커링하는 것과 같은 아이디어: listing_price는 결함을
    disclosure 안 한 액면가(data_spec.md "description에 결함을 절대 언급하지 말 것" 규칙과
    일치)고, 실제 가치는 그보다 price_impact 합만큼 낮다는 게 이 벤치마크의 전제다."""
    return item.listing_price - sum(d.price_impact for d in item.ground_truth_defects)


REGIMES: tuple[str, ...] = ("overlap", "urgency_shift", "no_deal")


def sample_episode(
    rng: random.Random,
    p_min: float = 0.0,
    p_max: float | None = None,
    K: int = 10,
    regimes: tuple[str, ...] = REGIMES,
    regime_weights: tuple[float, ...] | None = None,
    z_range: tuple[float, float] | None = None,
    q_range: tuple[float, float] | None = None,
    urgency_shift: float = 0.2,
    stance_prior: tuple[str, ...] = ("conciliatory", "neutral", "aggressive"),
    stance_weights: tuple[float, float, float] | None = None,
    harshness_range: tuple[float, float] = (0.20, 0.80),
    item: Item | None = None,
    role_A: Role | None = None,
    opener: str | None = None,
) -> Episode:
    """mu: episode의 regime, reservation, urgency, stance + item(evidence)을 샘플링.

    z/q 폭 샘플링(난이도 조절, regime별 feasible/infeasible 판정)은
    implementation/env.py와 동일하게 유지한다. 다만 ZOPA의 **중심**(m)은 더 이상
    임의의 uniform 난수가 아니라, item의 fair_price(listing_price - 결함 총액)에
    앵커링한다.

    p_max/z_range/q_range 스케일링 (2026-07-28 추가): 실 데이터는 listing_price가
    $20~$10,800로 넓게 퍼져있어서, 이 세 값이 절대 달러 고정값(옛 기본값 p_max=100,
    z_range=q_range=(5,40))이면 item.listing_price가 100을 넘는 순간부터 m이 p_max
    근처로 잘려 협상 범위가 실제 가격과 무관해진다.

    논문(§3.3/Appendix H.2.1)의 data-grounded 버전은 이걸 상품별 과거 최저/최고가
    (p_low/p_high)에서 유도한 분산 규모로 처리하는데, 우리 데이터는 아이템당
    listing_price 하나뿐이라 그 값이 없다 -- 그래서 논문 공식을 그대로 옮기지 못하고
    listing_price의 고정 퍼센트로 단순화한다(2026-07-28 사용자 결정, 데이터 한계로 인한
    근사 -- 논문 methods/limitations에 명시할 것): z(=r_buyer-r_seller, ZOPA 전체 폭)를
    listing_price의 15%~20%로 잡는다. p_max는 기본으로 listing_price를 쓰되(단독 호출
    시 안전한 하한), run_negotiation.py처럼 카테고리 최댓값 등 더 넓은 값을 알고 있는
    호출자는 p_max를 직접 넘겨 덮어쓸 수 있다.

    item=None(mock 경로)이면 이 자동 계산을 안 타고 옛 절대값 기본값(p_max=100,
    z_range=q_range=(5,40))을 그대로 쓴다 -- 기존 회귀 검증(6-family 300 episode 등,
    decisions_log.md)이 이 숫자에 의존하므로 하위호환 유지.

    role_A/opener (2026-07-30 추가): 기본(None)이면 예전처럼 rng.choice로 무작위 배정한다
    -- 하위호환. 값을 넘기면 그 값을 강제로 쓰고 rng.choice 자체를 안 부른다 (rng 소비
    횟수가 줄어들지만, episode_rng로 episode마다 독립 rng를 쓰는 호출부만 이 인자를 쓰므로
    다른 episode의 샘플링에 영향 없음). 논문(4.1절)은 role×opener를 25episode/cell로
    균등 배정하는 block design을 쓰는데, 우리 sample_episode의 기존 rng.choice는 완전
    무작위라 n=100 배치에서 실제로 칸당 3~15개로 쏠리는 게 관측됨(2026-07-30,
    decisions_log.md) -- 호출부(run_negotiation.py의 role_opener_for)가 episode_idx를
    4칸에 순환 배정해서 이 인자로 강제하면 균등 배정을 재현할 수 있다.
    """
    weights = list(regime_weights) if regime_weights else [1.0] * len(regimes)
    regime = rng.choices(regimes, weights=weights)[0]

    if item is not None:
        p_max = p_max if p_max is not None else item.listing_price
        z_range = z_range if z_range is not None else (0.15 * item.listing_price, 0.20 * item.listing_price)
        q_range = q_range if q_range is not None else (0.15 * item.listing_price, 0.20 * item.listing_price)
        episode_item = item
    else:
        p_max = p_max if p_max is not None else 100.0
        z_range = z_range if z_range is not None else (5.0, 40.0)
        q_range = q_range if q_range is not None else (5.0, 40.0)
        episode_item = sample_item(rng, listing_price=(p_min + p_max) / 2)
    m_anchor = fair_price(episode_item)

    if regime in ("overlap", "urgency_shift"):
        z = rng.uniform(*z_range)
        m = min(max(m_anchor, p_min + z / 2), p_max - z / 2)
        r_buyer, r_seller = m + z / 2, m - z / 2
    else:  # no_deal
        q = rng.uniform(*q_range)
        m = min(max(m_anchor, p_min + q / 2), p_max - q / 2)
        r_buyer, r_seller = m - q / 2, m + q / 2

    urgency_B = sample_urgency(rng, shifted=(regime == "urgency_shift"), shift=urgency_shift)
    stance_B = rng.choices(stance_prior, weights=stance_weights, k=1)[0]  # weights=None -> 균등 (regime과 동일 패턴)

    role_A = role_A if role_A is not None else rng.choice([Role.BUYER, Role.SELLER])
    opener = opener if opener is not None else rng.choice(["AgentOpens", "CounterpartOpens"])
    harshness = rng.uniform(*harshness_range)

    r_A = r_buyer if role_A == Role.BUYER else r_seller
    r_B = r_seller if role_A == Role.BUYER else r_buyer

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
    """Alternating-offer protocol 루프. implementation/env.py의 run_episode를 뼈대로 하되,
    Appendix B.3의 (i) 가격범위 / (ii) IR 위반 감지를 추가한다 (2026-07-13, 이전엔
    accept_before_any_offer 한 종류만 감지 -- paper/paper-code-map.md 5.2절, benchmark/
    CLAUDE.md 2026-07-12 항목에 기록된 버그). cited_defect_ids는 Action의 필드일 뿐이라
    루프 구조 자체는 바뀌지 않는다.

    accept_before_any_offer와 달리 (i)/(ii) 위반은 즉시 DISAGREEMENT로 끊지 않는다 --
    "받아들일 대상 자체가 없어 결과가 정의 불가능"한 accept_before_any_offer와 달리,
    가격범위/IR 위반은 "이상하지만 결과는 정의 가능한 제안"이라 violations에 기록만
    하고 협상을 그대로 진행시킨다 (LLM이 이상한 값을 내도 벤치마크가 죽지 않아야 한다는
    implementation/env.py의 기존 철학과 동일).
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

        if action.decision == Decision.OFFER:
            if _price_out_of_bounds(action.price, episode):
                violations.append((k, turn, "price_out_of_bounds"))
            if _worse_than_own_reservation(turn, action.price, episode):
                violations.append((k, turn, "ir_violation"))
        elif action.decision == Decision.ACCEPT and _worse_than_own_reservation(
            turn, last_offer[_other(turn)], episode
        ):
            violations.append((k, turn, "ir_violation"))

        history.append((k, turn, action))

        if action.decision == Decision.ACCEPT:
            return EpisodeResult(outcome=last_offer[_other(turn)], history=history, violations=violations)
        if action.decision == Decision.REJECT:
            return EpisodeResult(outcome=DISAGREEMENT, history=history, violations=violations)

        # OFFER: record and pass the turn
        last_offer[turn] = action.price
        turn = _other(turn)

    return EpisodeResult(outcome=DISAGREEMENT, history=history, violations=violations)  # round-limit disagreement
