"""Counterpart kernel pi_B -- implementation/kernel.py(Sec 3.2 Eq.5-9, Appendix C.1-C.5)를
뼈대로 하되, evidence(결함 인용) 축을 accept_prob에 추가한다.

설계 논의 전체 기록: benchmark/CLAUDE.md의 "구현 범위 -- 커널" 절 참고.
implementation/kernel.py를 import하지 않고 독립적으로 복사+확장했다 (이유는
env.py 모듈 docstring과 동일 -- implementation/은 학습용, benchmark/는 결과물).

프로토타입 스코프: 6 family 중 `Candid` 하나만 구현한다 (benchmark/CLAUDE.md
2026-07-11 결정 -- 가장 특이 성향 없는 베이스라인으로 grounding/utilization
metric 배관부터 노이즈 없이 검증하기 위함). 두 번째 family(예: Adversarial)는
시간이 남으면 ECON_PRESETS/FAMILIES에 항목을 추가하는 것만으로 확장 가능하도록
구조는 유지해둔다.
"""

from __future__ import annotations

import math
import random

from env import Action, Decision, Episode, Item, Role

# ---------------------------------------------------------------------------
# A. 공용 유틸 (implementation/kernel.py A섹션과 동일)
# ---------------------------------------------------------------------------


def other_role(role_A: Role) -> Role:
    return Role.SELLER if role_A == Role.BUYER else Role.BUYER


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def deadline_remaining(k: int, K: int) -> float:
    """D_tilde_bar_k = 1 - sqrt(k/K): 시간이 얼마나 남았는지 (accept_prob에서 사용)."""
    return 1.0 - math.sqrt(k / K)


def deadline_proximity(k: int, K: int) -> float:
    """D_tilde_k = sqrt(k/K): 마감에 얼마나 가까운지 (strategic cue 모델에서 사용, deadline_remaining과 반대 방향이니 헷갈리지 말 것)."""
    return math.sqrt(k / K)


def favorability(offer_price: float, r_B: float, role_B: Role, R: float) -> float:
    """Delta_bar_k: 이 제안이 counterpart한테 얼마나 유리한가 (role 정규화, IR 게이트에 사용). >=0이면 원칙적으로 받아들일 만함."""
    if role_B == Role.SELLER:
        return (offer_price - r_B) / R
    return (r_B - offer_price) / R


def _extract_prices(history: list[tuple[int, str, Action]], side: str) -> list[float]:
    return [a.price for (_, s, a) in history if s == side and a.decision == Decision.OFFER]


def _extract_actions(history: list[tuple[int, str, Action]], side: str) -> list[Action]:
    """_extract_prices와 같은 필터지만 Action 객체 전체를 반환한다.

    NEW (benchmark 전용): evidence_term_for가 cited_defect_ids를 읽어야 해서,
    가격뿐 아니라 Action 전체가 필요하다 -- implementation/kernel.py에는 없던 헬퍼.
    """
    return [a for (_, s, a) in history if s == side and a.decision == Decision.OFFER]


# ---------------------------------------------------------------------------
# B. 히스토리 기반 feature (implementation/kernel.py B섹션과 동일, Eq.12-14)
# ---------------------------------------------------------------------------

TAU_RIGID = 0.10


def _agent_sign(role_A: Role) -> float:
    return 1.0 if role_A == Role.BUYER else -1.0


def concede_magnitude_speed(agent_prices: list[float], role_A: Role, R: float) -> tuple[float, float]:
    """ConcedeMagnitude_k, ConcedeSpeed_k: 최근(최대 3개) 연속 offer 차이의 평균, role 정규화해서 양수=항상 양보."""
    s = _agent_sign(role_A)
    window = agent_prices[-4:]
    if len(window) < 2:
        return 0.0, 0.0
    diffs = [s * (window[i] - window[i - 1]) / R for i in range(1, len(window))]
    magnitude = sum(max(0.0, d) for d in diffs) / len(diffs)
    speed = sum(diffs) / len(diffs)
    return magnitude, speed


def rigidity(agent_prices: list[float], role_A: Role, R: float) -> float:
    """Rigidity_k: agent의 가장 최근 완료된 양보가 tau_rigid보다 작았으면 1."""
    if len(agent_prices) < 3:
        return 0.0
    s = _agent_sign(role_A)
    last_move = max(0.0, s * (agent_prices[-2] - agent_prices[-3])) / R
    return 1.0 if last_move < TAU_RIGID else 0.0


def counterpart_concession_c(own_prices: list[float], r_B: float, eps_c: float = 1e-6) -> float:
    """C^B_k: counterpart 자신의 최근 정규화된 양보 -- cue 모델 전용, 경제적 결정과 무관."""
    if len(own_prices) < 2:
        return 0.0
    p_prev, p_cur = own_prices[-2], own_prices[-1]
    return min(1.0, abs(p_cur - p_prev) / (abs(p_prev - r_B) + eps_c))


# ---------------------------------------------------------------------------
# C. Accept/Walkaway (implementation/kernel.py C섹션 + evidence 항 추가, 2026-07-11 결정)
# ---------------------------------------------------------------------------

ALPHA, BETA, GAMMA = 6.0, 1.0, 2.0
K_WALK_FRAC = 0.5
PHI0, PHI_DELTA, PHI_T = -4.5, 30.0, 1.5

EVIDENCE_BONUS = 2.0  # NEW: benchmark/CLAUDE.md "커널 -- evidence를 accept_prob에 반영" 참고. 크기는 프로토타입 단계 임시값 -- 실제로 돌려보고 튜닝 필요.


def evidence_term_for(agent_action: Action | None, item: Item, role_A: Role) -> float:
    """agent의 이번 offer가 실제 결함을 인용했는지에 따른 accept_prob 보정항.

    benchmark/CLAUDE.md 2026-07-11 결정 (두 번 수정된 최종안):
    - BUYER가 실제 결함을 인용하며 (낮은) 가격을 부르면: 그 가격의 근거가
      강해지므로 +EVIDENCE_BONUS -- counterpart(SELLER)가 더 받아들이기 쉬워짐.
    - SELLER가 자기 결함을 인정하면서 가격을 안 낮추면: 그 (높은) 가격의 근거가
      약해지므로 -EVIDENCE_BONUS -- counterpart(BUYER)가 덜 받아들이려 함.
    - 지어낸(ground truth에 없는) 결함 인용은 이 항에 영향 없음 -- 할루시네이션은
      metrics.py의 hallucination_rate가 별도로 채점한다. 여기서 경제적으로
      보상/처벌하지 않는 이유: 두 관심사(경제적 협상 역학 vs. 사실관계 정확성
      채점)를 분리해서 커널을 단순하게 유지하기 위함.

    agent_action이 None이거나 cited_defect_ids가 비어있으면 0.0 (효과 없음).
    """
    if agent_action is None or not agent_action.cited_defect_ids:
        return 0.0
    real_ids = {d.id for d in item.ground_truth_defects}
    cited_real_defect = any(cid in real_ids for cid in agent_action.cited_defect_ids)
    if not cited_real_defect:
        return 0.0
    sign = _agent_sign(role_A)  # BUYER -> +1, SELLER -> -1. B섹션의 부호 관례를 그대로 재사용.
    return sign * EVIDENCE_BONUS


def accept_prob(
    delta_bar: float,
    urgency: float,
    D_tilde_bar: float,
    concede_speed: float,
    rigid: float,
    rho: float,
    xi: float,
    evidence_term: float = 0.0,  # NEW
) -> float:
    if delta_bar < 0:
        return 0.0  # IR 게이트: 아무리 근거가 좋아도 손해 보는 제안은 무조건 거절. evidence_term은 이 게이트를 못 뚫는다 -- 게이트를 통과한(원칙적으로 받아들일 만한) 제안의 확률만 조정한다.
    g = ALPHA * delta_bar + BETA * urgency - GAMMA * D_tilde_bar + rho * concede_speed + xi * rigid + evidence_term
    return sigmoid(g)


def walkaway_prob(delta_bar: float, k: int, K: int) -> float:
    k_walk = math.ceil(K * K_WALK_FRAC)
    if k < k_walk or delta_bar >= 0:
        return 0.0
    tau_W = min(1.0, max(0.0, (k - k_walk) / (K - k_walk)))
    return sigmoid(PHI0 + PHI_DELTA * max(0.0, -delta_bar) + PHI_T * tau_W)


def resolve_decision(accept_p: float, walkaway_p: float, rng: random.Random) -> Decision:
    """pi_B(Accept)=a_k, pi_B(Reject)=(1-a_k)*w_k, pi_B(Offer)=(1-a_k)(1-w_k)."""
    u = rng.random()
    if u < accept_p:
        return Decision.ACCEPT
    if u < accept_p + (1 - accept_p) * walkaway_p:
        return Decision.REJECT
    return Decision.OFFER


# ---------------------------------------------------------------------------
# D. Counter-offer / concession (implementation/kernel.py D섹션과 동일, Eq.8-9)
# ---------------------------------------------------------------------------

LAMBDA0, LAMBDA1, LAMBDA3, LAMBDA4 = 0.12, 0.28, 0.10, 0.10


def concession_rate(urgency: float, stance: str, concede_magnitude: float, lambda2: float) -> float:
    raw = (
        LAMBDA0
        + LAMBDA1 * urgency
        - lambda2 * concede_magnitude
        - LAMBDA3 * (stance == "aggressive")
        + LAMBDA4 * (stance == "conciliatory")
    )
    return min(1.0, max(0.0, raw))


def counter_offer(prev_own_price: float, r_B: float, role_B: Role, lam: float, noise_std: float, rng: random.Random) -> float:
    candidate = prev_own_price - lam * (prev_own_price - r_B) + rng.gauss(0, noise_std)
    if role_B == Role.SELLER:
        return min(max(candidate, r_B), prev_own_price)  # M_B(k) = [r_B, p_{k-1}^B]
    return max(min(candidate, r_B), prev_own_price)  # M_B(k) = [p_{k-1}^B, r_B]


# ---------------------------------------------------------------------------
# E. Opening-offer 모델 (implementation/kernel.py E섹션과 동일, Eq.15-16)
# ---------------------------------------------------------------------------

OMEGA_KAPPA, OMEGA_ETA, OMEGA_ETA2 = 0.30, 0.15, 0.15
PHI_MIN, PHI_MAX = 0.5, 1.5
SIGMA0_BAR = 0.02


def opening_modulation(urgency: float, stance: str) -> float:
    raw = 1 - OMEGA_KAPPA * urgency + OMEGA_ETA * (stance == "aggressive") - OMEGA_ETA2 * (stance == "conciliatory")
    return min(PHI_MAX, max(PHI_MIN, raw))


def opening_offer(
    r_B: float, role_B: Role, p_min: float, p_max: float, harshness: float, urgency: float, stance: str, R: float, rng: random.Random
) -> float:
    if role_B == Role.SELLER:
        slack, direction, lo, hi = (p_max - r_B), 1.0, r_B, p_max
    else:
        slack, direction, lo, hi = (r_B - p_min), -1.0, p_min, r_B
    phi = opening_modulation(urgency, stance)
    noise = rng.gauss(0, SIGMA0_BAR * R)
    candidate = r_B + direction * harshness * phi * slack + noise
    return min(max(candidate, lo), hi)


# ---------------------------------------------------------------------------
# F. Cue 생성 (implementation/kernel.py F섹션과 동일, Appendix C.5.1-C.5.3) -- "말투", 경제적 결정과 무관
# ---------------------------------------------------------------------------

MU_S, TAU_S, SIGMA_S = 1.0, 0.5, 0.75
B_C, B_H, B_P = 1.0, 0.5, 1.0
ALPHA_C, ALPHA_P, BETA_C = 2.0, 2.0, 1.0
TAU_CONC, TAU_DEAD = 0.10, 0.80
SIGMA_S_STOCH, T_STOCH = 2.0, 2.5


def sentiment_cue_base(stance: str, rng: random.Random, sigma: float = SIGMA_S) -> str:
    mu = {"conciliatory": MU_S, "neutral": 0.0, "aggressive": -MU_S}[stance]
    z = rng.gauss(mu, sigma)
    if z > TAU_S:
        return "positive"
    if z < -TAU_S:
        return "negative"
    return "neutral"


def _softmax_sample(logits: dict[str, float], temperature: float, rng: random.Random) -> str:
    keys = list(logits.keys())
    scaled = [logits[key] / temperature for key in keys]
    m = max(scaled)
    weights = [math.exp(v - m) for v in scaled]
    total = sum(weights)
    probs = [w / total for w in weights]
    u = rng.random()
    cumulative = 0.0
    for key, p in zip(keys, probs):
        cumulative += p
        if u < cumulative:
            return key
    return keys[-1]


def strategic_cue_base(
    decision: Decision, stance: str, concede_c_B: float, deadline_D_tilde: float, rng: random.Random, temperature: float = 1.0
) -> str:
    if decision == Decision.ACCEPT:
        return "Concede"
    if decision == Decision.REJECT:
        return "Pressure"
    bias = {
        "conciliatory": {"Concede": B_C, "Hold": 0.0, "Pressure": -B_C},
        "neutral": {"Concede": 0.0, "Hold": B_H, "Pressure": 0.0},
        "aggressive": {"Concede": -B_P, "Hold": 0.0, "Pressure": B_P},
    }[stance]
    logits = {
        "Concede": bias["Concede"] + ALPHA_C * (concede_c_B - TAU_CONC),
        "Hold": bias["Hold"],
        "Pressure": bias["Pressure"] + ALPHA_P * (deadline_D_tilde - TAU_DEAD) - BETA_C * concede_c_B,
    }
    return _softmax_sample(logits, temperature, rng)


def sample_cues(
    decision: Decision, stance: str, cue_type: str, concede_c_B: float, deadline_D_tilde: float, rng: random.Random
) -> tuple[str, str]:
    if cue_type == "accurate":
        return sentiment_cue_base(stance, rng), strategic_cue_base(decision, stance, concede_c_B, deadline_D_tilde, rng)
    if cue_type == "uninformative":
        return "neutral", "Hold"
    if cue_type == "pressuring":
        return "negative", "Pressure"
    if cue_type == "noisy":
        s = sentiment_cue_base(stance, rng, sigma=SIGMA_S_STOCH)
        c = strategic_cue_base(decision, stance, concede_c_B, deadline_D_tilde, rng, temperature=T_STOCH)
        return s, c
    raise ValueError(f"unknown cue_type: {cue_type}")


# ---------------------------------------------------------------------------
# G. Family preset -- 프로토타입은 Candid 하나만 (benchmark/CLAUDE.md 2026-07-11 결정)
# ---------------------------------------------------------------------------

STANCES = ("conciliatory", "neutral", "aggressive")

ECON_PRESETS = {
    "type_instrumental": {  # Candid가 쓰는 프리셋. 다른 프리셋(high_reactivity 등)은 2번째 family 추가할 때 implementation/kernel.py G섹션에서 옮겨오면 됨.
        "rho": (0.0, -0.25, -0.75),
        "xi": (0.40, 0.0, -0.50),
        "lambda2": (0.30, 0.50, 1.00),
        "noise_std_frac": 0.01,
    },
}

FAMILIES = {
    "Candid": {"econ": "type_instrumental", "cue": "accurate", "stance_prior": None},
    # 2번째 family는 시간 남으면 여기 + ECON_PRESETS에 항목 추가 (benchmark/CLAUDE.md 참고, 예: Adversarial).
}


def stance_prior_for(family_name: str) -> tuple[float, float, float]:
    """Pr(stance = C, N, A). Candid는 균등."""
    override = FAMILIES[family_name]["stance_prior"]
    return override if override is not None else (1 / 3, 1 / 3, 1 / 3)


def _lookup_by_stance(preset: dict, stance: str) -> tuple[float, float, float]:
    idx = STANCES.index(stance)
    return preset["rho"][idx], preset["xi"][idx], preset["lambda2"][idx]


# ---------------------------------------------------------------------------
# H. Orchestrator
# ---------------------------------------------------------------------------


def make_counterpart_policy(family_name: str):
    """Returns a policy(episode, history, side, rng) -> Action closure bound to one family.

    family_name은 클로저에 갇혀서 Episode에 저장되지 않는다 -- family 정체가
    로그에 남지 않도록 (agent에게 숨겨야 함, implementation/kernel.py와 동일 이유).
    """
    econ_key = FAMILIES[family_name]["econ"]
    cue_type = FAMILIES[family_name]["cue"]
    preset = ECON_PRESETS[econ_key]

    def policy(episode: Episode, history: list[tuple[int, str, Action]], side: str, rng: random.Random) -> Action:
        t_B = episode.t_B
        role_B = other_role(episode.role_A)
        R = episode.p_max - episode.p_min
        k = len(history) + 1
        agent_actions = _extract_actions(history, "agent")  # NEW: evidence_term_for가 필요로 하는 Action 전체 (가격만으론 부족)
        agent_prices = [a.price for a in agent_actions]
        own_prices = _extract_prices(history, "counterpart")
        rho, xi, lambda2 = _lookup_by_stance(preset, t_B.stance)

        if not agent_prices:
            price = opening_offer(
                t_B.r, role_B, episode.p_min, episode.p_max, episode.harshness, t_B.urgency, t_B.stance, R, rng
            )
            decision = Decision.OFFER
        else:
            delta_bar = favorability(agent_prices[-1], t_B.r, role_B, R)
            D_tilde_bar = deadline_remaining(k, episode.K)
            magnitude, speed = concede_magnitude_speed(agent_prices, episode.role_A, R)
            rigid = rigidity(agent_prices, episode.role_A, R)
            evidence_term = evidence_term_for(agent_actions[-1], episode.item, episode.role_A)  # NEW

            a_p = accept_prob(delta_bar, t_B.urgency, D_tilde_bar, speed, rigid, rho, xi, evidence_term)
            w_p = walkaway_prob(delta_bar, k, episode.K)
            decision = resolve_decision(a_p, w_p, rng)

            if decision == Decision.OFFER:
                if not own_prices:
                    price = opening_offer(
                        t_B.r, role_B, episode.p_min, episode.p_max, episode.harshness, t_B.urgency, t_B.stance, R, rng
                    )
                else:
                    lam = concession_rate(t_B.urgency, t_B.stance, magnitude, lambda2)
                    noise_std = preset["noise_std_frac"] * R
                    price = counter_offer(own_prices[-1], t_B.r, role_B, lam, noise_std, rng)
            else:
                price = None

        concede_c_B = counterpart_concession_c(own_prices, t_B.r)
        D_tilde = deadline_proximity(k, episode.K)
        sentiment, posture = sample_cues(decision, t_B.stance, cue_type, concede_c_B, D_tilde, rng)

        # counterpart는 자기 자신에 대해 결함을 "인용"하지 않는다 -- cited_defect_ids는 agent 전용 필드.
        return Action(decision=decision, price=price, sentiment=sentiment, posture=posture)

    return policy
