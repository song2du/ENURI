"""Headline diagnostic metrics -- implementation/metrics.py의 Table 1 5종을 뼈대로,
evidence-based negotiation 전용 신규 지표(citation_precision/citation_coverage/
hallucination_rate/utilization_proximity)를 추가한다.

설계 논의 전체 기록: benchmark/CLAUDE.md의 "구현 범위 -- 메트릭" 절 참고.
implementation/metrics.py를 import하지 않고 독립적으로 복사+확장했다 (이유는
env.py 모듈 docstring과 동일).

BE_type(belief_error)은 오늘 설계 논의 대상이 아니었으므로 이번 프로토타입에는
포함하지 않는다 -- "안 되는 게 확인돼서 뺀" 게 아니라 "아직 논의를 안 한" 상태.
"""

from __future__ import annotations

from dataclasses import dataclass

from env import DISAGREEMENT, Decision, Episode, EpisodeResult, Role, utility


@dataclass
class EpisodeRecord:
    episode: Episode
    result: EpisodeResult


# ---------------------------------------------------------------------------
# 원자 함수 + 필터 
# ---------------------------------------------------------------------------


def zopa_delta(episode: Episode) -> float:
    """Delta_i = r_buyer - r_seller. Positive -> feasible (I+), negative -> infeasible (I-)."""
    r_buyer = episode.r_A if episode.role_A == Role.BUYER else episode.t_B.r
    r_seller = episode.t_B.r if episode.role_A == Role.BUYER else episode.r_A
    return r_buyer - r_seller


def agent_utility(episode: Episode, result: EpisodeResult) -> float:
    return utility(episode.role_A, result.outcome, episode.r_A)


def _feasible(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
    return [r for r in records if zopa_delta(r.episode) > 0]


def _infeasible(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
    return [r for r in records if zopa_delta(r.episode) < 0]


def _agreed(records: list[EpisodeRecord]) -> list[EpisodeRecord]:
    return [r for r in records if r.result.outcome is not DISAGREEMENT]


# ---------------------------------------------------------------------------
# Table 1 5종 (implementation/metrics.py와 동일 -- z/q geometry를 안 건드렸으니 그대로 재사용 가능)
# ---------------------------------------------------------------------------


def se_plus(records: list[EpisodeRecord]) -> float | None:
    """Surplus extracted, normalized by ZOPA width, over feasible episodes I+."""
    feasible = _feasible(records)
    if not feasible:
        return None
    normalized = [agent_utility(r.episode, r.result) / zopa_delta(r.episode) for r in feasible]
    return sum(normalized) / len(feasible)


def agr_plus(records: list[EpisodeRecord]) -> float | None:
    """Fraction of feasible episodes that reach agreement. Higher is better."""
    feasible = _feasible(records)
    if not feasible:
        return None
    return len(_agreed(feasible)) / len(feasible)


def cse_plus(records: list[EpisodeRecord]) -> float | None:
    """Normalized surplus conditional on agreement, over feasible+agreed episodes I_agr+."""
    agreed_feasible = _agreed(_feasible(records))
    if not agreed_feasible:
        return None
    normalized = [agent_utility(r.episode, r.result) / zopa_delta(r.episode) for r in agreed_feasible]
    return sum(normalized) / len(agreed_feasible)


def agr_minus(records: list[EpisodeRecord]) -> float | None:
    """Fraction of infeasible episodes that (wrongly) reach agreement. Lower is better."""
    infeasible = _infeasible(records)
    if not infeasible:
        return None
    return len(_agreed(infeasible)) / len(infeasible)


def crit_viol_pct(records: list[EpisodeRecord]) -> float | None:
    """Fraction of episodes with at least one logged protocol violation. Lower is better."""
    if not records:
        return None
    return sum(1 for r in records if r.result.violations) / len(records)


# ---------------------------------------------------------------------------
# Grounding accuracy -- NEW
# ---------------------------------------------------------------------------


def _agent_citations(result: EpisodeResult) -> set[str]:
    """이 episode에서 agent가 언급한 모든 defect id의 합집합 (여러 턴에 걸친 인용을 다 모음). C에 해당."""
    ids: set[str] = set()
    for (_, side, action) in result.history:
        if side == "agent" and action.cited_defect_ids:
            ids.update(action.cited_defect_ids)
    return ids


def citation_precision(records: list[EpisodeRecord]) -> float | None:
    """|C ∩ D| / |C|, episode마다 계산 후 평균. C가 빈(한 번도 인용 안 한) episode는
    0/0이라 정의 안 됨 -- 평균 분모에서 제외 (se_plus의 _feasible 빈 경우 패턴과 동일)."""
    scored = []
    for r in records:
        C = _agent_citations(r.result)
        if not C:
            continue
        D = {d.id for d in r.episode.item.ground_truth_defects}
        scored.append(len(C & D) / len(C))
    if not scored:
        return None
    return sum(scored) / len(scored)


def citation_coverage(records: list[EpisodeRecord]) -> float | None:
    """|C ∩ D| / |D| (recall), episode마다 계산 후 전체 episode에 대해 평균 -- 인용을
    아예 안 한 episode도 '커버리지 0%'라는 유효한 신호라 제외하지 않음 (citation_precision과 분모 처리가 다름, 의도적)."""
    if not records:
        return None
    scored = []
    for r in records:
        D = {d.id for d in r.episode.item.ground_truth_defects}
        if not D:
            continue  # ground truth 결함이 아예 없는 아이템이면 recall 자체가 정의 안 됨 (지금 sample_item은 항상 >=1개 만들지만, 방어적으로 guard)
        C = _agent_citations(r.result)
        scored.append(len(C & D) / len(D))
    if not scored:
        return None
    return sum(scored) / len(scored)


def hallucination_rate(records: list[EpisodeRecord]) -> float | None:
    """C \\ D(실제 없는 결함을 지어낸 것)가 한 번이라도 있었던 episode의 비율.
    CritViol%처럼 평균이 아니라 episode 단위 이진 플래그의 비율로 정의 -- citation_precision의
    단순 여집합(1-precision)이 되지 않도록 일부러 다르게 설계함 (benchmark/CLAUDE.md 참고:
    평균 정확도가 높아도 위험한 케이스가 존재하는지는 별도 축으로 봐야 함)."""
    if not records:
        return None
    flagged = 0
    for r in records:
        C = _agent_citations(r.result)
        D = {d.id for d in r.episode.item.ground_truth_defects}
        if C - D:
            flagged += 1
    return flagged / len(records)


# ---------------------------------------------------------------------------
# Utilization proximity -- NEW
# ---------------------------------------------------------------------------


def _agent_sign(role_A: Role) -> float:
    """kernel.py의 _agent_sign과 동일한 관례: BUYER는 가격 상승이 양보, SELLER는 가격 하락이 양보라 부호로 통일."""
    return 1.0 if role_A == Role.BUYER else -1.0


def utilization_proximity(records: list[EpisodeRecord]) -> float | None:
    """citation turn(agent 액션의 cited_defect_ids가 비어있지 않고 decision==OFFER인 턴)의
    offer가 agent 자신의 직전 offer 대비 양보였는지로 '활용됐는지' 판정한다.
    episode당 (활용된 citation 수 / 전체 citation 수)를 낸 뒤 여러 episode에 걸쳐 평균.

    !! Simplification (benchmark/CLAUDE.md 참고, 재검토 필요) !!
    ACCEPT/REJECT 턴에 붙은 citation(가격이 없어 '양보인지' 판정 불가)은 이 metric
    집계 대상에서 제외한다. agent의 첫 offer가 곧 citation turn인 경우(직전 offer가
    없음)도 '활용 안 됨'으로 보수적으로 처리한다.
    """
    scored = []
    for r in records:
        sign = _agent_sign(r.episode.role_A)
        agent_offer_prices: list[float] = []
        utilized = 0
        total_citations = 0
        for (_, side, action) in r.result.history:
            if side != "agent" or action.decision != Decision.OFFER:
                continue
            if action.cited_defect_ids:
                total_citations += 1
                if agent_offer_prices:  # 직전 offer가 있어야 '양보인지' 판정 가능
                    concession = sign * (action.price - agent_offer_prices[-1])
                    if concession > 0:
                        utilized += 1
            agent_offer_prices.append(action.price)
        if total_citations == 0:
            continue
        scored.append(utilized / total_citations)
    if not scored:
        return None
    return sum(scored) / len(scored)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_metrics(records: list[EpisodeRecord]) -> dict[str, float | None]:
    return {
        "SE+": se_plus(records),
        "AGR+": agr_plus(records),
        "CSE+": cse_plus(records),
        "AGR-": agr_minus(records),
        "CritViol%": crit_viol_pct(records),
        "citation_precision": citation_precision(records),
        "citation_coverage": citation_coverage(records),
        "hallucination_rate": hallucination_rate(records),
        "utilization_proximity": utilization_proximity(records),
    }
