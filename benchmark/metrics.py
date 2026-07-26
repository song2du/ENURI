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


def _recall(records: list[EpisodeRecord], quadrant: str | None = None) -> tuple[float | None, int]:
    """|C ∩ D| / |D| recall 평균 + 표본(episode) 수 -- citation_coverage/quadrant_detection_rate
    (NEW, 2026-07-26)가 공유하는 계산. quadrant가 주어지면 D를 그 사분면 결함으로만 제한한다.
    인용을 아예 안 한 episode도 '커버리지 0%'라는 유효한 신호라 제외하지 않음
    (citation_precision과 분모 처리가 다름, 의도적) -- D가 빈(이 사분면 결함이 아예 없는
    아이템, 또는 quadrant=None인데 결함 자체가 없는) episode만 제외."""
    scored = []
    for r in records:
        D = {d.id for d in r.episode.item.ground_truth_defects if quadrant is None or d.quadrant == quadrant}
        if not D:
            continue
        C = _agent_citations(r.result)
        scored.append(len(C & D) / len(D))
    if not scored:
        return None, 0
    return sum(scored) / len(scored), len(scored)


def citation_coverage(records: list[EpisodeRecord]) -> float | None:
    """|C ∩ D| / |D| (recall), episode마다 계산 후 전체 episode에 대해 평균 (`_recall` 참고)."""
    rate, _ = _recall(records)
    return rate


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


# NEW (2026-07-26, 교수님 코멘트 "결함 정보를 어디까지 활용하냐" -- benchmark/decisions_log.md
# 참고). 2x2 매트릭스가 재는 능력은 사실 두 개뿐이다: Detection(결함을 알아챘는가)과
# Calibration(알아챘다면 크기를 진짜 가치에 맞게 반영했는가). 가시성 축(Obvious/Subtle)이
# Detection 난이도를, 정합성 축(Aligned/Misaligned)이 Calibration 난이도를 조절한다(방향이
# 아니라 난이도 -- Aligned는 보이는 대로가 진실이라 착시 없음=쉬움, Misaligned는 보이는
# 것과 진실이 어긋나 착시 있음=어려움). "과잉반응 방지"(obvious_misaligned)는 별개의 능력이
# 아니라 Calibration을 억제 방향에서 본 것. 그래서 사분면마다 새 공식을 만들지 않고, 기존
# citation_coverage/severity_calibration을 사분면으로 슬라이스한 quadrant_detection_rate/
# quadrant_calibration 두 함수만 추가한다.
_QUADRANTS = ("obvious_aligned", "subtle_aligned", "obvious_misaligned", "subtle_misaligned")


def quadrant_detection_rate(records: list[EpisodeRecord], quadrant: str) -> tuple[float | None, int]:
    """citation_coverage(`_recall` 참고)를 quadrant 하나로 좁힌 버전 -- 그 사분면 결함이
    있는 아이템에서 그 결함들 중 얼마나 인용됐는지(recall) + 표본(episode) 수. n을 같이
    반환하는 이유는 subtle_misaligned_gap/severity_calibration과 동일(표본이 적을 때
    숫자만 보고 오독하는 것 방지)."""
    return _recall(records, quadrant=quadrant)


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
# 결함 활용 "깊이" 지표 -- NEW (2026-07-23, 교수님 코멘트: "결함 정보를 어디까지
# 활용하냐"는 새 축. citation_precision/coverage/hallucination_rate(언급했나/맞게
# 언급했나)와 달리, 이 둘은 "언급한 게 실제로 얼마나 의미 있게 반영됐나"를 잰다.
# ---------------------------------------------------------------------------


def _rank(values: list[float]) -> list[float]:
    """평균 순위(tie는 평균 순위로 처리) -- Spearman rho 계산용 보조함수.
    scipy 없이 순수 파이썬으로 구현 (프로젝트 전체가 표준 라이브러리만 씀)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed, tie구간은 평균
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None  # 한쪽이 상수(분산 0)면 상관계수 정의 안 됨
    return cov / (var_x * var_y) ** 0.5


def _spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """pulse.pptx가 이미 매핑-robustness 체크에 쓰던 것과 같은 지표(Spearman's rho)를
    여기서도 재사용 -- 순위 상관이라 x/y의 절대 스케일이 달라도(price_impact는 달러,
    가격변화는 R로 정규화된 비율) 그대로 비교 가능."""
    n = len(xs)
    if n < 2:
        return None
    return _pearson(_rank(xs), _rank(ys))


def _severity_calibration_pairs(records: list[EpisodeRecord], quadrant: str | None = None) -> tuple[list[float], list[float]]:
    """단일 인용 citation turn에서 (price_impact(x), |가격변화|/R(y)) 쌍을 모은다 --
    severity_calibration/quadrant_calibration(NEW, 2026-07-26)이 공유하는 수집 로직.
    quadrant가 주어지면 인용된 결함의 quadrant가 일치하는 턴만 모은다 (그 외 필터는
    severity_calibration과 완전히 동일 -- 단일 인용만, 지어낸 결함 제외, 방향 무시하고
    절댓값 사용; 근거는 severity_calibration docstring 참고)."""
    xs: list[float] = []
    ys: list[float] = []
    for r in records:
        defect_by_id = {d.id: d for d in r.episode.item.ground_truth_defects}
        R = r.episode.p_max - r.episode.p_min
        prev_price: float | None = None
        for (_, side, action) in r.result.history:
            if side != "agent" or action.decision != Decision.OFFER:
                continue
            if action.cited_defect_ids and len(action.cited_defect_ids) == 1 and prev_price is not None:
                cited_id = action.cited_defect_ids[0]
                if cited_id in defect_by_id:  # 지어낸 결함이면 x(price_impact)가 없어 제외
                    d = defect_by_id[cited_id]
                    if quadrant is None or d.quadrant == quadrant:
                        xs.append(d.price_impact)
                        ys.append(abs(action.price - prev_price) / R)
            prev_price = action.price
    return xs, ys


def severity_calibration(records: list[EpisodeRecord]) -> tuple[float | None, int]:
    """단일 인용 citation turn에서, 인용된 결함의 ground-truth price_impact(x)와 그 턴의
    가격 변화 크기(|가격변화|/R, y) 사이의 Spearman rho. 높을수록 "결함이 심각할수록 실제로
    더 크게 가격에 반영한다"는 calibration이 잘 된 것 (benchmark/CLAUDE.md 2026-07-23 결정).

    !! 설계 결정 (전부 CLAUDE.md에 기록됨) !!
    - 결함을 정확히 **하나만** 인용한 턴에 한정한다 -- 다중 인용시 어느 결함의
      price_impact를 x로 삼을지 애매해서(합/최댓값/평균 중 택1 필요) 프로토타입에서는
      제외. 지금 아이템당 결함이 1~2개뿐이라 이 제약이 거의 안 걸림 (idea.md 2026-07-15
      태그 이슈와 같은 시점에 재검토).
    - 지어낸(ground truth에 없는) 단일 인용은 제외한다 -- price_impact 자체가 정의 안 됨.
    - 방향(concession인지)은 안 본다 -- 그건 utilization_proximity의 몫. 여기선 순수하게
      "크기가 심각도에 비례하는가"만 보므로 부호 없는 절댓값을 쓴다 (agent가 심각한 결함을
      인용하고도 안 움직이거나 반대로 움직이면 그 자체로 낮은 y라 이미 calibration
      실패로 잡힘 -- 굳이 부호를 따로 볼 필요 없음).

    반환값 (rho, n) -- n(표본turn 수)을 항상 같이 반환해 오독 방지 (subtle_misaligned_gap과
    같은 이유).

    !! 주의 (2026-07-26) !! 여기 비교 대상인 price_impact는 절대적 진실이 아니라
    env.py의 SEVERITY_MAPPINGS에서 우리가 고른 매핑값이다 -- "calibration이 잘 됐다"는
    결과는 "그 매핑을 기준으로" 잘 됐다는 뜻이지, 매핑 선택과 무관한 절대적 결론이
    아니다 (decisions_log.md 2026-07-26 참고).
    """
    xs, ys = _severity_calibration_pairs(records)
    return _spearman_rho(xs, ys), len(xs)


def quadrant_calibration(records: list[EpisodeRecord], quadrant: str) -> tuple[float | None, int]:
    """severity_calibration을 quadrant 하나로 좁힌 버전 -- 인용된 결함의 quadrant==quadrant인
    단일-인용 턴만으로 Spearman rho 계산. 나머지 설계 결정/주의사항은 severity_calibration과
    동일 (docstring 참고)."""
    xs, ys = _severity_calibration_pairs(records, quadrant=quadrant)
    return _spearman_rho(xs, ys), len(xs)


def subtle_misaligned_gap(records: list[EpisodeRecord]) -> tuple[float | None, int, int]:
    """pulse.pptx 슬라이드2의 가장 어려운 4분면(Subtle-Misaligned -- 안 보이지만 가치
    영향이 큰 결함)이 있는 아이템의 episode들을, 그 결함을 잡아냈는지(cited_defect_ids에
    한 번이라도 등장) 여부로 두 그룹으로 나눠 CSE+ 차이를 본다:
    gap = CSE+(잡음) - CSE+(놓침). 값이 클수록 "이 킬러 결함을 잡아내는 게 실제로
    협상 결과에 크게 도움이 된다"는 뜻 (benchmark/CLAUDE.md 2026-07-23 결정).

    CSE+ 계산은 cse_plus를 그대로 재사용 -- feasible+agreed 조건부 정의가 Table 1 5종과
    어긋나지 않게 하기 위함 (직접 필터링하지 않고 위임).

    !! 주의 !!
    - 무작위 개입이 아니라 agent가 스스로 인용했는지로 나눈 **관찰적** 비교다 -- 순수
      인과추론은 아니고 "인용 능력과 결과가 얼마나 같이 움직이는가"에 가까움.
    - 표본 크기는 구현이 아니라 데이터(이 quadrant 아이템이 몇 개나 준비되는지) 문제 --
      n_caught/n_missed을 gap과 항상 같이 리포트해서, 표본이 적을 때 숫자만 보고
      오독하는 것을 방지한다.
    """
    caught: list[EpisodeRecord] = []
    missed: list[EpisodeRecord] = []
    for r in records:
        subtle_misaligned_ids = {
            d.id for d in r.episode.item.ground_truth_defects if d.quadrant == "subtle_misaligned"
        }
        if not subtle_misaligned_ids:
            continue  # 이 아이템엔 Subtle-Misaligned 결함이 아예 없음 -- 비교 대상 아님
        cited = _agent_citations(r.result)
        (caught if cited & subtle_misaligned_ids else missed).append(r)

    n_caught, n_missed = len(caught), len(missed)
    cse_caught = cse_plus(caught)
    cse_missed = cse_plus(missed)
    if cse_caught is None or cse_missed is None:
        return None, n_caught, n_missed
    return cse_caught - cse_missed, n_caught, n_missed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_metrics(records: list[EpisodeRecord]) -> dict[str, float | None]:
    sev_rho, sev_n = severity_calibration(records)
    sm_gap, sm_n_caught, sm_n_missed = subtle_misaligned_gap(records)
    result = {
        "SE+": se_plus(records),
        "AGR+": agr_plus(records),
        "CSE+": cse_plus(records),
        "AGR-": agr_minus(records),
        "CritViol%": crit_viol_pct(records),
        "citation_precision": citation_precision(records),
        "citation_coverage": citation_coverage(records),
        "hallucination_rate": hallucination_rate(records),
        "utilization_proximity": utilization_proximity(records),
        "severity_calibration_rho": sev_rho,
        "severity_calibration_n": sev_n,
        "subtle_misaligned_gap": sm_gap,
        "subtle_misaligned_gap_n_caught": sm_n_caught,
        "subtle_misaligned_gap_n_missed": sm_n_missed,
    }
    # NEW (2026-07-26): 4개 사분면 x 2개 축(detection/calibration) 균일 리포트 -- "진짜
    # 시험하는 것"만 선택적으로 내지 않고 8개(+n 4쌍) 다 낸다. 예상 밖의 사분면에서 구멍이
    # 발견될 수 있다는 이유로 전부 리포트하기로 함 (decisions_log.md 2026-07-26 참고).
    for q in _QUADRANTS:
        det_rate, det_n = quadrant_detection_rate(records, q)
        cal_rho, cal_n = quadrant_calibration(records, q)
        result[f"detection_rate_{q}"] = det_rate
        result[f"detection_rate_{q}_n"] = det_n
        result[f"calibration_rho_{q}"] = cal_rho
        result[f"calibration_rho_{q}_n"] = cal_n
    return result
