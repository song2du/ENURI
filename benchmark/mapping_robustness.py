"""매핑 로버스트니스 실험 오케스트레이션 (pulse.pptx 슬라이드3, benchmark/CLAUDE.md "다음 작업" 참고).

절차 (슬라이드3 원문 ①~④, Paper/pulse_extract.txt):
① 결함->가치 매핑을 여러 버전(env.py의 PRICE_IMPACT_MAPPINGS, 4개) 준비 -- 이미 돼 있음.
② 매핑마다 --data 데이터셋 전체를 각 agent로 돌려서 배치 metric을 낸다.
③ agent들 사이의 순위를 매핑끼리 Spearman's rho로 비교.
④ rho가 낮은 지점("빵꾸")을 진단 -- 전체 episode 합산 + quadrant별(4개) 두 단위로 계산.

**순위 지표 2026-07-29 결정**: SE+(원 논문 headline, price_impact가 fair_price/ZOPA를 옮기므로
매핑에 실제로 민감)를 주 지표로, utilization_proximity(이 벤치마크의 새 축 -- 증거를 양보로
연결했는가)를 보조 지표로 둘 다 계산/리포트한다.

**Confound 통제 (2026-07-26 decisions_log.md)**: 매핑을 바꾸면 채점 기준뿐 아니라 fair_price
경유로 ZOPA 자체도 같이 움직인다 -- "전체 벤치마크를 돌린다"(슬라이드3 ②)가 그런 의도라고
판단했기 때문. 그래서 매핑끼리 난이도가 아니라 "매핑 선택"만 비교되도록, 매핑에 상관없이
매 (agent, mapping) 셀마다 동일한 시드에서 시작한다 -- sample_item/데이터 순서가 시드에만
의존하므로 어떤 결함이 뽑히는지는 매핑 무관 고정, price_impact 숫자만 매핑 따라 달라진다.

**지금은 배관 검증 단계 (2026-07-29 결정)**: OPENROUTER_API_KEY 미발급 상태라 Claude/Gemini/
Qwen을 아직 못 붙인다. DEFAULT_AGENTS는 FC-1/10/30 baseline + gpt-4o뿐 -- 4개 매핑 x 4개
agent x 4개 quadrant 전체 파이프라인이 안 죽고 도는지, rho 계산까지 끝까지 나오는지 확인하는
용도다. 지금 나오는 rho 숫자 자체(agent가 4개뿐, 그중 3개는 LLM도 아님)는 슬라이드3가 말하는
"본 벤치마크의 순위가 매핑에 안 흔들린다" 주장의 근거가 못 된다 -- 키 도착 후 DEFAULT_AGENTS를
실제 로스터로 교체해서 재실행할 것.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from data_loader import load_items
from env import PRICE_IMPACT_MAPPINGS, Item
from fixed_concession_agent import FC_RATES, make_fixed_concession_policy
from kernel import FAMILIES, make_counterpart_policy, stance_prior_for
from llm_agent import make_llm_agent_policy
from metrics import QUADRANTS, EpisodeRecord, compute_metrics, spearman_rho
from run_negotiation import run_one_episode
from voice import add_voice

# 리포트 저장 기본 위치 -- episodes.jsonl(run_negotiation.py)과 같은 이유(2026-07-28)로
# 휘발 방지: 이 스크립트도 stdout만 찍으면 프로세스 종료와 함께 결과가 사라진다.
DEFAULT_OUT_PATH = Path(__file__).parent / "result" / "mapping_robustness.json"

# 순위를 매길 metric -- compute_metrics()가 반환하는 dict의 키. 둘 다 리포트한다 (모듈
# docstring의 "순위 지표 2026-07-29 결정" 참고).
RANKING_METRICS = ("SE+", "utilization_proximity")


@dataclass
class AgentConfig:
    label: str
    make_policy: Callable[[], Callable]  # 인자 없이 바로 agent policy를 만드는 클로저 (지연 생성)


def default_agents() -> list[AgentConfig]:
    """배관 검증용 고정 로스터 (모듈 docstring 참고) -- OPENROUTER_API_KEY 도착 후 교체."""
    return [
        AgentConfig("FC-1", lambda: make_fixed_concession_policy(FC_RATES["FC-1"])),
        AgentConfig("FC-10", lambda: make_fixed_concession_policy(FC_RATES["FC-10"])),
        AgentConfig("FC-30", lambda: make_fixed_concession_policy(FC_RATES["FC-30"])),
        AgentConfig("gpt-4o", lambda: make_llm_agent_policy(model="gpt-4o", provider="openai")),
    ]


def _category_p_max(items: list[Item]) -> dict[str, float]:
    """run_negotiation.py main()과 동일한 계산(카테고리 내 listing_price 최댓값) --
    한 곳(여기)에서만 하면 되도록 별도 함수로 뽑음."""
    category_p_max: dict[str, float] = {}
    for it in items:
        category_p_max[it.category] = max(category_p_max.get(it.category, 0.0), it.listing_price)
    return category_p_max


def run_cell(
    agent_cfg: AgentConfig,
    items: list[Item],
    category_p_max: dict[str, float],
    *,
    episodes: int,
    seed: int,
    family: str,
) -> list[EpisodeRecord]:
    """(agent, mapping) 한 칸 -- episodes개 배치 실행.

    seed는 agent_cfg/items(=mapping)에 상관없이 매 셀마다 동일한 값에서 새로 시작한다
    (모듈 docstring "Confound 통제" 참고) -- rng를 셀마다 새로 만들어야 이 통제가 성립한다.
    """
    rng = random.Random(seed)
    agent_policy = agent_cfg.make_policy()
    counterpart_policy = add_voice(make_counterpart_policy(family))
    stance_weights = stance_prior_for(family)

    records = []
    for i in range(1, episodes + 1):
        item = items[(i - 1) % len(items)]
        p_max = category_p_max[item.category]
        records.append(run_one_episode(rng, agent_policy, counterpart_policy, stance_weights, item, p_max))
    return records


def records_by_quadrant(records: list[EpisodeRecord]) -> dict[str, list[EpisodeRecord]]:
    """episode의 quadrant = 그 episode 아이템의 유일한 ground-truth 결함(아이템당 0-1개,
    data_spec.md 규칙)의 quadrant. 결함이 없는 아이템의 episode는 어느 quadrant 버킷에도
    안 들어간다 -- quadrant_detection_rate 등 기존 metric과 동일한 취급."""
    buckets: dict[str, list[EpisodeRecord]] = {q: [] for q in QUADRANTS}
    for r in records:
        defects = r.episode.item.ground_truth_defects
        if not defects:
            continue
        buckets[defects[0].quadrant].append(r)
    return buckets


def _rank_vector(scores: dict[str, dict], agents: list[AgentConfig], metric: str) -> list[float] | None:
    """mapping 하나의 scores[agent_label] -> metric 벡터. agent 중 하나라도 그 metric이
    None(표본 부족 등으로 정의 안 됨)이면 그 매핑에서 순위 자체가 안 나오므로 None 반환."""
    values = [scores[a.label].get(metric) for a in agents]
    if any(v is None for v in values):
        return None
    return values


def rho_report(
    scores_by_mapping: dict[str, dict[str, dict]], agents: list[AgentConfig], mappings: list[str]
) -> dict[str, dict[str, float | None]]:
    """모든 매핑 쌍(uv, C(n,2)개) x RANKING_METRICS 별 spearman_rho. 슬라이드3 ③~④가
    "모든 매핑 쌍"이라고 했지 인접 쌍만이 아니므로 itertools.combinations로 전체 쌍을 본다."""
    report: dict[str, dict[str, float | None]] = {}
    for metric in RANKING_METRICS:
        pair_rhos: dict[str, float | None] = {}
        for m1, m2 in itertools.combinations(mappings, 2):
            xs = _rank_vector(scores_by_mapping[m1], agents, metric)
            ys = _rank_vector(scores_by_mapping[m2], agents, metric)
            rho = spearman_rho(xs, ys) if xs is not None and ys is not None else None
            pair_rhos[f"{m1} vs {m2}"] = rho
        report[metric] = pair_rhos
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="매핑 로버스트니스 오케스트레이션 (pulse.pptx 슬라이드3)")
    parser.add_argument("--data", type=str, required=True, help="실 데이터셋 results.jsonl 경로")
    parser.add_argument("--episodes", type=int, default=40, help="(mapping, agent) 한 칸당 episode 수")
    parser.add_argument("--seed", type=int, default=1, help="매핑/agent 무관 매 셀 공통 시드 (confound 통제)")
    parser.add_argument("--family", type=str, default="Candid", choices=list(FAMILIES.keys()))
    parser.add_argument(
        "--mappings", type=str, nargs="+", default=list(PRICE_IMPACT_MAPPINGS.keys()),
        choices=list(PRICE_IMPACT_MAPPINGS.keys()), help="비교할 매핑 부분집합 (기본: 전체 4개)",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help=f"결과 JSON 저장 경로 (기본값: {DEFAULT_OUT_PATH})")
    args = parser.parse_args()

    agents = default_agents()
    mappings = args.mappings

    print(f"mappings={mappings} agents={[a.label for a in agents]} episodes/cell={args.episodes} family={args.family}")
    print()

    # mapping -> items: data_loader.load_items가 defect_type 기반으로 price_impact를 매핑별로
    # 다시 계산한다 (결함 identity는 파일 순서에서만 오므로 매핑 무관 고정, 2026-07-26 결정).
    items_by_mapping = {m: load_items(args.data, mapping=PRICE_IMPACT_MAPPINGS[m]) for m in mappings}
    category_p_max_by_mapping = {m: _category_p_max(items) for m, items in items_by_mapping.items()}

    # scores_by_mapping[mapping][agent_label] = compute_metrics(records) (전체 episode 합산)
    # quadrant_scores_by_mapping[mapping][quadrant][agent_label] = compute_metrics(quadrant 부분집합)
    scores_by_mapping: dict[str, dict[str, dict]] = {}
    quadrant_scores_by_mapping: dict[str, dict[str, dict[str, dict]]] = {}

    for m in mappings:
        scores_by_mapping[m] = {}
        quadrant_scores_by_mapping[m] = {q: {} for q in QUADRANTS}
        items = items_by_mapping[m]
        category_p_max = category_p_max_by_mapping[m]

        for agent_cfg in agents:
            records = run_cell(agent_cfg, items, category_p_max, episodes=args.episodes, seed=args.seed, family=args.family)
            scores_by_mapping[m][agent_cfg.label] = compute_metrics(records)

            for q, q_records in records_by_quadrant(records).items():
                # compute_metrics는 빈 리스트도 안전하게 처리(전부 None) -- 빈/비어있지 않은
                # 케이스를 분기하지 않아야 모든 셀이 같은 키 구조를 갖는다.
                q_metrics = compute_metrics(q_records)
                q_metrics["n"] = len(q_records)
                quadrant_scores_by_mapping[m][q][agent_cfg.label] = q_metrics

            print(f"[{m}] {agent_cfg.label}: SE+={scores_by_mapping[m][agent_cfg.label]['SE+']}")

    overall_rho = rho_report(scores_by_mapping, agents, mappings)

    quadrant_rho: dict[str, dict[str, dict[str, float | None]]] = {}
    for q in QUADRANTS:
        q_scores_by_mapping = {m: quadrant_scores_by_mapping[m][q] for m in mappings}
        quadrant_rho[q] = rho_report(q_scores_by_mapping, agents, mappings)

    print()
    print("=== overall ranking robustness (전체 episode 합산) ===")
    for metric, pair_rhos in overall_rho.items():
        print(f"-- metric={metric} --")
        for pair, rho in pair_rhos.items():
            print(f"  {pair}: rho={rho}")

    print()
    print("=== per-quadrant ranking robustness ===")
    for q in QUADRANTS:
        print(f"-- quadrant={q} --")
        for metric, pair_rhos in quadrant_rho[q].items():
            print(f"  metric={metric}")
            for pair, rho in pair_rhos.items():
                print(f"    {pair}: rho={rho}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "data": args.data,
                    "episodes_per_cell": args.episodes,
                    "seed": args.seed,
                    "family": args.family,
                    "mappings": mappings,
                    "agents": [a.label for a in agents],
                },
                "scores_by_mapping": scores_by_mapping,
                "quadrant_scores_by_mapping": quadrant_scores_by_mapping,
                "overall_rho": overall_rho,
                "quadrant_rho": quadrant_rho,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n(결과를 {out_path}에 저장)")


if __name__ == "__main__":
    main()
