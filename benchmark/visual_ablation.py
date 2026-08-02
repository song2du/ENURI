"""visual 유/무 baseline 실험 오케스트레이션 (pulse.pptx 슬라이드6, benchmark/CLAUDE.md "다음 작업" 참고).

절차:
① agent 로스터를 정한다 -- **LLM 기반 agent만 대상**이다. FC baseline(`fixed_concession_agent.py`)은
   애초에 이미지를 전혀 안 보는 규칙 기반이라 (`run_negotiation.py`의 `--no-image` 설명: "--fc-rate는
   원래 이미지를 안 보므로 무관") on/off를 나눌 대상 자체가 아니다.
② agent마다 **같은 100개 episode**를 `use_image=True`(이미지 있음)/`False`(이미지 있어도 안 보냄)
   두 조건으로 각각 배치 실행한다. `_has_real_image`(데이터에 실제 이미지가 있는가, 데이터 사실)와
   `use_image`(이번 실행이 그걸 실제로 보낼지, 실험 조건)가 `llm_agent.py`에서 이미 분리돼 있어서,
   같은 아이템에 대해 "이미지를 보여줬을 때"와 "보여줬으면 있었을 이미지를 일부러 숨겼을 때"를
   그대로 대조할 수 있다.
③ 두 조건의 `compute_metrics()` 결과 차이 `gap = on - off`를 모든 metric에 대해 계산한다 --
   "이미지가 있을 때 SE+/CSE+/citation류 지표가 실제로 개선되는가"가 이 실험의 핵심 질문
   (CLAUDE.md: "visual evidence 없으면 협상 결과가 실제로 달라지는가" gap 검증).

**Confound 통제 (mapping_robustness.py와 동일한 이유, 2026-07-30)**: agent/image 조건에
상관없이 매 episode_idx가 항상 같은 시나리오(아이템+역할+유보가격+상대방 성향+K)에서
시작해야 "이미지 유무"만 비교 대상이 된다. `episode_rng(seed, i)`로 매 episode 독립적으로
rng를 새로 만든다 (run_negotiation.py의 episode_rng docstring 참고 -- FC-1/FC-10 배치
비교에서 발견된 rng drift 버그의 수정 방식을 그대로 재사용). 이미지 on/off는 애초에
episode 샘플링에 관여하지 않으므로(프롬프트에 뭘 실어 보내느냐의 차이일 뿐), 이 통제
없이도 아이템 순서 자체는 같았겠지만, 역할/유보가격/K 등 나머지 조건까지 완전히 고정하려면
동일하게 필요하다.

**로스터 (2026-07-30, 배관 검증 단계)**: 아직 `OPENROUTER_API_KEY`가 없어서 OpenAI 직결
모델만 포함한다 -- `gpt-4o`(메인 비교 대상) + `gpt-4o-mini`(CLAUDE.md "평가 대상 모델"이
요구하는 약한 모델 바닥 앵커, 지금까지 어느 로스터에도 없었음). 키 도착 후 Claude/Gemini/
Qwen도 `provider="openrouter"`로 추가 가능 (`mapping_robustness.py`의 `default_agents()`와
같은 교체 지점).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from data_loader import load_items
from env import REGIMES, Item
from kernel import make_counterpart_policy, stance_prior_for
from llm_agent import make_llm_agent_policy
from metrics import EpisodeRecord, compute_metrics
from run_negotiation import category_item_schedule, episode_rng, role_opener_for, run_one_episode
from voice import add_voice

# 리포트 저장 기본 위치 -- mapping_robustness.json과 같은 이유(2026-07-28)로 휘발 방지.
DEFAULT_OUT_PATH = Path(__file__).parent / "result" / "visual_ablation.json"

IMAGE_STATES = (True, False)  # (on, off) 순서 고정 -- gap 계산(on - off)과 리포트 출력 순서가 여기 의존


@dataclass
class AgentConfig:
    label: str
    model: str
    provider: str = "openai"  # 키 도착 후 openrouter 모델 추가할 때 여기만 바꾸면 됨


def default_agents() -> list[AgentConfig]:
    """배관 검증용 고정 로스터 (모듈 docstring 참고) -- OPENROUTER_API_KEY 도착 후 Claude/Qwen 등 추가."""
    return [
        AgentConfig("gpt-4o", "gpt-4o"),
        AgentConfig("gpt-4o-mini", "gpt-4o-mini"),
    ]


def _category_p_max(items: list[Item]) -> dict[str, float]:
    """run_negotiation.py main()/mapping_robustness.py와 동일한 계산 (카테고리 내 listing_price 최댓값)."""
    category_p_max: dict[str, float] = {}
    for it in items:
        category_p_max[it.category] = max(category_p_max.get(it.category, 0.0), it.listing_price)
    return category_p_max


def run_cell(
    agent_cfg: AgentConfig,
    use_image: bool,
    item_schedule: list[Item],
    category_p_max: dict[str, float],
    *,
    episodes: int,
    seed: int,
    family: str,
    regime: str,
) -> list[EpisodeRecord]:
    """(agent, image on/off) 한 칸 -- episodes개 배치 실행.

    seed는 image 조건에 상관없이 매 episode_idx가 동일한 시나리오에서 시작한다 (모듈 docstring
    "Confound 통제" 참고) -- episode_rng(seed, i)를 매 episode 새로 만들어야 이 통제가 성립한다
    (mapping_robustness.py run_cell과 동일한 패턴). regime 고정 + role×opener 균등 배정도
    같은 이유로 동일하게 적용 (run_negotiation.py --regime, decisions_log.md 2026-07-30).
    item_schedule도 마찬가지로 category_item_schedule(run_negotiation.py, 2026-08-02)이
    episode_idx 기준으로 미리 만든 카테고리 균등 배정 스케줄이라 image on/off 무관 i가 같으면
    항상 같은 카테고리.
    """
    agent_policy = make_llm_agent_policy(model=agent_cfg.model, provider=agent_cfg.provider, use_image=use_image)
    counterpart_policy = add_voice(make_counterpart_policy(family))
    stance_weights = stance_prior_for(family)

    records = []
    for i in range(1, episodes + 1):
        item = item_schedule[i - 1]
        p_max = category_p_max[item.category]
        ep_rng = episode_rng(seed, i)
        role_A, opener = role_opener_for(i)
        records.append(run_one_episode(
            ep_rng, agent_policy, counterpart_policy, stance_weights, item, p_max,
            regimes=(regime,), role_A=role_A, opener=opener,
        ))
    return records


def gap_report(on_metrics: dict, off_metrics: dict) -> dict:
    """gap[k] = on[k] - off[k], 둘 다 숫자일 때만 계산 -- 한쪽이라도 None(표본 부족 등으로
    정의 안 됨)이면 gap도 None (mapping_robustness.py의 _rank_vector와 같은 이유: 값이
    없는 걸 억지로 0 취급하면 "차이가 없다"로 오독될 수 있음).

    참고 (2026-07-30, 스모크테스트로 정정): `*_n` 류(표본 수) 키가 전부 image 조건과
    무관하게 같은 건 아니다. `detection_rate_{quadrant}_n`(metrics.py `_recall`)만
    D=ground-truth defect 집합(아이템이 정함, 이미지 전송 여부와 무관)에서 오므로 gap이
    항상 0이어야 정상 -- 이게 깨지면 버그 신호. 반면 `severity_calibration_n`/
    `calibration_rho_{quadrant}_n`/`subtle_misaligned_gap_n_caught`/`_n_missed`는
    agent가 실제로 몇 번 인용했는지(`_agent_citations`, 행동)에서 오는 값이라 image
    조건에 따라 달라지는 게 오히려 정상이고 기대되는 신호다 (이미지를 못 보면 인용
    자체를 못 하니까) -- 이 실험이 재려는 것 중 하나이지 오염이 아니다."""
    gap: dict[str, float | None] = {}
    for k, v_on in on_metrics.items():
        v_off = off_metrics.get(k)
        if isinstance(v_on, (int, float)) and isinstance(v_off, (int, float)):
            gap[k] = v_on - v_off
        else:
            gap[k] = None
    return gap


def main() -> None:
    parser = argparse.ArgumentParser(description="visual 유/무 baseline 오케스트레이션 (pulse.pptx 슬라이드6)")
    parser.add_argument("--data", type=str, required=True, help="실 데이터셋 results.jsonl 경로")
    parser.add_argument("--episodes", type=int, default=100, help="(agent, image on/off) 한 칸당 episode 수")
    parser.add_argument("--seed", type=int, default=1, help="image 조건 무관 매 episode 공통 시드 (confound 통제)")
    parser.add_argument("--family", type=str, default="Candid", help="counterpart family (메인 실험 축 고정 결정, CLAUDE.md 참고)")
    parser.add_argument(
        "--regime", type=str, default="overlap", choices=list(REGIMES),
        help="episode regime 고정 (기본 overlap, run_negotiation.py --regime과 동일 이유, decisions_log.md 2026-07-30)",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help=f"결과 JSON 저장 경로 (기본값: {DEFAULT_OUT_PATH})")
    args = parser.parse_args()

    agents = default_agents()
    items = load_items(args.data)
    category_p_max = _category_p_max(items)
    # 카테고리 균등 배정 스케줄 (2026-08-02, run_negotiation.py의 category_item_schedule 참고) --
    # 셔플만으로는 카테고리 비율이 운에 맡겨져서(예: car 38개가 --episodes 100 기준 적게/안
    # 뽑힐 위험) 카테고리 marginal을 role_opener_for처럼 정확히 균등 배정하는 걸로 교체했다.
    item_schedule = category_item_schedule(items, args.episodes, args.seed)

    print(f"agents={[a.label for a in agents]} episodes/cell={args.episodes} family={args.family} regime={args.regime}")
    print()

    # scores[agent_label]["on"|"off"] = compute_metrics(records)
    scores: dict[str, dict[str, dict]] = {}
    gaps: dict[str, dict] = {}

    for agent_cfg in agents:
        scores[agent_cfg.label] = {}
        for use_image in IMAGE_STATES:
            state_key = "on" if use_image else "off"
            records = run_cell(
                agent_cfg, use_image, item_schedule, category_p_max,
                episodes=args.episodes, seed=args.seed, family=args.family, regime=args.regime,
            )
            m = compute_metrics(records)
            scores[agent_cfg.label][state_key] = m
            print(f"[{agent_cfg.label}] image={state_key:3s}: SE+={m['SE+']} citation_coverage={m['citation_coverage']}")
        gaps[agent_cfg.label] = gap_report(scores[agent_cfg.label]["on"], scores[agent_cfg.label]["off"])
        print()

    print("=== visual on/off gap (on - off, 양수면 이미지가 있을 때 더 좋음) ===")
    for label, gap in gaps.items():
        print(f"-- {label} --")
        for k, v in gap.items():
            v_str = f"{v:+.3f}" if v is not None else "None"
            print(f"  {k}: {v_str}")
        print()

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
                    "regime": args.regime,
                    "agents": [a.label for a in agents],
                },
                "scores": scores,
                "gaps": gaps,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"(결과를 {out_path}에 저장)")


if __name__ == "__main__":
    main()
