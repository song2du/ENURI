"""(agent, mapping) 셀 하나가 중간에 죽었을 때, 이미 끝난 episode를 버리지 않고
`--start-episode`부터 이어서 채우는 복구용 스크립트 (2026-08-05 추가).

`mapping_robustness.py`의 `run_cell()`은 항상 episode 1부터 돌게만 되어있어서(재개
기능 없음), 정식 실행 중 크래시(sleep모드/오버플로/API 500/Gemini 일일 요청 한도 등,
decisions_log.md 2026-08-04~05 참고)가 나면 그 셀을 통째로 다시 도는 수밖에 없었다.
이 스크립트는 `run_cell()`의 핵심 로직(item_schedule/category_p_max/episode_rng/
role_opener_for/run_one_episode)을 그대로 재사용하되, 범위만 `[start_episode, episodes]`로
좁혀서 이미 로그에 남아있는 episode는 건드리지 않고 그 뒤부터만 이어붙인다.

**주의**: `--data`/`--mapping`/`--episodes`/`--seed`를 원래 돌렸던 값과 똑같이 줘야
item_schedule/episode_rng/role_opener_for가 동일한 시퀀스를 재현한다 (안 그러면 episode
72가 원래 돌았던 72와 다른 시나리오가 되어 다른 agent들과의 confound 통제가 깨짐).
`--episodes-log-path`는 반드시 append로 이어붙일 기존 파일을 가리켜야 한다 (log_file은
항상 "a" 모드로 연다).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_loader import load_items
from env import PRICE_IMPACT_MAPPINGS
from kernel import make_counterpart_policy, stance_prior_for
from llm_agent import make_llm_agent_policy
from mapping_robustness import _category_p_max
from run_negotiation import category_item_schedule, episode_rng, episode_to_dict, role_opener_for, run_one_episode
from voice import add_voice


def main() -> None:
    parser = argparse.ArgumentParser(description="중간에 죽은 (agent, mapping) 셀을 start-episode부터 이어서 채운다")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mapping", type=str, required=True, choices=list(PRICE_IMPACT_MAPPINGS.keys()))
    parser.add_argument("--agent-label", type=str, required=True, help="episode_to_dict의 run_meta에 남길 라벨 (예: gemini-3.1-pro-preview)")
    parser.add_argument("--model", type=str, required=True, help="llm_agent.make_llm_agent_policy의 model 인자")
    parser.add_argument("--provider", type=str, required=True, choices=["openai", "openrouter", "google"])
    parser.add_argument("--episodes", type=int, default=100, help="원래 셀의 총 episode 수 (원래 실행값과 반드시 동일해야 함)")
    parser.add_argument("--start-episode", type=int, required=True, help="여기부터 이어서 돈다 (1-based, 예: 71개 이미 있으면 72)")
    parser.add_argument("--seed", type=int, default=1, help="원래 실행값과 반드시 동일해야 함")
    parser.add_argument("--family", type=str, default="Candid")
    parser.add_argument("--regime", type=str, default="overlap")
    parser.add_argument("--episodes-log-path", type=str, required=True, help="append로 이어붙일 기존 episodes.jsonl 경로")
    args = parser.parse_args()

    items = load_items(args.data, mapping=PRICE_IMPACT_MAPPINGS[args.mapping])
    category_p_max = _category_p_max(items)
    # episodes(총량)/seed가 원래 실행과 같아야 item_schedule이 동일하게 재현된다 --
    # category_item_schedule은 (items, episodes, seed)로 결정론적이다.
    item_schedule = category_item_schedule(items, args.episodes, args.seed)

    agent_policy = make_llm_agent_policy(model=args.model, provider=args.provider)
    counterpart_policy = add_voice(make_counterpart_policy(args.family))
    stance_weights = stance_prior_for(args.family)
    run_meta = {"mapping": args.mapping, "agent": args.agent_label, "family": args.family, "regime": args.regime, "seed": args.seed}

    log_path = Path(args.episodes_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        for i in range(args.start_episode, args.episodes + 1):
            item = item_schedule[i - 1]
            p_max = category_p_max[item.category]
            ep_rng = episode_rng(args.seed, i)
            role_A, opener = role_opener_for(i)
            record = run_one_episode(
                ep_rng, agent_policy, counterpart_policy, stance_weights, item, p_max,
                regimes=(args.regime,), role_A=role_A, opener=opener,
            )
            log_file.write(json.dumps(episode_to_dict(record, episode_idx=i, run_meta=run_meta), ensure_ascii=False) + "\n")
            log_file.flush()
            print(f"[{args.mapping}] {args.agent_label}: episode {i}/{args.episodes} done")

    print(f"\n({args.agent_label}의 {args.start_episode}~{args.episodes} episode를 {log_path}에 이어씀)")


if __name__ == "__main__":
    main()
