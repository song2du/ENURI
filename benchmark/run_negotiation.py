"""터미널에서 편하게 협상을 돌려보기 위한 실행 엔트리.

실제 실험/평가 코드가 아니라 개발 중 눈으로 확인해보기 위한 스크립트 -- seed/model/
counterpart family/판수를 인자로 바꿔가며 GPT agent vs 규칙기반 counterpart 협상을 돌린다.
run.sh를 통해 실행하면 가상환경 python까지 자동으로 잡아준다.

--episodes 1 (기본값): 협상 1건의 전체 transcript를 자세히 출력.
--episodes N (N>1): 판마다 한 줄 요약만 찍고, 마지막에 metrics.py의 compute_metrics()로
    배치 전체 집계를 출력.

voice(counterpart 메시지 렌더링, voice.py)는 기본적으로 항상 켜져 있다. 처음엔 "voice는
citation_precision류 metric에 영향 안 준다"는 이유로 배치(--episodes N>1)일 때 기본 off로
가려 했는데, llm_agent.py의 format_history가 agent 프롬프트에 counterpart의 message를
그대로 넣기 때문에(에 counterpart 턴이 `OFFER $X -- "..."`로 노출됨) voice를 끄면 agent가
보는 정보 자체가 줄어들어 agent의 실제 판단(가격/결정)이 달라질 수 있고, 그러면 그 위에서
계산되는 SE+/AGR+ 같은 가격 경로 의존 metric도 voice on/off에 따라 값이 바뀔 수 있다 --
즉 voice off는 "메트릭에 영향 없는 최적화"가 아니라 "측정 조건 자체를 바꾸는 변경"이었다
(2026-07-11 수정). 그래서 --no-voice는 "진짜 metric을 볼 때"가 아니라 코드가 안 죽는지만
빠르게 확인하는 smoke test 용도로만 쓸 것 -- 실제로 보고할 metric은 항상 voice on 상태로
뽑을 것.
"""

from __future__ import annotations

import argparse
import random

from env import sample_episode, run_episode
from kernel import FAMILIES, make_counterpart_policy
from llm_agent import make_llm_agent_policy
from metrics import EpisodeRecord, compute_metrics
from voice import add_voice


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT agent vs 규칙기반 counterpart 협상 실행 (+ 배치 metric)")
    parser.add_argument("--seed", type=int, default=1, help="랜덤 시드 (episode 배치 재현용)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="agent에 쓸 OpenAI 모델")
    parser.add_argument(
        "--family", type=str, default="Candid", choices=list(FAMILIES.keys()), help="counterpart family"
    )
    parser.add_argument("--episodes", type=int, default=1, help="몇 판 돌릴지 (1이면 transcript, 2 이상이면 metric 집계)")
    parser.add_argument(
        "--no-voice",
        dest="voice",
        action="store_false",
        default=True,
        help="counterpart 메시지 렌더링 끄기 -- smoke test 전용, 진짜 metric 뽑을 땐 쓰지 말 것 (모듈 docstring 참고)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    agent_policy = make_llm_agent_policy(model=args.model)
    counterpart_policy = make_counterpart_policy(args.family)
    if args.voice:
        counterpart_policy = add_voice(counterpart_policy)

    print(f"model={args.model} family={args.family} episodes={args.episodes} voice={args.voice}")
    print()

    records: list[EpisodeRecord] = []
    for i in range(1, args.episodes + 1):
        episode = sample_episode(rng)
        result = run_episode(episode, agent_policy, counterpart_policy, rng)
        records.append(EpisodeRecord(episode=episode, result=result))

        if args.episodes == 1:
            print(f"role_A: {episode.role_A} | r_A: {episode.r_A:.2f} | K: {episode.K}")
            print(f"item: {episode.item.title} - {episode.item.description}")
            print(f"defects (ground truth, 참고용 -- agent는 못 봄): {[d.id for d in episode.item.ground_truth_defects]}")
            print()
            for k, side, action in result.history:
                price_str = f"${action.price:.2f}" if action.price is not None else ""
                print(f"{k:2d} [{side:11s}] {action.decision.name:7s} {price_str}  msg={action.message!r}")
            print()
            print("outcome:", result.outcome)
            if result.violations:
                print("violations:", result.violations)
        else:
            outcome_str = f"${result.outcome:.2f}" if isinstance(result.outcome, float) else "DISAGREEMENT"
            n_cited = sum(1 for (_, side, a) in result.history if side == "agent" and a.cited_defect_ids)
            print(f"[{i:3d}/{args.episodes}] role_A={episode.role_A.name:6s} outcome={outcome_str:>10s}  citation turns={n_cited}")

    if args.episodes > 1:
        print()
        print(f"=== metrics (batch of {args.episodes} episodes) ===")
        for name, value in compute_metrics(records).items():
            value_str = f"{value:.3f}" if value is not None else "None"
            print(f"  {name}: {value_str}")


if __name__ == "__main__":
    main()
