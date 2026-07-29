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
import json
import random
from pathlib import Path

from data_loader import load_items
from env import DISAGREEMENT, sample_episode, run_episode
from fixed_concession_agent import FC_RATES, make_fixed_concession_policy
from kernel import FAMILIES, make_counterpart_policy, stance_prior_for
from llm_agent import make_llm_agent_policy
from metrics import EpisodeRecord, compute_metrics
from voice import add_voice

# episode 기록(JSONL) 기본 저장 위치 -- benchmark/result/episodes.jsonl. 여기 상수만 바꾸면
# 저장 위치가 코드 전체에서 바뀐다 (2026-07-28, 매번 --log-path를 안 줘도 되게).
# __file__ 기준 절대경로라 run_negotiation.py를 어느 작업 디렉터리에서 실행해도 항상
# benchmark/result/를 가리킨다.
DEFAULT_LOG_PATH = Path(__file__).parent / "result" / "episodes.jsonl"


def _episode_to_dict(record: EpisodeRecord, *, episode_idx: int, run_meta: dict) -> dict:
    """EpisodeRecord(episode+result) -> JSON 직렬화 가능한 dict.

    "1단계"(2026-07-28 결정, decisions_log.md): 커널 내부 확률(accept_prob/evidence_term 등)은
    kernel.py의 policy() 안에서 계산되고 밖으로 안 나가서 여기 없다 -- 그 값까지 필요해지면
    kernel.py 반환 계약을 바꾸는 "2단계" 작업. 지금은 EpisodeRecord가 이미 갖고 있는 것만 옮긴다.
    """
    ep, res = record.episode, record.result
    item = ep.item
    return {
        **run_meta,
        "episode_idx": episode_idx,
        "item": {
            "category": item.category,
            "title": item.title,
            "listing_price": item.listing_price,
            "image_ref": item.image_ref,
            "defects": [
                {
                    "id": d.id,
                    "description": d.description,
                    "defect_type": d.defect_type,
                    "quadrant": d.quadrant,
                    "severity": d.severity,
                    "price_impact": d.price_impact,
                }
                for d in item.ground_truth_defects
            ],
        },
        "episode_config": {
            "regime": ep.regime,
            "p_min": ep.p_min,
            "p_max": ep.p_max,
            "role_A": ep.role_A.name,
            "r_A": ep.r_A,
            "t_B": {"r": ep.t_B.r, "urgency": ep.t_B.urgency, "stance": ep.t_B.stance},  # 협상 중엔 agent에게 비공개, 사후 분석용으로만 기록
            "opener": ep.opener,
            "K": ep.K,
            "harshness": ep.harshness,
        },
        "transcript": [
            {
                "turn": k,
                "side": side,
                "decision": action.decision.name,
                "price": action.price,
                "message": action.message,
                "sentiment": action.sentiment,
                "posture": action.posture,
                "cited_defect_ids": list(action.cited_defect_ids or []),
            }
            for k, side, action in res.history
        ],
        "violations": [{"turn": k, "side": side, "type": vtype} for k, side, vtype in res.violations],
        "outcome": None if res.outcome is DISAGREEMENT else res.outcome,
    }


def run_one_episode(rng, agent_policy, counterpart_policy, stance_weights, item, p_max) -> EpisodeRecord:
    """episode 하나 샘플링 + 실행. main()의 배치 루프와 mapping_robustness.py 양쪽이 공유하는
    최소 단위 -- 이 두 줄을 각 파일이 따로 복붙하면 env.py가 두 벌로 갈라졌던 사고(2026-07-28,
    decisions_log.md의 SEVERITY_MAPPINGS/PRICE_IMPACT_MAPPINGS 축 불일치)가 재발할 수 있어서 뽑아냄."""
    episode = sample_episode(rng, stance_weights=stance_weights, item=item, p_max=p_max)
    result = run_episode(episode, agent_policy, counterpart_policy, rng)
    return EpisodeRecord(episode=episode, result=result)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT agent vs 규칙기반 counterpart 협상 실행 (+ 배치 metric)")
    parser.add_argument("--seed", type=int, default=1, help="랜덤 시드 (episode 배치 재현용)")
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="agent에 쓸 모델 -- provider=openai면 'gpt-4o'처럼 직결 이름, provider=openrouter면 "
             "'anthropic/claude-opus-4.6'처럼 'provider/model' 네이밍 (llm_agent.py 참고)",
    )
    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "openrouter"],
        help="agent LLM 호출 경로 (llm_agent.py의 make_llm_agent_policy). openrouter는 "
             "OPENROUTER_API_KEY 필요 -- 발급 전까지는 openai(기본값)만 동작",
    )
    parser.add_argument(
        "--fc-rate", type=str, default=None, choices=list(FC_RATES.keys()),
        help="agent를 LLM 대신 fixed-concession baseline으로 대체 (FC-1/FC-10/FC-30) -- "
             "API 키 없이 배관 테스트할 때도 유용 (--model은 이때 무시됨)",
    )
    parser.add_argument(
        "--family", type=str, default="Candid", choices=list(FAMILIES.keys()), help="counterpart family"
    )
    parser.add_argument("--episodes", type=int, default=1, help="몇 판 돌릴지 (1이면 transcript, 2 이상이면 metric 집계)")
    parser.add_argument(
        "--data", type=str, default=None,
        help="실 데이터셋 results.jsonl 경로 -- 지정 시 mock sample_item 대신 이 파일의 아이템을 순환",
    )
    parser.add_argument(
        "--log-path", type=str, default=str(DEFAULT_LOG_PATH),
        help=f"episode별 기록(transcript/violations/outcome 등)을 JSONL로 이어쓸 경로 (기본값: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--no-log",
        dest="log",
        action="store_false",
        default=True,
        help="episode 기록 저장 끄기 -- 기본은 항상 저장(--log-path 참고), 정말 안 남기고 싶을 때만",
    )
    parser.add_argument(
        "--no-voice",
        dest="voice",
        action="store_false",
        default=True,
        help="counterpart 메시지 렌더링 끄기 -- smoke test 전용, 진짜 metric 뽑을 땐 쓰지 말 것 (모듈 docstring 참고)",
    )
    parser.add_argument(
        "--no-image",
        dest="use_image",
        action="store_false",
        default=True,
        help="실 이미지가 있어도 agent에게 안 보냄 -- pulse.pptx 슬라이드6 visual 유/무 baseline용 "
             "(llm_agent.py의 make_llm_agent_policy use_image 참고). --fc-rate는 원래 이미지를 안 보므로 무관",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = None
    category_p_max: dict[str, float] = {}
    if args.data is not None:
        items = load_items(args.data)
        rng.shuffle(items)  # 파일이 카테고리별로 정렬돼 있어(bike가 먼저 등) 순서대로 뽑으면 편향됨
        # p_max는 아이템 자기 listing_price보다 넓게(카테고리 내 최댓값) 잡는다 -- env.py
        # sample_episode의 p_max 기본값(item.listing_price)은 단독 호출용 안전장치일 뿐,
        # 여기서는 전체 데이터셋을 알고 있으니 카테고리 경계를 직접 계산해서 넘겨준다.
        for it in items:
            category_p_max[it.category] = max(category_p_max.get(it.category, 0.0), it.listing_price)
    if args.fc_rate is not None:
        agent_policy = make_fixed_concession_policy(FC_RATES[args.fc_rate])
    else:
        agent_policy = make_llm_agent_policy(model=args.model, provider=args.provider, use_image=args.use_image)
    counterpart_policy = make_counterpart_policy(args.family)
    if args.voice:
        counterpart_policy = add_voice(counterpart_policy)
    # 2026-07-25 버그 수정: family의 stance_prior가 지금까지 sample_episode에 전달되지
    # 않아서 (예: Adversarial의 aggressive-skewed 확률이) 무시되고 있었음 -- env.py의
    # sample_episode docstring 참고. args.family 하나로 한 번만 계산해서 매 episode에 재사용.
    stance_weights = stance_prior_for(args.family)

    agent_label = args.fc_rate if args.fc_rate is not None else args.model
    data_label = f"{args.data} ({len(items)} items)" if items else "mock"
    print(
        f"agent={agent_label} provider={args.provider} family={args.family} "
        f"episodes={args.episodes} voice={args.voice} image={args.use_image} data={data_label}"
    )
    print()

    run_meta = {
        "seed": args.seed,
        "agent": agent_label,
        "provider": args.provider,
        "family": args.family,
        "data": args.data,
        "voice": args.voice,
        "use_image": args.use_image,
    }
    log_file = None
    if args.log:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)  # benchmark/result/가 없으면 만듦
        log_file = open(log_path, "a", encoding="utf-8")  # append -- 여러 번 돌린 기록이 한 파일에 누적됨

    records: list[EpisodeRecord] = []
    for i in range(1, args.episodes + 1):
        item = items[(i - 1) % len(items)] if items else None
        p_max = category_p_max[item.category] if item else None
        record = run_one_episode(rng, agent_policy, counterpart_policy, stance_weights, item, p_max)
        episode, result = record.episode, record.result
        records.append(record)

        if log_file:
            log_file.write(json.dumps(_episode_to_dict(record, episode_idx=i, run_meta=run_meta), ensure_ascii=False) + "\n")
            log_file.flush()  # 중간에 죽어도(API 에러 등) 그때까지 판은 남도록

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

    if log_file:
        log_file.close()
        print(f"\n(episode 기록 {args.episodes}건을 {args.log_path}에 이어씀)")

    if args.episodes > 1:
        print()
        print(f"=== metrics (batch of {args.episodes} episodes) ===")
        for name, value in compute_metrics(records).items():
            value_str = f"{value:.3f}" if value is not None else "None"
            print(f"  {name}: {value_str}")


if __name__ == "__main__":
    main()
