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
import sys
from pathlib import Path

# Windows 콘솔 기본 코드페이지(cp949 등)는 LLM이 흔히 내는 em-dash/커브 따옴표 같은 유니코드
# 문자를 못 찍어서 UnicodeEncodeError로 죽는다 (2026-08-01, --episodes 1 상세 transcript
# 출력에서 실측 -- 배치 모드는 요약 한 줄만 찍어서 이 문제를 원래 잘 안 밟았음). stdout/stderr를
# UTF-8로 강제하고, 그래도 안 되는 글자는 버리지 않고 대체 문자로 바꿔서 최소한 안 죽게 한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from data_loader import load_items
from env import DISAGREEMENT, REGIMES, Item, Role, sample_episode, run_episode
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


def episode_to_dict(record: EpisodeRecord, *, episode_idx: int, run_meta: dict) -> dict:
    """EpisodeRecord(episode+result) -> JSON 직렬화 가능한 dict.

    "1단계"(2026-07-28 결정, decisions_log.md): 커널 내부 확률(accept_prob/evidence_term 등)은
    kernel.py의 policy() 안에서 계산되고 밖으로 안 나가서 여기 없다 -- 그 값까지 필요해지면
    kernel.py 반환 계약을 바꾸는 "2단계" 작업. 지금은 EpisodeRecord가 이미 갖고 있는 것만 옮긴다.

    이름에 밑줄 없앰(2026-08-02): 원래 main() 내부 전용이었는데 mapping_robustness.py도
    개별 episode 기록을 남기려고 이 함수를 그대로 재사용하게 되면서 모듈 간 공유 유틸리티가
    됐다."""
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


def run_one_episode(
    rng, agent_policy, counterpart_policy, stance_weights, item, p_max,
    *, regimes: tuple[str, ...] | None = None, role_A: Role | None = None, opener: str | None = None,
) -> EpisodeRecord:
    """episode 하나 샘플링 + 실행. main()의 배치 루프와 mapping_robustness.py/visual_ablation.py
    양쪽이 공유하는 최소 단위 -- 이 두 줄을 각 파일이 따로 복붙하면 env.py가 두 벌로 갈라졌던
    사고(2026-07-28, decisions_log.md의 SEVERITY_MAPPINGS/PRICE_IMPACT_MAPPINGS 축 불일치)가
    재발할 수 있어서 뽑아냄.

    regimes/role_A/opener(2026-07-30 추가)는 기본 None이면 sample_episode 자체 기본값을
    그대로 쓴다(무전달 -- sample_episode의 기본값이 나중에 바뀌어도 여기서 중복 안 되게).
    값을 주면 강제 배정(env.py의 sample_episode 참고) -- role_opener_for()로 role×opener를
    균등 배정하거나, regime을 하나로 고정할 때 씀."""
    kwargs = {}
    if regimes is not None:
        kwargs["regimes"] = regimes
    if role_A is not None:
        kwargs["role_A"] = role_A
    if opener is not None:
        kwargs["opener"] = opener
    episode = sample_episode(rng, stance_weights=stance_weights, item=item, p_max=p_max, **kwargs)
    result = run_episode(episode, agent_policy, counterpart_policy, rng)
    return EpisodeRecord(episode=episode, result=result)


# role×opener 4칸을 episode_idx 순서대로 순환 배정 -- 논문(4.1절)의 25episode/cell block
# design을 재현하기 위한 결정론적 스케줄. sample_episode의 기본 rng.choice(완전 무작위)로는
# n=100 배치에서 칸당 3~15개로 쏠리는 게 실측됨(2026-07-30, decisions_log.md) -- 순서 자체는
# 임의로 고정한 것(Buyer 먼저, AgentOpens 먼저)이라 특별한 의미는 없다.
_ROLE_OPENER_CELLS: tuple[tuple[Role, str], ...] = (
    (Role.BUYER, "AgentOpens"),
    (Role.BUYER, "CounterpartOpens"),
    (Role.SELLER, "AgentOpens"),
    (Role.SELLER, "CounterpartOpens"),
)


def role_opener_for(episode_idx: int) -> tuple[Role, str]:
    """episode_idx(1부터 시작) -> (role_A, opener) 균등 배정. N=100이면 4칸에 25개씩
    정확히 나뉜다 (100 % 4 == 0). N이 4의 배수가 아니면 마지막 칸들이 1개씩 더 받는다."""
    return _ROLE_OPENER_CELLS[(episode_idx - 1) % 4]


def category_item_schedule(items: list[Item], episodes: int, seed: int) -> list[Item]:
    """episodes 길이의 아이템 스케줄 생성 -- 카테고리 marginal을 role_opener_for처럼 정확히
    균등 배정한다 (2026-08-02 결정: 셔플+순환은 카테고리 비율을 "운에 맡기는" 것이라, 표본이
    적은 카테고리(car 38개, 데이터셋 최소)가 배치에서 적게/안 뽑힐 위험을 그대로 안고 있었음).

    role_opener_for는 (i-1)%4로 위상이 고정돼 있다 -- 카테고리 배정도 그대로 (i-1)%n_category로
    하면 두 축이 완전히 맞물려서(confound) 예를 들어 "역할=Buyer, opener=AgentOpens" 칸이 항상
    같은 카테고리랑만 짝지어지는 문제가 생긴다. 그래서 "칸당 개수"는 role_opener_for처럼 정확히
    맞추되, "어느 episode에 어느 카테고리가 오는지 순서"는 독립적으로 섞어서 role_opener_for
    위상과 무관하게 만든다.

    카테고리 내부 아이템 순서도 섞는다 -- 원본 데이터가 카테고리별로 정렬돼 있어(예: bike가
    파일 앞쪽) 안 섞으면 매번 그 카테고리의 앞쪽 아이템만 반복해서 뽑히게 된다.

    seed 하나로 결정론적: mapping_robustness.py처럼 매핑마다 별도 Item 객체 리스트로 이 함수를
    호출해도(카테고리 구성/순서가 같은 원본 파일에서 오므로) 매핑끼리 동일한 스케줄이 나온다
    (episode_rng와 같은 이유로, "매핑/agent가 달라도 같은 episode_idx는 같은 시나리오"라는
    Confound 통제 불변식을 유지하기 위함).
    """
    by_category: dict[str, list[Item]] = {}
    for it in items:
        by_category.setdefault(it.category, []).append(it)
    categories = sorted(by_category)  # dict 순회 순서에 기대지 않고 이름으로 고정 -- 재현성
    n = len(categories)

    rng = random.Random(seed)
    for cat in categories:
        rng.shuffle(by_category[cat])

    base, rem = divmod(episodes, n)
    schedule_categories: list[str] = []
    for idx, cat in enumerate(categories):
        count = base + (1 if idx < rem else 0)  # N이 n의 배수가 아니면 앞쪽 카테고리부터 1개씩 더 받음
        schedule_categories.extend([cat] * count)
    rng.shuffle(schedule_categories)  # role_opener_for의 (i-1)%4 위상과 안 겹치게 순서 무작위화

    cursors = {cat: 0 for cat in categories}
    schedule: list[Item] = []
    for cat in schedule_categories:
        pool = by_category[cat]
        schedule.append(pool[cursors[cat] % len(pool)])  # pool보다 episodes 배정이 많으면 wraparound
        cursors[cat] += 1
    return schedule


def episode_rng(base_seed: int, episode_idx: int) -> random.Random:
    """episode마다 독립적인 rng를 만든다 (2026-07-30, decisions_log.md 참고).

    배경: 이 함수가 생기기 전에는 main()의 배치 루프가 rng 하나를 계속 이어 쓰면서(episode마다
    새로 안 만듦) run_one_episode에 넘겼다. 그 rng는 sample_episode의 역할/유보가격/상대방
    성향 추첨뿐 아니라 run_episode turn-loop 안에서 counterpart의 확률적 결정에도 계속
    소비된다. 문제는 agent마다 협상이 끝나는 턴 수가 달라서(FC-1은 느리게 양보, FC-10은
    빠르게 양보 등) 같은 seed로 시작해도 rng가 소비되는 "속도"가 agent마다 달라진다는 것 --
    그러면 특정 episode에서 두 agent의 협상 턴 수가 처음으로 갈라지는 순간부터, 그 다음
    episode의 sample_episode()가 뽑는 값(아이템은 고정이어도 역할/유보가격/K 등)이 agent마다
    달라져 버린다. 실측 사례: FC-1 vs FC-10을 같은 --seed --data --episodes 100으로 따로
    돌렸더니 role_A 시퀀스가 17번째 episode까지는 완전히 같다가 18번째부터 갈라짐 -- 그
    뒤로는 "같은 100개 시나리오를 비교"하는 게 아니라 서로 다른 시나리오를 비교하고 있었다.

    수정: episode_idx마다 base_seed*1_000_000+episode_idx로 완전히 새 rng를 만들어 쓴다.
    그러면 어떤 agent를 넣든, 몇 턴 만에 협상이 끝나든, "다음 episode"의 시작 rng 상태에
    영향을 주지 않는다 -- 같은 episode_idx는 항상 같은 시나리오(아이템+역할+유보가격+
    상대방 성향+K+harshness)로 고정된다. (turn-loop 안에서 agent가 실제로 어떤 가격을
    부르는지에 따라 counterpart의 반응이 달라지는 건 여전히 살아있다 -- 그건 우리가 보고
    싶은 진짜 차이지, 없애야 할 오염이 아니다.)

    1_000_000을 곱하는 이유는 암호학적 해시가 아니라 그냥 base_seed별 구간이 안 겹치게
    떼어놓는 용도 -- episode 수가 이 프로토타입 규모(수백 건)를 한참 넘지 않는 한 안전하다.
    """
    return random.Random(base_seed * 1_000_000 + episode_idx)


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
    parser.add_argument(
        "--regime", type=str, default="overlap", choices=list(REGIMES),
        help="episode economic regime을 하나로 고정 (기본 overlap -- family=Candid와 같은 이유로 "
             "evidence 축과 무관한 축을 고정, 2026-07-30 결정). overlap/urgency_shift는 항상 feasible, "
             "no_deal은 항상 infeasible이라 이걸 고정하면 FAGR-는 영구적으로 정의 안 됨(None) -- "
             "decisions_log.md 참고",
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

    items = None
    category_p_max: dict[str, float] = {}
    item_schedule: list[Item] = []
    if args.data is not None:
        items = load_items(args.data)
        # p_max는 아이템 자기 listing_price보다 넓게(카테고리 내 최댓값) 잡는다 -- env.py
        # sample_episode의 p_max 기본값(item.listing_price)은 단독 호출용 안전장치일 뿐,
        # 여기서는 전체 데이터셋을 알고 있으니 카테고리 경계를 직접 계산해서 넘겨준다.
        for it in items:
            category_p_max[it.category] = max(category_p_max.get(it.category, 0.0), it.listing_price)
        # 카테고리 marginal을 정확히 균등 배정 (2026-08-02, category_item_schedule docstring 참고) --
        # 예전엔 셔플+순환(items[(i-1)%len(items)])이라 카테고리 비율이 셔플 운에 맡겨져 있었음.
        item_schedule = category_item_schedule(items, args.episodes, args.seed)
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
        f"agent={agent_label} provider={args.provider} family={args.family} regime={args.regime} "
        f"episodes={args.episodes} voice={args.voice} image={args.use_image} data={data_label}"
    )
    print()

    run_meta = {
        "seed": args.seed,
        "agent": agent_label,
        "provider": args.provider,
        "family": args.family,
        "regime": args.regime,
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
        item = item_schedule[i - 1] if items else None
        p_max = category_p_max[item.category] if item else None
        # episode마다 독립적인 rng (episode_rng 참고) -- agent를 바꿔도 같은 i는 항상 같은
        # 시나리오가 되도록 보장, agent 간 배치 비교의 전제조건.
        ep_rng = episode_rng(args.seed, i)
        # regime 고정 + role×opener 균등 배정 (2026-07-30, decisions_log.md) -- 둘 다 i에서만
        # 결정되므로 agent가 바뀌어도 i가 같으면 항상 같은 배정.
        role_A, opener = role_opener_for(i)
        record = run_one_episode(
            ep_rng, agent_policy, counterpart_policy, stance_weights, item, p_max,
            regimes=(args.regime,), role_A=role_A, opener=opener,
        )
        episode, result = record.episode, record.result
        records.append(record)

        if log_file:
            log_file.write(json.dumps(episode_to_dict(record, episode_idx=i, run_meta=run_meta), ensure_ascii=False) + "\n")
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
