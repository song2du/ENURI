"""결함 quadrant/evidence 관련 기능(citation 3종 metric, quadrant별 evidence bonus,
severity_calibration, subtle_misaligned_gap)이 실제로 작동하는지 이미지/VLM 없이 눈으로
확인하기 위한 데모+테스트 스크립트.

!! 이 파일의 agent는 절대 real agent가 아니고, 실제 평가에 쓰면 안 됨 !!
`episode.item.ground_truth_defects`를 직접 읽어서 결함을 "치팅"으로 인용한다 -- 실제
평가 대상 agent는 이 필드를 절대 못 읽어야 한다는 정보 비공개 규약(env.py의 Item
docstring)을 의도적으로 깨는 테스트 전용 더미다. 지금 citation 관련 코드가 실전(진짜
LLM agent, 이미지 없음)에서 거의 발동을 안 하는 이유는 (1) 진짜 이미지가 없고 (2) 설령
있어도 agent가 결함의 "정확한 id 문자열"을 알 방법이 없어서(idea.md 2026-07-15의
닫힌 태그 어휘 미해결 문제)인데, 이 스크립트는 그 두 선행조건을 치팅으로 우회해서
"배관 자체는 맞게 짜였는가"만 이미지 없이 먼저 확인하는 용도다.

voice(counterpart 메시지 렌더링)는 이 데모에서 끈다 -- 목적이 evidence/quadrant 기계
자체를 보는 거라, 굳이 OpenAI 키/네트워크에 의존하게 만들 필요가 없어서 (run_negotiation.py
의 --no-voice와 달리, 여기는 메트릭에 voice가 원래 영향 없는 순수 규칙기반 배치라 꺼도 됨).
"""

from __future__ import annotations

import argparse
import random

from env import Action, Decision, Role, sample_episode, run_episode
from kernel import FAMILIES, evidence_term_for, favorability, make_counterpart_policy
from metrics import EpisodeRecord, compute_metrics


def _extract_prices(history, side):
    return [a.price for (_, s, a) in history if s == side and a.decision == Decision.OFFER]


def _agent_opening_price(r_A, role_A, p_min, p_max, anchor_frac):
    extreme = p_min if role_A == Role.BUYER else p_max
    return r_A + anchor_frac * (extreme - r_A)


def _clip_offer(price, r_A, role_A, p_min, p_max):
    if role_A == Role.BUYER:
        return min(max(price, p_min), r_A)
    return max(min(price, p_max), r_A)


def make_citing_dummy_agent(
    anchor_frac: float = 0.6,
    base_concession_frac: float = 0.08,
    cite_prob: float = 0.8,
    hallucinate_prob: float = 0.15,
    severity_scale: float = 0.01,
):
    """치팅 dummy agent 팩토리 (make_counterpart_policy와 같은 클로저 패턴).

    매 OFFER 턴마다 cite_prob 확률로 뭔가를 인용한다:
    - hallucinate_prob 확률로 존재하지 않는 가짜 id를 인용 (hallucination_rate>0을 보려는 용도)
    - 그 외엔 실제 결함 하나를 골라 인용하고, **그 결함의 price_impact가 클수록 이번 턴
      가격을 더 크게 움직인다** (severity_scale로 스케일) -- severity_calibration이 양의
      상관을 보여주도록 의도적으로 "심각도에 비례해서 잘 반영하는" agent로 설계함.
    """

    def policy(episode, history, side, rng: random.Random) -> Action:
        role_A, r_A = episode.role_A, episode.r_A
        p_min, p_max = episode.p_min, episode.p_max
        R = p_max - p_min
        own_prices = _extract_prices(history, "agent")
        opp_prices = _extract_prices(history, "counterpart")
        real_defects = episode.item.ground_truth_defects  # !! CHEAT: 테스트 전용, 실제 agent는 절대 못 읽음

        if not opp_prices:
            price = _agent_opening_price(r_A, role_A, p_min, p_max, anchor_frac)
            price = _clip_offer(price, r_A, role_A, p_min, p_max)
            return Action(decision=Decision.OFFER, price=price, message="opening offer")

        fav = favorability(opp_prices[-1], r_A, role_A, R)
        if fav >= 0:
            return Action(decision=Decision.ACCEPT)

        prev_own = own_prices[-1] if own_prices else _agent_opening_price(r_A, role_A, p_min, p_max, anchor_frac)

        # 이번 턴 인용 결정
        cited_ids = None
        extra_concession_frac = 0.0
        if real_defects and rng.random() < cite_prob:
            if rng.random() < hallucinate_prob:
                cited_ids = ("fake_defect_xyz",)  # 지어낸 결함 -- hallucination_rate용
            else:
                d = real_defects[rng.randrange(len(real_defects))]
                cited_ids = (d.id,)
                extra_concession_frac = severity_scale * d.price_impact  # 심각도에 비례해서 더 크게 움직임

        concession_frac = base_concession_frac + extra_concession_frac
        target = prev_own - concession_frac * (prev_own - r_A)
        price = _clip_offer(target, r_A, role_A, p_min, p_max)
        return Action(decision=Decision.OFFER, price=price, message="citing offer", cited_defect_ids=cited_ids)

    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence/quadrant 기계가 실제로 작동하는지 보여주는 데모")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--family", type=str, default="Candid", choices=list(FAMILIES.keys()))
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    agent = make_citing_dummy_agent()
    counterpart = make_counterpart_policy(args.family)  # voice 없이 순수 kernel만

    # --- ① 단일 episode: 매 턴 evidence_term_for 값을 직접 찍어서 quadrant bonus를 눈으로 확인 ---
    # 인용이 한 번도 안 나오는 episode(운 나쁘게 cite_prob 롤이 다 빗나가거나, agent가 offer할
    # 기회도 없이 바로 accept해버리는 경우)를 뽑을 수도 있어서, 인용이 최소 1번 나오는 episode가
    # 나올 때까지 최대 30번 재시도한다 -- 데모 목적이라 재현성보다 "실제로 보여주기"가 우선.
    print("=" * 70)
    print("① 단일 episode transcript -- 매 턴 evidence_term(quadrant bonus) 직접 계산해서 표시")
    print("=" * 70)
    for attempt in range(30):
        episode = sample_episode(rng)
        history = []
        turn = "agent" if episode.opener == "AgentOpens" else "counterpart"
        transcript = []
        saw_citation = False
        for k in range(1, episode.K + 1):
            policy = agent if turn == "agent" else counterpart
            action = policy(episode, history, side=turn, rng=rng)
            evidence_note = ""
            if turn == "agent" and action.cited_defect_ids:
                term = evidence_term_for(action, episode.item, episode.role_A)
                evidence_note = f"  <- cited={action.cited_defect_ids}, evidence_term_for={term:+.2f}"
                saw_citation = True
            price_str = f"${action.price:.2f}" if action.price is not None else ""
            transcript.append(f"{k:2d} [{turn:11s}] {action.decision.name:7s} {price_str}{evidence_note}")
            if action.decision in (Decision.ACCEPT, Decision.REJECT):
                break
            history.append((k, turn, action))
            turn = "counterpart" if turn == "agent" else "agent"
        if saw_citation:
            break
    print(f"role_A={episode.role_A.name} item defects (ground truth): "
          f"{[(d.id, d.quadrant, d.price_impact) for d in episode.item.ground_truth_defects]}")
    print()
    print("\n".join(transcript))
    print()

    # --- ② 배치: compute_metrics()로 citation/quadrant 관련 지표가 실제로 non-trivial 값을 내는지 ---
    print("=" * 70)
    print(f"② 배치 {args.episodes} episodes -- citation/quadrant 관련 metric 전부")
    print("=" * 70)
    records = []
    for _ in range(args.episodes):
        ep = sample_episode(rng)
        result = run_episode(ep, agent, counterpart, rng)
        records.append(EpisodeRecord(episode=ep, result=result))

    for name, value in compute_metrics(records).items():
        value_str = f"{value:.3f}" if value is not None else "None"
        print(f"  {name}: {value_str}")


if __name__ == "__main__":
    main()
