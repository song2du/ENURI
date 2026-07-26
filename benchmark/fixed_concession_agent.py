"""Fixed-concession baseline agent -- TERMS-Bench Table 2의 FC-1%/10%/30% 바닥 앵커
(terms-bench.txt:456 "conceding 1%, 10%, and 30% of the remaining distance to
reservation per Offer"). benchmark/CLAUDE.md "다음 작업"의 fixed-concession
baseline 항목, 2026-07-26 설계 논의 결과.

LLM/VLM을 전혀 안 쓰는 순수 규칙 기반 agent다 -- "정보에 접근 가능한 것만으론
부족하고 실제로 추론해서 반영해야 한다"를 보여주는 비교군(신규 비교군 ②,
decisions_log.md 2026-07-23 참고)의 바닥 지점 역할. `citing_dummy_agent.py`와
다른 점: 그쪽은 evidence/quadrant 기계 배관을 검증하려고 ground_truth_defects를
직접 읽는 "치팅" dummy였지만, 이 agent는 애초에 evidence를 전혀 처리하지 않는다
(cited_defect_ids는 항상 None) -- "정보를 아예 안 본다"는 진짜 바닥선이다.

양보 공식은 새로 만들지 않고 kernel.py의 counter_offer를 그대로 재사용한다 --
counterpart의 6-family가 concession_rate()(urgency/stance 기반 동적 계산)를
counter_offer의 lam 인자에 넣는 것과 똑같은 자리에, 여기서는 고정 상수
(0.01/0.10/0.30)를 넣을 뿐이다. opening offer도 마찬가지로 opening_offer를
재사용한다.

!! Simplification (accept 규칙) !! 논문 Table 2/H.1.1~H.1.7 어디에도 FC baseline
전용 accept 알고리즘 박스는 없다 (2026-07-26, terms-bench.txt 전체 재검색으로
확인). 대신 H.1.5(Agent Interface, terms-bench.txt:3421-3424)가 LLM의 JSON 파싱
실패 시 쓰는 deterministic fallback을 명시한다: "accept if the standing
counterpart offer is weakly preferred to walking away". 이게 논문이 보여주는
유일한 결정론적 accept 규칙이라, FC baseline에도 동일하게 적용한다 -- 즉
`favorability(counterpart offer) >= 0`이면 즉시 수락. 논문이 FC baseline에
이 규칙을 쓴다고 명시한 건 아니므로 정확한 재현이 아니라 최선의 추정이다.

!! Simplification (opening offer의 urgency/stance) !! `opening_offer`(kernel.py)는
counterpart의 심리 상태(urgency/stance)로 첫 제안의 강도를 조절하는데, agent
쪽에는 그런 심리 상태 필드가 없다(Episode.t_B만 있고 t_A는 없음 -- 논문 자체가
평가 대상 agent에게는 숨겨진 심리 파라미터를 안 준다는 설계). FC baseline은
아예 심리 상태라는 개념이 없는 기계적 규칙이므로, urgency=0.0/stance="neutral"
(= opening_modulation이 보정 없이 phi=1을 내는 중립값)을 넣어 episode.harshness
하나만으로 첫 제안 강도가 정해지게 한다.

이 agent는 절대 REJECT하지 않는다 -- env.py의 run_episode가 REJECT를 즉시
DISAGREEMENT로 끝내는 종단 액션으로 취급하므로(env.py:361-362), "협상이 안 풀리면
포기한다"는 능동적 판단 자체가 이 baseline의 설계 범위 밖이다. K턴 안에 합의
못 하면 자연스럽게 round-limit DISAGREEMENT로 끝난다(env.py:368) -- 다른 IR
지키는 dummy agent들과 동일하게 AGR-/CritViol% 구조적으로 0을 유지한다.
"""

from __future__ import annotations

import random

from env import Action, Decision, Episode, Role
from kernel import counter_offer, favorability, opening_offer

# 논문 표기(FC-1%/10%/30%) <-> 실제 lam 값 매핑. run_negotiation.py 등에서 이름으로
# 고를 수 있게 상수로 노출.
FC_RATES = {"FC-1": 0.01, "FC-10": 0.10, "FC-30": 0.30}


def make_fixed_concession_policy(rate: float):
    """rate: 매 Offer마다 reservation까지 남은 거리의 몇 %를 좁힐지 (0.01/0.10/0.30).

    Returns a policy(episode, history, side, rng) -> Action closure, `llm_agent.py`의
    `make_llm_agent_policy`/`kernel.py`의 `make_counterpart_policy`와 동일한 팩토리 패턴
    -- run_negotiation.py의 agent_policy 자리에 그대로 꽂힌다.
    """

    def policy(episode: Episode, history: list[tuple[int, str, Action]], side: str, rng: random.Random) -> Action:
        role_A = episode.role_A
        r_A = episode.r_A
        R = episode.p_max - episode.p_min

        # alternating-offer 프로토콜이라, 지금 내 턴이면 history의 마지막 항목은
        # (있다면) 항상 상대의 가장 최근 액션이다 -- 내가 opener면 history가 비어있음.
        last_action = history[-1][2] if history else None

        if last_action is not None and last_action.decision == Decision.OFFER:
            if favorability(last_action.price, r_A, role_A, R) >= 0:
                # walkaway보다 안 나쁘면 즉시 수락 (H.1.5 fallback 관례, 위 모듈 docstring 참고)
                return Action(decision=Decision.ACCEPT)

        own_prices = [a.price for (_, s, a) in history if s == side and a.decision == Decision.OFFER]
        if not own_prices:
            price = opening_offer(
                r_A, role_A, episode.p_min, episode.p_max, episode.harshness,
                urgency=0.0, stance="neutral", R=R, rng=rng,
            )
        else:
            price = counter_offer(own_prices[-1], r_A, role_A, lam=rate, noise_std=0.0, rng=rng)

        return Action(decision=Decision.OFFER, price=price, cited_defect_ids=None)

    return policy
