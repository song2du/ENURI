"""매핑 로버스트니스 실험 오케스트레이션 (pulse.pptx 슬라이드3, benchmark/CLAUDE.md "다음 작업" 참고).

절차 (슬라이드3 원문 ①~④, Paper/pulse_extract.txt):
① 결함->가치 매핑을 여러 버전(env.py의 PRICE_IMPACT_MAPPINGS, 4개) 준비 -- 이미 돼 있음.
② 매핑마다 --data 데이터셋 전체를 각 agent로 돌려서 배치 metric을 낸다.
③ agent들 사이의 순위를 매핑끼리 Spearman's rho로 비교.
④ rho가 낮은 지점("빵꾸")을 진단 -- 전체 episode 합산 + quadrant별(4개) 두 단위로 계산.

**순위 지표 2026-07-29 결정**: SE+(원 논문 headline, price_impact가 fair_price/ZOPA를 옮기므로
매핑에 실제로 민감)를 주 지표로, utilization_proximity(이 벤치마크의 새 축 -- 증거를 양보로
연결했는가)를 보조 지표로 둘 다 계산/리포트한다.

**Confound 통제 (2026-07-26 decisions_log.md, 2026-07-30 강화)**: 매핑을 바꾸면 채점 기준뿐
아니라 fair_price 경유로 ZOPA 자체도 같이 움직인다 -- "전체 벤치마크를 돌린다"(슬라이드3 ②)가
그런 의도라고 판단했기 때문. 그래서 매핑끼리(그리고 agent끼리) 난이도가 아니라 "매핑/agent
선택"만 비교되도록, 매핑/agent에 상관없이 매 (agent, mapping) 셀의 매 episode_idx가 동일한
시나리오에서 시작한다 -- `episode_rng(seed, i)`로 episode마다 독립적으로 rng를 새로 만들기
때문에(run_negotiation.py의 episode_rng docstring 참고), 아이템뿐 아니라 역할/유보가격/
상대방 성향/K 등 episode 조건 전체가 매핑/agent 무관 고정되고, price_impact 숫자만 매핑
따라 달라진다. (2026-07-30 이전에는 rng를 셀 전체에서 이어 써서 아이템 정체성만 고정되고,
agent별로 협상 턴 수가 갈리면 그 뒤 episode 조건이 슬쩍 달라지는 버그가 있었다 -- FC-1/FC-10
배치 비교에서 실측으로 발견.)

**배관 검증 단계 종료 (2026-08-01)**: `OPENROUTER_API_KEY` 도착 확인, `default_agents()`를
FC-1/10/30 + gpt-5.5 + gpt-4o-mini + Claude Opus 4.6 + Qwen3.6-Plus + Kimi K3
(총 8개, `default_agents()` docstring 참고)로 교체 완료 (gpt-4o -> gpt-5.5는 2026-08-01
프론티어 세대 정렬 결정, 아래 참고). 이제부터 나오는 rho 숫자가 슬라이드3
"본 벤치마크의 순위가 매핑에 안 흔들린다" 주장의 실제 근거가 될 수 있다.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from data_loader import load_items
from env import PRICE_IMPACT_MAPPINGS, REGIMES, Item
from fixed_concession_agent import FC_RATES, make_fixed_concession_policy
from kernel import FAMILIES, make_counterpart_policy, stance_prior_for
from llm_agent import make_llm_agent_policy
from metrics import QUADRANTS, EpisodeRecord, compute_metrics, spearman_rho
from run_negotiation import category_item_schedule, episode_rng, episode_to_dict, role_opener_for, run_one_episode
from voice import add_voice

# 리포트 저장 기본 위치 -- episodes.jsonl(run_negotiation.py)과 같은 이유(2026-07-28)로
# 휘발 방지: 이 스크립트도 stdout만 찍으면 프로세스 종료와 함께 결과가 사라진다.
DEFAULT_OUT_PATH = Path(__file__).parent / "result" / "mapping_robustness.json"

# 개별 episode 기록(transcript 등) 저장 위치 (2026-08-02 추가, 사용자 지적: "수치만이
# 아니라 agent별/episode별 기록도 있어야 하지 않냐"). scores_by_mapping은 집계값뿐이라
# 나중에 특정 episode를 직접 까보거나 compute_metrics()에 없는 새 지표를 계산하려면
# 원본 transcript가 필요한데, 그동안 이 파일은 그걸 안 남기고 있었음 -- run_negotiation.py의
# episodes.jsonl과 같은 포맷(episode_to_dict)을 그대로 재사용해서 같은 방식으로 저장한다.
DEFAULT_EPISODES_LOG_PATH = Path(__file__).parent / "result" / "mapping_robustness_episodes.jsonl"

# 순위를 매길 metric -- compute_metrics()가 반환하는 dict의 키 (모듈 docstring의 "순위 지표
# 2026-07-29 결정" + 2026-08-02 확장 참고).
# - SE+: 원 논문 headline, 모든 agent(FC 포함)에서 항상 값이 나옴.
# - utilization_proximity: 인용→양보 "방향"이 맞았는가 (evidence 축, 크기는 안 봄).
# - severity_calibration_rho: 인용한 결함의 price_impact(달러, **매핑이 정의하는 값**)와
#   실제 가격조정 폭의 상관관계 -- 매핑을 바꾸면 price_impact 자체가 바뀌므로, 이 셋 중
#   매핑에 가장 직접적으로 민감한 evidence 지표 (2026-08-02 추가, 사용자 지적: "visual
#   관련 지표도 순위비교에 들어가야 하지 않나").
RANKING_METRICS = ("SE+", "utilization_proximity", "severity_calibration_rho")

# citation 기반 지표 (2026-08-02 추가): utilization_proximity/severity_calibration_rho는
# agent가 실제로 결함을 인용해야만 값이 정의된다. FC-1/10/30은 설계상 cited_defect_ids가
# 항상 None이라(fixed_concession_agent.py) 이 지표들이 구조적으로 항상 None -- _rank_vector가
# "agent 중 하나라도 None이면 그 매핑 전체 None" 규칙이라서, FC를 포함한 채로 이 지표들의
# rho를 구하면 8개 agent 전부에서 영원히 None만 나온다(실측: mappings=1개짜리 스모크테스트만
# 해서 여태 못 봤던 문제 -- pairwise 비교 자체가 매핑 2개 이상이어야 발생). 그래서 이 지표들만
# rho_report에서 agent 목록을 cites_evidence=True로 필터링해서 계산한다 -- SE+는 여전히 FC
# 포함 전체로 계산.
CITATION_METRICS = {"utilization_proximity", "severity_calibration_rho"}


@dataclass
class AgentConfig:
    label: str
    make_policy: Callable[[], Callable]  # 인자 없이 바로 agent policy를 만드는 클로저 (지연 생성)
    cites_evidence: bool = True  # False면 CITATION_METRICS 순위비교에서 자동 제외 (FC 계열)


def default_agents() -> list[AgentConfig]:
    """실 로스터 (2026-08-01, OPENROUTER_API_KEY 도착 후 교체 -- 이전엔 FC+gpt-4o 배관검증용
    placeholder였음, 모듈 docstring 참고). 구성:
    - FC-1/10/30: 논문(Table 2)도 LLM들을 항상 이 3개 baseline과 나란히 비교하므로 유지.
    - gpt-5.5: 메인 비교 대상 프론티어 슬롯. 원래 gpt-4o였다가 2026-08-01 교체 -- Claude
      Opus 4.6/Qwen3.6-Plus/Kimi K3가 전부 최신 프론티어인데 gpt-4o만 세대가 뒤처져서
      "프론티어끼리 비교"라는 로스터 취지에 안 맞았음(사용자 지적). 논문(Table 9)도 이
      슬롯에 GPT-5.4/GPT-5.5를 쓰므로 그대로 맞춤. 스모크테스트로 비전+무결함 아이템에서
      할루시네이션 안 하는 것까지 확인.
    - gpt-4o-mini: CLAUDE.md "평가 대상 모델"이 요구하는 약한 모델 바닥 앵커 -- gpt-4o와
      달리 얘는 원래 "구세대/약한 모델"이 의도된 설계라 그대로 둠.
    - claude-opus-4.6/qwen3.6-plus/kimi-k3: 사용자 지정(2026-08-01, 미국산 2 + 중국산 2
      균형 -- gpt-5.5/claude vs qwen/kimi), OpenRouter 슬러그도 사용자가 openrouter.ai/models
      에서 직접 확인해서 줌. deepseek/deepseek-v4-pro는 스모크테스트에서 "No endpoints found
      that support image input" 404로 탈락(비전 미지원 -- 이 벤치마크는 이미지 필수라
      애초에 평가 대상이 될 수 없음) -- moonshotai/kimi-k3로 교체, 스모크테스트에서 이미지
      기반 결함 인용까지 확인됨.
    - gemini-3.1-pro-preview/gemma-4-31b-it (2026-08-03 추가, **2026-08-05 provider
      재변경**): 논문(Table 9)에도 있는 Gemini 3.1 Pro Preview/Gemma 4 31B IT. 처음엔
      OpenRouter 마진을 피하려고 `provider="google"`로 직결했는데(사용자 Google 계정
      크레딧, llm_agent.py 모듈 docstring 참고), 정식 실행 중 `RateLimitError` 429로
      확인됨 -- Google Gemini API가 **모델당 하루 250 요청**(`generate_requests_per_
      model_per_day`)으로 막혀있어(크레딧/결제와 무관한 별도 한도), 100-episode 셀 하나가
      필요로 하는 호출 수(협상 턴수 감안 200~800회)에도 못 미침. 논문도 실제로는 Gemini를
      OpenRouter로 불렀다는 걸 재확인(§H.1.1 "LLMs are called via OpenRouter") -- 결국
      `provider="openrouter"`, 모델 문자열 `"google/gemini-3.1-pro-preview"`/
      `"google/gemma-4-31b-it"`로 되돌림(decisions_log.md 2026-08-05 참고). 이제 이 두
      agent도 다른 OpenRouter agent와 같은 연구실 예산에서 나감 -- "개인 Google 크레딧으로
      비용 절감" 전략은 이 배치 규모에서는 폐기.
    - qwen3-vl-32b-instruct/thinkingmachines-inkling (2026-08-03 추가, 사용자 지정 --
      Qwen3.6-Plus와 다른 사이즈/계열의 오픈 웨이트 추가 비교, inkling은 "요즘 SOTA급
      오픈 웨이트로 화제"라는 사용자 판단). 둘 다 openrouter로 스모크테스트 통과. **주의
      (methods/limitations에 남길 것)**: thinkingmachines/inkling은 스모크 1-episode에서
      ground_truth_defects가 빈 아이템(결함 없음)에 대해 "rust on the chain", "scratches
      on the frame", "staining/wear on the seat" 등 실재하지 않는 결함을 사진에서 봤다고
      주장하며 가격을 깎았다 -- hallucination_rate가 실제로 잡아야 할 실패 패턴의 실측
      사례. 표본 1개라 로스터에서 빼진 않았지만(사용자 결정), 정식 실행 결과에서 이
      agent의 hallucination_rate를 우선 확인할 것.
    - claude-sonnet-4.6/gpt-4o (2026-08-03 추가, 사용자 지정): 팀원 제안(벤더당 여러 티어
      비교)을 받아들여 Claude/GPT 쪽에 티어 하나씩 추가. gpt-4o는 2026-08-01에 "세대
      뒤처짐"으로 gpt-5.5에 자리를 내주고 빠졌던 모델인데, 이번엔 그 자리를 대체하는 게
      아니라 gpt-5.5(프론티어 슬롯)와 별도로 "구세대 GPT" 비교 포인트로 재투입된 것 --
      gpt-4o-mini(약한 모델 바닥 앵커)와는 다른 목적. 스모크테스트 둘 다 실 결함(scratch)을
      정확히 인용하며 협상 완료.
    - gemini-3.6-flash (2026-08-03 추가, 사용자 지정): 팀원 제안 중 보류했던 Gemini 3-티어
      확장을 사용자가 최종적으로 채택 -- Gemini 쪽도 Claude/GPT처럼 Pro(gemini-3.1-pro-preview)
      + 경량 티어(flash) 비교 구도로 맞춤. `provider="google"`로 직결(개인 크레딧). 스모크
      테스트에서 실 결함(rust)을 정확히 인용하며 협상 완료.
    - grok-4.5/mimo-v2.5/nemotron-3-nano-omni-30b-a3b-reasoning (2026-08-03 추가, 사용자
      지정): 논문 13개 로스터의 GLM-5.1(텍스트 전용이라 이 벤치마크는 이미지 필수 --
      평가 대상 불가)/Doubao-Seed-2.0-Pro(2.0-pro는 현재 미제공, 2.0-lite뿐이라 논문
      버전과 안 맞음) 자리를 대체하는 새 벤더들 -- xAI(논문의 Grok 4.2를 최신 버전 4.5로),
      Xiaomi(논문엔 없는 새 벤더), NVIDIA(마찬가지로 새 벤더, omni-multimodal reasoning
      모델). 셋 다 openrouter 스모크테스트 통과, 결함 없는 아이템에서 grok-4.5는 결함을
      지어내지 않고 정확히 판단. `nemotron-...:free`는 OpenRouter 무료 티어라 100-episode
      정식 실행 때 rate limit으로 막힐 수 있음 -- 실행 전 재확인 필요.

    **최종 정리 (2026-08-03, 사용자 철학 기반 재정렬)**: "벤더당 여러 티어"보다 "벤더 안
    명확한 이분법 + 벤더 다양성"으로 로스터 원칙을 재정의 -- GPT(프론티어 vs 구세대: gpt-5.5
    vs gpt-4o), Claude(추론 vs 일상: opus vs sonnet), Qwen(범용 vs VL 특화: 3.6-plus vs
    3-vl-32b), Gemini(상용 vs 오픈웨이트: 3.1-pro vs gemma-4), 그리고 짝 없이 벤더
    다양성만으로 넣는 kimi-k3/grok-4.5/thinkingmachines-inkling/nemotron-...(nvidia,
    inkling/nemotron은 깔끔한 이분법엔 안 맞지만 "벤치마크는 커버리지가 중요하다"는
    사용자 판단으로 유지). gpt-4o-mini는 한 번 제외 논의됐다가("바닥을 미리 정하고
    가는 건 발견이 아니라 조작") 최종적으로 유지 결정 -- 논문에서 gpt-4o-mini가 바닥
    앵커인 이유가 "그냥 약해서"가 아니라 "가장 단순한 FC 베이스라인조차 못 이기는
    유일한 모델"이라는 진단 포인트라서, 이 metric들의 변별력 자체를 검증할 캘리브레이션
    기준으로 필요하다는 반론(Claude)을 받아들임. gemini-3.6-flash/xiaomi-mimo-v2.5는
    이 철학에서 자리가 없어 제외하되 **코드는 지우지 않고 주석 처리**로 남김(사용자 지정
    컨벤션 -- 나중에 예산 여유 생기면 바로 되살릴 수 있게).
    """
    return [
        AgentConfig("FC-1", lambda: make_fixed_concession_policy(FC_RATES["FC-1"]), cites_evidence=False),
        AgentConfig("FC-10", lambda: make_fixed_concession_policy(FC_RATES["FC-10"]), cites_evidence=False),
        AgentConfig("FC-30", lambda: make_fixed_concession_policy(FC_RATES["FC-30"]), cites_evidence=False),
        AgentConfig("gpt-5.5", lambda: make_llm_agent_policy(model="gpt-5.5", provider="openai")),
        AgentConfig("gpt-4o-mini", lambda: make_llm_agent_policy(model="gpt-4o-mini", provider="openai")),
        AgentConfig("gpt-4o", lambda: make_llm_agent_policy(model="gpt-4o", provider="openai")),
        AgentConfig("claude-opus-4.6", lambda: make_llm_agent_policy(model="anthropic/claude-opus-4.6", provider="openrouter")),
        AgentConfig("claude-sonnet-4.6", lambda: make_llm_agent_policy(model="anthropic/claude-sonnet-4.6", provider="openrouter")),
        AgentConfig("qwen3.6-plus", lambda: make_llm_agent_policy(model="qwen/qwen3.6-plus", provider="openrouter")),
        AgentConfig("kimi-k3", lambda: make_llm_agent_policy(model="moonshotai/kimi-k3", provider="openrouter")),
        AgentConfig("gemini-3.1-pro-preview", lambda: make_llm_agent_policy(model="google/gemini-3.1-pro-preview", provider="openrouter")),
        #AgentConfig("gemini-3.6-flash", lambda: make_llm_agent_policy(model="google/gemini-3.6-flash", provider="openrouter")),
        AgentConfig("gemma-4-31b-it", lambda: make_llm_agent_policy(model="google/gemma-4-31b-it", provider="openrouter")),
        AgentConfig("qwen3-vl-32b-instruct", lambda: make_llm_agent_policy(model="qwen/qwen3-vl-32b-instruct", provider="openrouter")),
        AgentConfig("grok-4.5", lambda: make_llm_agent_policy(model="x-ai/grok-4.5", provider="openrouter")),
        #AgentConfig("mimo-v2.5", lambda: make_llm_agent_policy(model="xiaomi/mimo-v2.5", provider="openrouter")),
        AgentConfig("nemotron-3-nano-omni-reasoning", lambda: make_llm_agent_policy(model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", provider="openrouter")),
        AgentConfig("thinkingmachines-inkling", lambda: make_llm_agent_policy(model="thinkingmachines/inkling", provider="openrouter")),
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
    item_schedule: list[Item],
    category_p_max: dict[str, float],
    *,
    episodes: int,
    seed: int,
    family: str,
    regime: str,
    mapping: str,
    log_file=None,
) -> list[EpisodeRecord]:
    """(agent, mapping) 한 칸 -- episodes개 배치 실행.

    log_file(2026-08-02 추가, 기본 None=기록 안 함)이 주어지면 episode마다 끝나는 대로
    `episode_to_dict`(run_negotiation.py와 공유)로 직렬화해서 즉시 write+flush한다 --
    run_negotiation.py의 episodes.jsonl과 동일한 이유(중간에 죽어도 그때까지 기록은 남게)
    + 이 파일 자체의 "집계 aggregate만 저장하고 개별 episode 기록이 없다"는 문제를 같이
    해결한다 (사용자 지적, 2026-08-02). run_meta에 mapping을 추가로 태그해서, 나중에
    episodes.jsonl과 합쳐 봐도 어느 실험에서 나온 기록인지 구분 가능하게 한다.

    seed는 agent_cfg/items(=mapping)에 상관없이 매 셀마다 동일한 값에서 새로 시작한다
    (모듈 docstring "Confound 통제" 참고). **2026-07-30 갱신**: 예전엔 rng 하나를 셀
    전체에서 이어 썼는데, agent마다 협상이 끝나는 턴 수가 달라서 rng 소비 속도가 달라지고,
    그러면 두 agent의 협상이 처음 갈라지는 episode부터 그 다음 episode의 시나리오(아이템은
    같아도 역할/유보가격/K 등)가 agent마다 달라져 버리는 문제가 있었다(run_negotiation.py의
    episode_rng docstring에 실측 사례 기록). 그래서 지금은 episode_rng(seed, i)로 매
    episode마다 완전히 새 rng를 만든다 -- 이래야 "매핑/agent에 상관없이 같은 episode_idx는
    같은 시나리오"라는 이 함수의 원래 의도가 아이템 정체성뿐 아니라 역할/유보가격/K 등
    episode 조건 전체로 확장되어 성립한다.

    regime은 하나로 고정(run_negotiation.py --regime과 동일한 이유, decisions_log.md
    2026-07-30)하고, role×opener는 episode_idx로 균등 배정(role_opener_for)한다 -- 매핑을
    바꿔도 agent를 바꿔도 i가 같으면 항상 같은 regime/role/opener. item_schedule도 마찬가지로
    category_item_schedule(run_negotiation.py, 2026-08-02)이 episode_idx 기준으로 미리 만든
    카테고리 균등 배정 스케줄이라 매핑/agent 무관 i가 같으면 항상 같은 카테고리.
    """
    agent_policy = agent_cfg.make_policy()
    counterpart_policy = add_voice(make_counterpart_policy(family))
    stance_weights = stance_prior_for(family)
    run_meta = {"mapping": mapping, "agent": agent_cfg.label, "family": family, "regime": regime, "seed": seed}

    records = []
    # tqdm 진행률 표시 (2026-08-04 추가, 사용자 요청) -- 셀 하나가 최대 100episode라
    # 느린 agent(예: kimi-k3)는 몇 분씩 걸리는데 그동안 터미널에 아무 신호가 없어서
    # "죽었나 도는 중인가" 구분이 안 됐다. agent_cfg.label 붙여서 지금 몇 번째 (agent,
    # mapping) 셀인지도 같이 보이게 함.
    for i in tqdm(range(1, episodes + 1), desc=f"[{mapping}] {agent_cfg.label}", unit="ep"):
        item = item_schedule[i - 1]
        p_max = category_p_max[item.category]
        ep_rng = episode_rng(seed, i)
        role_A, opener = role_opener_for(i)
        record = run_one_episode(
            ep_rng, agent_policy, counterpart_policy, stance_weights, item, p_max,
            regimes=(regime,), role_A=role_A, opener=opener,
        )
        records.append(record)
        if log_file is not None:
            log_file.write(json.dumps(episode_to_dict(record, episode_idx=i, run_meta=run_meta), ensure_ascii=False) + "\n")
            log_file.flush()
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


def _rank_vector(scores: dict[str, dict], agent_labels: list[str], metric: str) -> list[float] | None:
    """mapping 하나의 scores[agent_label] -> metric 벡터. agent 중 하나라도 그 metric이
    None(표본 부족 등으로 정의 안 됨)이면 그 매핑에서 순위 자체가 안 나오므로 None 반환."""
    values = [scores[label].get(metric) for label in agent_labels]
    if any(v is None for v in values):
        return None
    return values


def rho_report(
    scores_by_mapping: dict[str, dict[str, dict]],
    agent_labels: list[str],
    mappings: list[str],
    *,
    metrics: tuple[str, ...] = RANKING_METRICS,
    citation_capable_labels: set[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """모든 매핑 쌍(uv, C(n,2)개) x metrics 별 spearman_rho. 슬라이드3 ③~④가 "모든 매핑
    쌍"이라고 했지 인접 쌍만이 아니므로 itertools.combinations로 전체 쌍을 본다.

    metric이 CITATION_METRICS에 속하면 agent_labels를 citation_capable_labels로만 걸러서
    쓴다 (CITATION_METRICS 상수 정의 참고 -- FC 계열을 포함한 채로는 이 지표들이 구조적으로
    항상 None이라 순위 비교 자체가 불가능해짐).

    `AgentConfig`가 아니라 label 리스트(+ citation 가능 여부는 별도 set)만 받는다
    (2026-08-02 리팩터 -- 사용자 지적: "spearman 돌릴 지표를 미리 정해야 하는 거 아니냐").
    아니다 -- `scores_by_mapping`(=`compute_metrics()`의 전체 dict, 모든 metric key 포함)이
    이미 저장돼 있으므로, 실험을 다시 안 돌리고 저장된 `mapping_robustness.json`을 읽어서
    `metrics=(...)`에 원하는 걸 아무거나 넘겨 이 함수를 그대로 재사용할 수 있다 -- 이 함수가
    `AgentConfig`(agent_policy 클로저 포함, 실행 중에만 의미 있음)에 묶여있으면 그 재사용이
    막히므로 label 기반으로 뺐다."""
    citation_capable_labels = citation_capable_labels if citation_capable_labels is not None else set(agent_labels)
    report: dict[str, dict[str, float | None]] = {}
    for metric in metrics:
        eligible = [l for l in agent_labels if l in citation_capable_labels] if metric in CITATION_METRICS else agent_labels
        pair_rhos: dict[str, float | None] = {}
        for m1, m2 in itertools.combinations(mappings, 2):
            xs = _rank_vector(scores_by_mapping[m1], eligible, metric)
            ys = _rank_vector(scores_by_mapping[m2], eligible, metric)
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
        "--regime", type=str, default="overlap", choices=list(REGIMES),
        help="episode regime 고정 (기본 overlap, run_negotiation.py --regime과 동일 이유). "
             "FAGR-는 이 고정 때문에 영구적으로 None -- decisions_log.md 2026-07-30 참고",
    )
    parser.add_argument(
        "--mappings", type=str, nargs="+", default=list(PRICE_IMPACT_MAPPINGS.keys()),
        choices=list(PRICE_IMPACT_MAPPINGS.keys()), help="비교할 매핑 부분집합 (기본: 전체 4개)",
    )
    parser.add_argument(
        "--agents", type=str, nargs="+", default=None,
        help="비교할 agent label 부분집합 (기본: default_agents() 8개 전부) -- "
             "예: --agents FC-1 kimi-k3. 데모/스모크테스트용, rho는 최소 2개 agent가 있어야 의미 있음",
    )
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_PATH), help=f"결과 JSON 저장 경로 (기본값: {DEFAULT_OUT_PATH})")
    parser.add_argument(
        "--episodes-log-path", type=str, default=str(DEFAULT_EPISODES_LOG_PATH),
        help=f"개별 episode 기록(transcript 등) JSONL 경로 (기본값: {DEFAULT_EPISODES_LOG_PATH})",
    )
    parser.add_argument(
        "--no-episodes-log", dest="episodes_log", action="store_false", default=True,
        help="개별 episode 기록 저장 끄기 -- 기본은 항상 저장, run_negotiation.py --no-log와 동일한 용도",
    )
    args = parser.parse_args()

    agents = default_agents()
    if args.agents is not None:
        wanted = set(args.agents)
        unknown = wanted - {a.label for a in agents}
        if unknown:
            raise SystemExit(f"--agents에 모르는 label: {sorted(unknown)} (전체: {[a.label for a in agents]})")
        agents = [a for a in agents if a.label in wanted]
    agent_labels = [a.label for a in agents]
    citation_capable_labels = {a.label for a in agents if a.cites_evidence}
    mappings = args.mappings

    print(f"mappings={mappings} agents={[a.label for a in agents]} episodes/cell={args.episodes} family={args.family} regime={args.regime}")
    print()

    # mapping -> items: data_loader.load_items가 defect_type 기반으로 price_impact를 매핑별로
    # 다시 계산한다 (결함 identity는 파일 순서에서만 오므로 매핑 무관 고정, 2026-07-26 결정).
    items_by_mapping = {
        m: load_items(args.data, mapping=PRICE_IMPACT_MAPPINGS[m]) for m in mappings
    }
    category_p_max_by_mapping = {m: _category_p_max(items) for m, items in items_by_mapping.items()}
    # 카테고리 균등 배정 스케줄 (2026-08-02, run_negotiation.py의 category_item_schedule 참고) --
    # 셔플만으로는 카테고리 비율이 운에 맡겨져서(예: car 38개가 --episodes 100 기준 적게/안
    # 뽑힐 위험) 카테고리 marginal을 role_opener_for처럼 정확히 균등 배정하는 걸로 교체했다.
    # 매핑마다 동일 seed로 호출하므로(같은 원본 파일 순서에서 카테고리 구성이 오므로) 매핑끼리
    # 동일한 스케줄이 나온다 -- "매핑/agent 무관 i가 같으면 같은 시나리오"라는 Confound 통제
    # 불변식이 아이템 축에서도 유지된다 (episode_rng와 같은 이유).
    item_schedule_by_mapping = {
        m: category_item_schedule(items, args.episodes, args.seed) for m, items in items_by_mapping.items()
    }

    # scores_by_mapping[mapping][agent_label] = compute_metrics(records) (전체 episode 합산)
    # quadrant_scores_by_mapping[mapping][quadrant][agent_label] = compute_metrics(quadrant 부분집합)
    scores_by_mapping: dict[str, dict[str, dict]] = {}
    quadrant_scores_by_mapping: dict[str, dict[str, dict[str, dict]]] = {}

    out_path = Path(args.out)
    config = {
        "data": args.data,
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "family": args.family,
        "regime": args.regime,
        "mappings": mappings,
        "agents": agent_labels,
    }

    overall_rho: dict[str, dict[str, float | None]] = {}
    quadrant_rho: dict[str, dict[str, dict[str, float | None]]] = {}

    episodes_log_file = None
    if args.episodes_log:
        episodes_log_path = Path(args.episodes_log_path)
        episodes_log_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_log_file = open(episodes_log_path, "a", encoding="utf-8")  # append -- 여러 번 돌린 기록이 누적됨

    for m in mappings:
        scores_by_mapping[m] = {}
        quadrant_scores_by_mapping[m] = {q: {} for q in QUADRANTS}
        item_schedule = item_schedule_by_mapping[m]
        category_p_max = category_p_max_by_mapping[m]

        for agent_cfg in agents:
            records = run_cell(
                agent_cfg, item_schedule, category_p_max,
                episodes=args.episodes, seed=args.seed, family=args.family, regime=args.regime,
                mapping=m, log_file=episodes_log_file,
            )
            scores_by_mapping[m][agent_cfg.label] = compute_metrics(records)

            for q, q_records in records_by_quadrant(records).items():
                # compute_metrics는 빈 리스트도 안전하게 처리(전부 None) -- 빈/비어있지 않은
                # 케이스를 분기하지 않아야 모든 셀이 같은 키 구조를 갖는다.
                q_metrics = compute_metrics(q_records)
                q_metrics["n"] = len(q_records)
                quadrant_scores_by_mapping[m][q][agent_cfg.label] = q_metrics

            print(f"[{m}] {agent_cfg.label}: SE+={scores_by_mapping[m][agent_cfg.label]['SE+']}")

            # 중간저장 (2026-08-02 추가): 32칸(agent x mapping) 다 끝나야만 저장하던 예전
            # 방식은 중간에 죽으면(실제로 qwen/kimi 응답 파싱 버그로 두 번 죽었음) 그 앞까지
            # 낸 결과가 통째로 날아갔다. 매 셀이 끝날 때마다 여기까지의 결과를 덮어써서, 죽어도
            # 완료된 만큼은 파일에 남긴다 (재시작해서 이어달리기까지는 아직 아님 -- 그냥 안전망).
            done_mappings = [mm for mm, cell in scores_by_mapping.items() if len(cell) == len(agents)]
            overall_rho = rho_report(scores_by_mapping, agent_labels, done_mappings, citation_capable_labels=citation_capable_labels)
            quadrant_rho = {
                q: rho_report(
                    {mm: quadrant_scores_by_mapping[mm][q] for mm in done_mappings},
                    agent_labels, done_mappings, citation_capable_labels=citation_capable_labels,
                )
                for q in QUADRANTS
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "config": config,
                        "scores_by_mapping": scores_by_mapping,
                        "quadrant_scores_by_mapping": quadrant_scores_by_mapping,
                        "overall_rho": overall_rho,
                        "quadrant_rho": quadrant_rho,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    if episodes_log_file:
        episodes_log_file.close()

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

    print(f"\n(집계 결과를 {out_path}에 저장 -- 매 셀마다 이미 중간저장됨)")
    if episodes_log_file:
        print(f"(개별 episode 기록을 {args.episodes_log_path}에 이어씀)")


if __name__ == "__main__":
    main()
