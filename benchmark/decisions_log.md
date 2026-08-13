# benchmark/decisions_log.md

`benchmark/CLAUDE.md`에서 분리한 상세 결정 근거/검증 기록 아카이브 (2026-07-26 분리).
CLAUDE.md는 "지금 상태 + 다음 할 일"만 가볍게 유지하고, "왜 이렇게 결정했는지"의
전체 경위·검증 숫자·뒤집힌 시행착오는 전부 여기에 시간순으로 쌓는다 (학술제 논문
methods/limitations 쓸 때 근거 자료로 재사용 목적).

새 설계 결정이 생기면 이 파일 맨 아래에 날짜와 함께 추가할 것 (CLAUDE.md 대신).

---

## 구현 범위 (2026-07-11 설계 논의, 항목별 결정 경위)

- [x] 환경(Γ) — 가격 geometry: 프로토타입 단계에서는 `implementation/env.py`의
  z/q/regime 추상 샘플링을 **그대로 재사용**한다. `Defect`(ground truth 결함)
  목록은 독립적인 별도 데이터로 얹되, reservation price(z/q) 생성과는 아직
  논리적으로 연결하지 않는다. (2026-07-11 결정)
  - **⚠️ Simplification — 궁극적으로 반드시 재검토할 것**: 결함이 심각한
    아이템인데 ZOPA(z/q)는 결함과 무관하게 랜덤으로 나올 수 있음 — 가격
    geometry와 결함 ground truth가 서로 독립적으로 샘플링되는 상태. 원래는
    `r_buyer`/`r_seller`(또는 `listing_price`)가 `Defect.price_impact` 합산에서
    논리적으로 유도돼야 함. 오늘은 마감(프로토타입) 때문에 범위를 좁혔지만,
    이 갭은 grounding/utilization metric의 타당성 자체에 영향을 줄 수 있는
    리스크라 방치하면 안 됨. **실제 환경 코드를 쓸 때 이 문단을 그대로
    price-geometry 생성 함수 docstring/주석으로 옮겨 남길 것.**
    → **2026-07-23에 `fair_price(item)` 앵커링으로 해소됨** (아래 "로더 +
    price geometry↔defect 연결 완료" 절 참고).
- [x] 환경(Γ) — evidence 데이터 모델: `Item`(category/title/description/
  listing_price/image_ref) + `Defect`(id/description/price_impact/salience)
  구조로 구현 완료 (`benchmark/env.py`). `Action`에도 구조화된 인용 필드
  `cited_defect_ids`를 추가 — 자유 텍스트 `message`만으로는 룰 매처가 채점 못
  함.
  - **`Defect.salience`(육안 발견 난이도) — 스코프 아웃 확정 (2026-07-11)**:
    원래 결함 이미지 합성 품질이 grounding accuracy 측정의 confound가 되는
    문제에 대응하려고 넣으려 했으나, 제대로 재려면 결함을 합성한 사람이 아닌
    **제3의 독립 코더 3인 이상**이 결함 위치/종류를 모르는 **블라인드** 상태로
    직접 찾아내야 하고, 그 결과의 신뢰도를 **ICC(intraclass correlation)**로
    확인해야 함 — 이건 8주 스코프 밖의 별도 작업이라 지금은 보류. 연구자
    본인이 혼자 대충 매기는 것도 안 하기로 함(이미 결함 위치를 아는 상태라
    편향돼서 confound를 막으려던 목적 자체가 무의미해짐). 코드상
    `Defect.salience`는 `float | None = None`으로 항상 `None` — 어떤 로직도
    이 필드를 참조하지 않음. 상세: `benchmark/data_spec.md` "향후 계획" 절.
- [x] 커널 — family 선정 (2026-07-11 결정): 프로토타입은 `Candid` 하나만 사용
  (`type_instrumental` econ + `accurate` cue + 균등 stance). 6 family 중 가장
  "특이 성향 없는" 베이스라인이라, grounding/utilization metric이 제대로
  작동하는지부터 노이즈 없이 검증하기 위함. `econ`/`cue`/`stance_prior` 조합이
  달라지면 counterpart의 accept/counter-offer 패턴이 달라져 `utilization_proximity`
  및 기존 Table 1 5종(SE+/AGR+/...)에는 간접 영향을 주지만, `citation_precision`/
  `citation_coverage`/`hallucination_rate`는 agent 출력과 ground truth만 비교하므로
  family와 무관. 2번째 family(예: 난이도 상 variant로 `Adversarial`)는 시간 남으면
  8주 스코프 안에서 추가 고려.
  → **2026-07-23에 6-family 전체로 확장됨** (아래 "6-family 이식 완료" 절 참고).
- [x] 커널 — evidence를 `accept_prob`에 반영 (2026-07-11 결정):
  ```
  sign = +1 if role_A == BUYER else -1
  g = ALPHA*delta_bar + ... + sign * EVIDENCE_BONUS * 1{이번 offer(agent의 현재
        액션)가 실제 Defect를 정확히 인용함}
  ```
  `kernel.py`의 `_agent_sign` 패턴(역할별 부호 뒤집기)을 그대로 재사용. BUYER가
  실제 결함을 인용하며 낮은 가격을 부르면 그 가격의 근거가 강해져 counterpart
  (SELLER)의 accept_prob↑. SELLER가 자기 결함을 인정하면서 가격을 안 낮추면
  그 높은 가격의 근거가 약해져 counterpart(BUYER)의 accept_prob↓ — 단순히
  "역할별로 다르게 처리"가 아니라 "확인된 결함에 가격이 안 맞으면 설득력이
  떨어진다"는 일관된 논리. 보너스/마이너스 크기는 프로토타입 단계에서 대칭
  (`EVIDENCE_BONUS` 하나로 부호만 반전)으로 단순화 — 비대칭이 필요해 보이면
  나중에 분리.
  - **최초 설계에서 두 번 수정된 경위**: ①"agent가 잘 설계됐다면 self-defeating한
    인용은 안 할 것"이라 가정하고 role 무관 flat bonus로 시작 → 이 벤치마크의
    목적 자체가 "agent가 VLM+전략 추론을 동시에 못 해낼 수 있다"를 진단하는
    것이라 agent 역량을 전제로 kernel을 설계하면 안 된다는 지적으로 role
    조건부(BUYER만 보너스, SELLER는 무효과)로 수정 → "SELLER가 결함을 인정하며
    가격을 안 낮추면 그 자체로 설득력이 떨어져야 한다"는 지적으로 SELLER는
    무효과가 아니라 마이너스로 최종 수정.
  → **2026-07-25에 EVIDENCE_BONUS(단일 상수) → QUADRANT_BONUS(quadrant별 lookup)로
    교체됨** (아래 "quadrant를 사후 라벨에서 실시간 커널 파라미터로 승격" 절 참고).
- [x] 메트릭 — grounding accuracy: 기존 Table 1의 5개(SE+/AGR+/CSE+/AGR-/CritViol%)는
  유지하고, 아래 3개를 새로 추가한다 (2026-07-11 결정). `C` = agent가 episode에서
  인용한 `cited_defect_ids`의 합집합, `D` = 그 아이템의 실제 `Defect` 집합.
  - **`citation_precision`**: `|C∩D| / |C|`. episode마다 계산 후 평균. `C`가
    빈(=한 번도 인용 안 한) episode는 정의상 `0/0`이라 **평균 분모에서 제외**
    (`metrics.py`의 `se_plus`가 `_feasible`이 빈 경우 `None` 반환하는 것과 동일
    패턴).
  - **`citation_coverage`** (recall): `|C∩D| / |D|`. episode마다 계산 후 **전체
    episode에 대해 평균** — 인용을 아예 안 한 episode도 "커버리지 0%"라는 유효한
    신호라 제외하지 않음.
  - **`hallucination_rate`**: `C \ D`(실제 없는 결함을 지어낸 것)가 한 번이라도
    있었던 episode의 비율. `CritViol%`처럼 **평균이 아니라 episode 단위 이진
    플래그의 비율**로 정의 — `citation_precision`의 단순 여집합(`1-precision`)이
    되지 않도록 일부러 다르게 설계함 (평균 정확도는 높아도 위험한 케이스가
    존재하는지는 별도 축으로 봐야 하므로, `AGR-`/`CritViol%`가 평균이 아니라
    비율로 "위험 발생 여부"를 따로 보는 것과 같은 이유).
  - 전제: `Action`에 구조화된 `cited_defect_ids` 필드가 있어야 룰 매처로 채점
    가능 (위 "환경 — evidence 데이터 모델" 항목과 연결됨, 아직 미구현).
- [x] 메트릭 — utilization proximity (2026-07-11 결정): "citation turn"(agent
  액션의 `cited_defect_ids`가 비어있지 않은 턴 `k`)의 offer가 agent 자신의 직전
  offer 대비 **양보(concession)**였는지로 "활용됐는지" 판정한다
  (`kernel.py`의 `_agent_sign`/`concede_magnitude_speed` 패턴을 counterpart 대신
  agent 자신의 가격 시퀀스에 적용). `utilization_proximity(episode) = 활용된
  citation 수 / 전체 citation 수`, citation이 없으면 `None`. 여러 episode에
  걸쳐 평균.
  - **⚠️ Simplification — 스코프 아웃, 재검토 필요**: 이 정의는 `Action.decision
    == OFFER`인 citation turn에만 적용 가능(가격이 있어야 "양보인지" 판정
    가능). `ACCEPT`/`REJECT` 턴에 붙은 citation(예: "이 결함 때문에 이 가격에
    받아들인다"/"이 결함 때문에 이 가격은 못 받는다"도 논리적 근거 제시로서
    "활용"에 해당할 수 있음)은 오늘 프로토타입 집계 대상에서 **제외**한다.
    나중에 "ACCEPT/REJECT에 대한 근거 적절성"을 별도 축으로 다룰지 재검토할
    것. **실제 코드로 옮길 때 이 문단을 `utilization_proximity` 함수
    docstring에 남길 것.**
- [x] evidence-agnostic 프레이밍 여부 — **스코프 아웃 확정** (2026-07-11):
  `Paper/idea.md`(2026-07-09 항목)의 "evaluation awareness" ablation은 8주
  스코프에 포함하지 않는다. 1차 측정 목표(같은 증거를 얼마나 정확·전략적으로
  쓰는가)와 직접 안 엮이는 별개 연구 질문이라, `idea.md`에 future work로만
  남겨두고 여기 구현 범위에서는 다루지 않음. (2026-07-10 감정/관계 문헌
  아이디어와 동일하게 future work 처리.)

## Agent (2026-07-11)
- `implementation/`과 다르게, 여기서는 **실제 VLM 기반 LLM agent**를 평가 대상으로
  붙인다 (더미/규칙기반 아님).
- counterpart는 여전히 규칙 기반 kernel이다 (VLM 아님) — `implementation/kernel.py`
  방식 그대로 계승.

## 진행 상황 (2026-07-11 세션)
- 이 문서(benchmark/CLAUDE.md) 초안 작성 (2026-07-11).
- 환경 설계 논의 시작 (2026-07-11): evidence 데이터 모델(Item/Defect) 초안,
  가격 geometry는 프로토타입 단계에서 기존 z/q 생성기 재사용하기로 결정
  (위 "구현 범위"의 Simplification 참고 — 나중에 재검토 필요).
- Timing bucket 정의 완료 (2026-07-11): `k <= K//3` early, `k <= 2*(K//3)` mid,
  나머지 late — 균등 3등분. 협상 결과에 영향 주는 게 아니라 **사후 분석 전용**이라
  `metrics.py`의 `compute_metrics` 채점 파이프라인에는 안 넣고 별도 분석
  스크립트/함수로 분리하기로 함 (결함 인용 시점 vs. outcome 교차분석용).
- Grounding accuracy metric 3종 확정 (2026-07-11): `citation_precision`,
  `citation_coverage`(recall), `hallucination_rate`(episode 단위 이진 플래그) —
  자세한 정의는 위 "구현 범위" 참고. precision과 hallucination_rate를 서로의
  단순 여집합으로 만들지 않기 위해 hallucination_rate만 CritViol%식 이진 플래그로
  설계한 게 핵심 결정.
- Utilization proximity metric 확정 (2026-07-11): citation turn의 offer가 agent
  자신의 직전 offer 대비 양보였는지로 판정, episode당 "활용 비율"을 내고 평균.
  OFFER 턴에 붙은 citation만 다루고 ACCEPT/REJECT 턴 citation은 스코프 아웃
  (Simplification, 위 "구현 범위" 참고).
- 커널 family(`Candid` 단독) 및 evidence→`accept_prob` 반영 방식(역할별 부호
  반전 `EVIDENCE_BONUS`) 확정. evidence-agnostic 프레이밍 ablation은 스코프
  아웃 확정 (2026-07-11).
- **"구현 범위"의 미확정 항목이 모두 결정됨 — 오늘 설계 논의 단계 완료.**
- `benchmark/env.py`/`kernel.py`/`metrics.py` 프로토타입 구현 완료 (2026-07-11).
  세 파일 다 `implementation/`을 import하지 않는 독립 사본+확장. 두 개의 dummy
  agent(정직하게 실제 결함만 인용 vs. 없는 결함을 지어냄)로 env+kernel+metrics를
  end-to-end로 돌려 검증: `citation_precision`/`hallucination_rate`/
  `citation_coverage`가 "좋은 인용 vs. 나쁜 인용"을 의도한 대로 구분해냄
  (honest: precision=1.0/hallucination=0.0, liar: precision=0.0/hallucination=1.0).
  `kernel.py`의 `evidence_term_for`도 role별 부호(BUYER +/SELLER -)가 정확히
  뒤집히는 것 단위 테스트로 확인.
- **오늘 아직 안 한 것 (다음 세션 이어갈 목록, 2026-07-11 기준)**:
  - `검증 방식` 절 여전히 미정 (지금까지는 `implementation/`처럼 즉석 assert
    스크립트로만 확인, 정식 sanity check 스크립트로 정리 안 됨)
  - `Item.image_ref`/결함 이미지 합성 파이프라인 — 실제 이미지 없이 placeholder
    문자열만 있음 (스코프 밖으로 명시했던 부분, 그대로 유지)
  - VLM agent 연동 — 아직 dummy 규칙 기반 agent로만 배관 검증함, 실제 VLM 붙이는
    작업은 미시작
  - 2번째 counterpart family(예: Adversarial) 미추가
  - BE_type/belief_error — 오늘 설계 논의 대상이 아니었어서 `metrics.py`에 없음
  - price geometry ↔ 결함 연결 (앞서 남긴 Simplification) 재검토 안 됨
- `benchmark/llm_agent.py` 추가 (2026-07-11): 실제 OpenAI(GPT) 모델이 협상하는
  `make_llm_agent_policy()` — `kernel.py`의 `make_counterpart_policy`와 같은 팩토리
  패턴. `env.py`/`kernel.py`와 엮어서 실제 episode 1개 end-to-end 검증 완료 (GPT가
  seller로 협상, Candid counterpart가 합리적인 가격에 ACCEPT, 프로토콜 위반 없음).
  Provider는 Claude 대신 OpenAI로 결정(2026-07-11 논의). 이미지가 아직 없어서
  `cited_defect_ids`는 항상 비어있음 — VLM 통합은 미시작, 순수 텍스트 협상 배관만
  검증된 상태. `model` 기본값은 임시 placeholder, 실제 모델 선정은 아직 논의 안 됨.
  가상환경(`.venv`) + `requirements.txt`(openai, python-dotenv) + `.env`(gitignore
  처리) 세팅 완료. `.claude/settings.json`에 `.env` 파일 Read/Edit/Grep deny 규칙
  추가 (API 키 보호, sandbox 미사용이라 Bash 우회 가능성은 남아있음).
- `benchmark/voice.py` 추가 (2026-07-11): counterpart의 "voice" 레이어 — 논문
  Appendix C.5.4(Language Realization)와 대응. `kernel.py`가 이미 확정한
  decision/price/sentiment/posture를 그대로 받아 자연어 메시지로만 렌더링하고
  economic 결정 자체는 절대 안 바꿈. `add_voice(counterpart_policy)`가
  `make_counterpart_policy(family)`를 감싸는 wrapper 구조 — `kernel.py`는
  OpenAI 의존성 없는 순수 시뮬레이터로 유지(API 키 없이도 기존 sanity check가
  계속 돌아가야 하므로), voice는 그 위에 얹는 선택적 레이어로 분리.
  **`VOICE_MODEL`은 `llm_agent.py`의 agent 모델과 완전히 분리·고정** — 처음엔
  agent와 같은 모델을 재사용하려다, "voice가 agent와 같은 모델이면 자기 편향
  (self-bias) 오염이 생긴다"는 지적으로 수정함(논문도 "voice model is fixed to
  GPT-5.2"라고 평가 대상과 분리해서 고정).
  - 모델 배분 최종 결정(2026-07-11): **agent = `gpt-4o`**(전략적 판단 필요),
    **voice = `gpt-4o-mini`**(이미 확정된 decision을 문장으로 포장만 하는 좁은
    렌더링 작업이라 가벼운 모델로 충분) — 처음엔 반대로(agent=mini/voice=4o)
    갔다가, "결국 같은 4o 계열 아니냐"는 지적에 "완전히 다른 모델은 아니지만
    같은 회사/세대라 스타일 공유 가능성은 있다"고 인정한 뒤, 최소한 **더 어려운
    작업에 더 강한 모델을 배분**하는 쪽으로 재조정. 완전히 다른 provider(예:
    voice=Claude)로 분리하는 안은 오늘 스코프 아웃(이미 OpenAI 하나로 가기로
    정한 결정을 뒤집는 비용이 큼 — 다음에 재검토 가능).
  `run_negotiation.py`로 실제 episode 돌려 counterpart 메시지가 실제로 채워지는
  것과, 모델 스왑 후에도 정상 동작하는 것 확인 완료.
- `run_negotiation.py`에 `--episodes N` 배치 실행 + `metrics.py`의
  `compute_metrics()` 집계 출력 추가 (2026-07-11). `--episodes 1`(기본값)은 기존
  transcript 전체 출력 그대로, `N>1`이면 판별 한 줄 요약 + 마지막에 metric 표.
  - **정정된 판단**: 처음엔 "voice는 metric에 영향 없으니 배치 모드에서 기본
    off로 끄자"고 하려다, "agent가 counterpart의 message를 실제로 보는 거
    아니냐"는 지적으로 재검토함 — `llm_agent.py`의 `format_history`가 agent
    프롬프트에 counterpart의 message를 그대로 넣기 때문에, voice를 끄면 agent가
    보는 정보 자체가 줄어 agent의 실제 판단(가격/결정)이 달라질 수 있고, 그
    위에서 계산되는 SE+/AGR+ 같은 가격 경로 의존 metric도 값이 바뀔 수 있음.
    즉 voice off는 "metric에 영향 없는 최적화"가 아니라 "측정 조건 자체를
    바꾸는 변경"이었음 — `citation_precision`류(agent 자신의 cited_defect_ids만
    봄)에만 영향 없다는 게 맞는 말이었지, 전체 metric에 대한 얘기는 아니었음.
    **최종 결정: voice는 항상 기본 on**, `--no-voice`는 코드가 안 죽는지만 빠르게
    확인하는 smoke test 전용으로만 문서화(실제 보고할 metric은 항상 voice on
    상태로 뽑을 것).
- **오늘 to-do 3개(환경 프로토타입 방향 / grounding+utilization metric 스펙 /
  timing bucket) 설계 논의 완료.** 다음 단계: 커널 family 선정 + 이 설계들을
  실제 코드(`benchmark/env.py` 등)로 옮기기 시작.

## 2026-07-23 — 교수님 피드백 반영: 측정 축 재정의

교수님 코멘트를 받아 지난 세션에서 정한 "순수 TERMS-Bench 보류 항목(BE_type, 6-family,
오라클) 다 채우기" 계획을 재검토함. 핵심 코멘트: **"TERMS-Bench는 상대방에 대한 믿음을
기반으로 하지만, 우리는 시각 정보가 주는 영향을 보고자 하는 것 — 보고자 하는 게 다르다."**

- **`BE_type` — 드롭 확정**: TERMS-Bench의 opponent-modeling 축(BE_type)은 counterpart의
  숨겨진 **심리/전략 상태**(r_B/κ_B/η_B)에 대한 belief 정확도를 잰다. 우리가 재려는 건
  이미지 속 **객관적 결함 정보**를 지각(detection)→가치환산(grounding)→전략반영(utilization)
  하는 능력이라 성격이 다른 축. 지난 세션엔 "시간이 없어서 보류"로 남겨뒀었는데, 사실은
  "이 벤치마크의 측정 대상이 아니다"가 더 근본적인 이유였음 — 되짚어서 다행.
- **`CritViol%` — 드롭 확정**: 교수님 코멘트. 프론티어 모델 기준 원 논문에서도 대부분
  0%(Table 2)라 우리 세팅에서도 핵심 차별화 지점이 아닐 가능성 높음.
- **오라클(Eq.4/Appendix D DP) — 스코프 아웃 재확정, 근거 보강**: 기존엔 "10페이지 분량,
  투자 대비 가치 낮음"(2026-07-10, `implementation/CLAUDE.md`)이 이유였는데, 이제
  "애초에 counterpart 불확실성 분해 프레임이라 우리 측정 대상과 개념적으로 안 맞음"이라는
  더 근본적 이유가 추가됨. 계속 미구현.
- **6-family — 유지(구현 예정)**: 위 재프레이밍과 별개로, 협상 자체의 로버스트니스
  검증(다양한 counterpart 성향에 agent가 안정적으로 대응하는가)은 여전히 유효한 질문이라
  `implementation/kernel.py`의 6 family를 `benchmark/kernel.py`로 이식하기로 함.
- **`SE+`/`AGR+`/`CSE+`/`FAGR-` — 유지**: outcome 품질(surplus, 합의 적절성) 자체는
  visual 축과 무관하게 여전히 봐야 할 기본 성과 지표라 유지.

### 신규 비교군 (교수님 코멘트)
1. **visual 유/무**: 같은 아이템/counterpart로 visual evidence를 주는 조건과 안 주는
   조건을 비교 — pulse.pptx 슬라이드6에서 이미 나온 "gap 없으면 태스크 무의미" 대전제 실험.
2. **visual 있음 + LLM agent vs visual 있음 + fixed-concession baseline**: TERMS-Bench
   Table 2의 `Fixed 30%/10%/1%` 바닥 앵커를 우리 세팅에 이식. "정보에 접근 가능한 것만으론
   부족하고, 실제로 추론해서 반영해야 한다"는 것을 보여주는 비교.

### 신규 metric 2종 (설계 확정, 이 시점엔 미구현 -- 같은 날 뒤에 코드화 완료됨, 아래
"심각도 비례/subtle-misaligned gap 코드화 완료" 절 참고)
- **심각도 비례(severity calibration)**: citation turn(`cited_defect_ids` 비어있지 않은
  `Offer` turn) 중 **결함을 정확히 하나만 인용한 턴에 한정**하여, (a) 그 턴의 가격
  변화폭(`kernel.py`의 `ConcedeMagnitude` 계산 방식 재사용, 직전 자기 offer 대비/R
  정규화)과 (b) 인용된 결함의 ground-truth `price_impact` 사이의 Spearman ρ를 계산.
  ρ가 높을수록 "결함이 심각할수록 실제로 더 크게 깎는다"는 calibration이 잘 된 것.
  - **Simplification (다중 인용턴 제외)**: 한 턴에 결함을 여러 개 인용하면 어느 결함의
    `price_impact`를 기준으로 삼을지 애매해짐(합/최댓값/평균 중 택1 필요). 지금은 아이템당
    결함이 1~2개뿐이라 다중 인용이 드물어 이 제약이 거의 안 걸리지만, 결함 개수를 늘리면
    반드시 재검토할 것 — `idea.md`(2026-07-15, 태그 어휘 관련 미해결 하위 문제)와 같은
    시점에 같이 다룰 것.
- **Subtle-Misaligned surplus gap**: pulse.pptx 슬라이드2의 4분면 중 가장 어려운
  "안 보이지만 가치 영향 큰" 결함이 있는 아이템의 episode들을, 그 결함 id가
  `cited_defect_ids`에 한 번이라도 등장했는지(잡음) 여부로 두 그룹으로 나눠 평균 `CSE+`
  차이를 리포트: `gap = CSE+(잡음) − CSE+(놓침)`.
  - **주의**: agent가 스스로 인용했는지로 그룹을 나누는 **관찰적 비교**라 순수 인과추론은
    아님 (무작위 개입이 아니라 agent 행동에 따른 사후 분리).
  - **표본 크기 문제 — 구현이 아니라 데이터 문제**: 이 quadrant 아이템이 몇 개나 준비될지는
    데이터 준비 담당 팀원에게 확인 필요. 표본이 적으면 gap 숫자가 불안정할 수 있으니,
    **그룹별 episode 수(n)를 gap과 함께 항상 같이 리포트**하기로 함 (숫자만 보고 오독하는
    것 방지 — `citation_precision`이 빈 분모 episode를 평균에서 제외하는 것과 같은 이유의
    "신뢰도 가시화" 조치).

## 2026-07-23 (계속) — 6-family 이식 완료

`benchmark/kernel.py`의 `ECON_PRESETS`/`FAMILIES`에 `implementation/kernel.py`의
나머지 3개 프리셋(`high_reactivity`/`moderate_stochastic`/`hardball`) + 5개
family(`Taciturn`/`Expressive`/`Strategic`/`Stochastic`/`Adversarial`)를 그대로
복사해 이식. section A-F(공용 유틸, history feature, accept/walkaway+evidence
항, counter-offer, opening-offer, cue 생성)는 두 파일이 이미 동일했어서 손댈
필요 없었음 — `make_counterpart_policy`가 `FAMILIES`/`ECON_PRESETS` 딕셔너리를
제네릭하게 조회하는 구조라 이식만으로 바로 동작. `run_negotiation.py`의
`--family` 옵션도 `FAMILIES.keys()`를 동적으로 읽어서 코드 변경 불필요.

검증: 규칙기반 dummy agent(`implementation/agent.py`와 동일 로직)로 family당
300 episode 실행 (evidence 없는 순수 협상 경로만 확인 -- item은 있지만 agent가
`cited_defect_ids`를 안 채우므로 `evidence_term_for`는 항상 0 반환). 크래시
없음. `implementation/`에서 이미 확인됐던 패턴 재현: Taciturn/Strategic은
cue가 항상 `(neutral, Hold)`, Adversarial은 항상 `(negative, Pressure)`로
붕괴; 전 family `AGR-=0.0`/`CritViol%=0.0`(IR 지키는 agent라 구조적으로
보장); Adversarial이 SE+/AGR+ 최저(hardball preset이라 예상대로 가장 어려움).

## 2026-07-23 (계속) — 심각도 비례/subtle-misaligned gap 코드화 완료

`severity_calibration`, `subtle_misaligned_gap` 구현 (`benchmark/metrics.py`), `compute_metrics`에
`severity_calibration_rho`/`_n`, `subtle_misaligned_gap`/`_n_caught`/`_n_missed` 5개 키로 노출.

- **선행 스키마 추가**: `Defect.quadrant`(env.py) 신규 필드 -- "obvious_aligned" 등 pulse.pptx
  슬라이드2/4의 4분면 태그. `salience`(측정 필요, 스코프 아웃)와 다르게 이건 결함 합성 시
  제작자가 부여하는 설계 레이블이라 salience 스코프아웃 결정과 안 충돌함. `_DEFECT_CATALOG`에
  mock 태그 부여(`missing_part`를 `subtle_misaligned`로 지정) -- 실제 이미지 파이프라인
  연동 시 팀원의 실제 분류로 교체 예정.
- `_rank`/`_pearson`/`_spearman_rho` 헬퍼 추가 -- scipy 없이 순수 파이썬으로 Spearman rho
  구현 (프로젝트가 표준 라이브러리만 쓰는 기존 방침 유지).
- **검증**: 손으로 만든 fake episode로 (a) `_rank`/`_spearman_rho`가 완전 단조 데이터에서
  ±1.0을 정확히 내는지, 상수 입력엔 `None`을 내는지, (b) `severity_calibration`이 단일
  인용 턴만 세는지(다중 인용 턴 제외 확인), 지어낸 결함 인용을 제외하는지, (c)
  `subtle_misaligned_gap`이 quadrant 없는 아이템을 양쪽 그룹에서 완전히 제외하는지, gap
  숫자(0.75-0.25=0.5)가 정확한지 전부 확인. 규칙기반 dummy agent(인용을 아예 안 함)로 50
  episode 배치도 돌려서 엣지케이스 처리 확인: `severity_calibration_rho=None, n=0`,
  `subtle_misaligned_gap=None, n_caught=0, n_missed=18` -- 인용 없을 때 두 지표 다 의도대로
  `None`/`0`으로 빠짐 (기존 citation_precision 등과 동일한 "정의 안 되면 None" 관례 유지).

## 2026-07-23 (계속) — VLM 배관 스모크테스트 (CraigslistBargain 실데이터)

팀원의 결함-합성 이미지가 아직 없어서, 그 대신 CraigslistBargain 실제 데이터로 "이미지+forced
tool call이 같이 동작하는가"만 먼저 검증. 데이터 출처 조사 경위: 원본 scraper JSON의
`image_urls`(images.craigslist.org CDN)는 30개 샘플 전부 404 -- 2018년 스크래핑이라 만료됨.
HF `stanfordnlp/craigslist_bargains`도 확인했으나 그쪽 `Images` 필드는 로컬 상대경로일 뿐 URL이
아님. 최종적으로 cocoa repo README가 링크한 Codalab 아카이브(bundle
`0xb93730d80e1c4d4cb4c6bf7c9ebef12f`, "images accompanying the original Craigslist posts")에서
`{category}/{post_id}_0.jpg` 경로로 실제 이미지를 받을 수 있음을 curl로 확인.

- **`llm_agent.py` 이미지 지원 추가**: `_has_real_image`(item.image_ref가 mock placeholder가
  아니라 실제 http(s) URL인지)와 `_user_content`(참이면 OpenAI 비전 API의 text+image_url
  컨텐츠 블록 리스트로, 아니면 기존처럼 평범한 문자열로) 추가. 기존 mock 경로(placeholder
  이미지) 동작은 안 바뀜 -- 조건부 분기라 하위호환.
- **`benchmark/craigslist_smoke.py`** 신규: 실제 furniture/electronics 아이템 2개(텍스트는
  cocoa scraper JSON에서 그대로, 이미지는 위 Codalab 경로)로 `Item`을 구성해 실제 LLM agent
  vs Candid counterpart 협상 1건씩 실행. 결과: 둘 다 4턴 만에 합리적인 가격에 합의, 크래시 없음.
- **`benchmark/craigslist_vision_check.py`** 신규: 협상 프레이밍은 시각적 디테일을 굳이
  언급 안 할 수 있어 증거가 약하다고 판단 -- 협상과 분리해서 "이 사진에 뭐가 보이냐"고 직접
  질문하는 최소 스크립트로 재검증. 결과: 두 응답 다 **텍스트 설명에 없는** 디테일을 정확히
  묘사함 (테이블: "야외 잔디밭 위에 놓여있음" -- 텍스트엔 야외 얘기 전혀 없음; 스피커: "골판지
  박스 2개, 파란/흰색 인쇄, 동심원 디자인, 마모/구김" -- 텍스트는 "new in boxes"라고만 함).
  이걸로 GPT-4o가 이미지를 실제로 처리한다는 것을 확실하게 검증 -- **VLM 배관 스모크테스트
  완료**. (Defect ground truth가 있는 real grounding 검증은 팀원 이미지 도착 후 별도.)
- **환경 세팅**: 이 세션엔 레포 루트 `.venv`/실제 키 든 `.env`가 없어서(placeholder만 있었음)
  `benchmark/.venv` + `benchmark/requirements.txt`(openai/python-dotenv만, freeze로 고정)를
  새로 만듦. `run.sh`의 venv 경로를 `../.venv` -> `.venv`로 수정 (benchmark/ 안에서 상대경로로
  찾도록). 사용자가 루트 `.env`에 실제 키를 채워넣은 후 정상 동작 확인.

## 2026-07-23 (계속) — 로더 + price geometry↔defect 연결 완료

- **`benchmark/data_loader.py`** 신규: `load_items(path)`가 `data_spec.md` 스펙(JSON 배열
  또는 JSONL, 첫 non-whitespace 문자로 자동 판별)을 읽어 `Item`/`Defect` 리스트로 변환.
  `item_id`/`image_ref_original`은 `Item`에 대응 필드가 없어 파싱만 하고 버림 (필요해지면
  나중에 필드 추가 -- 안 쓰는 필드를 미리 안 만든다는 원칙 유지). `data_spec.md`의 예시
  JSON으로 JSON-array/JSONL 두 형식 다 round-trip 검증 완료.
- **`env.py`에 `fair_price(item)` 추가 + `sample_episode` 수정**: 2026-07-11에 남겨뒀던
  "결함이 심각한 아이템인데 ZOPA는 결함과 무관하게 랜덤" Simplification을 해소. ZOPA의
  중심(m)을 더 이상 임의의 uniform 난수가 아니라 `fair_price(item) = listing_price -
  sum(defect.price_impact)`에 앵커링 (TERMS-Bench §3.3 data-grounded extension의
  reference-price 앵커링과 같은 아이디어). z/q **폭** 샘플링(난이도/regime 로직)은 그대로
  유지 -- "중심이 어디냐"만 결함에 연동됨. m이 [p_min,p_max] 밖으로 나가면 유효 구간 끝으로
  clip (결함이 아주 심하거나 price bounds가 좁을 때의 안전장치).
  - **검증**: data_spec.md 예시(listing_price=250, 결함 합=23, fair_price=227)로 같은
    아이템을 `sample_episode`에 넣었을 때 ZOPA 중심이 정확히 227로 나오는 것과, 결함 없는
    동일 아이템(listing_price=250)으로는 중심이 250으로 나와 "결함 클수록 중심이 아래로
    당겨진다"를 확인. 기존 6-family 회귀테스트(300 episode)와 severity_calibration/
    subtle_misaligned_gap 손으로 만든 테스트 전부 재실행해서 안 깨진 것 확인 (숫자는
    RNG 소비 순서가 바뀌어 살짝 달라졌지만 -- episode_item 생성이 더 앞으로 옮겨짐 --
    구조적 불변량(AGR-=0, CritViol%=0, cue 붕괴 패턴)은 그대로 유지).

## 2026-07-25 — quadrant를 사후 라벨에서 실시간 커널 파라미터로 승격

**발견된 설계 갭**: `Defect.quadrant`(가시성×정합성 4분면)를 만들어놓고도, 실제로는
`subtle_misaligned_gap`같은 **사후 분석 metric**에서만 썼음. TERMS-Bench가 economic
reactivity × cue reliability 매트릭스를 counterpart FAMILIES의 **실시간 커널 파라미터**
(`_lookup_by_stance`로 매 턴 조회)로 쓰는 것과 대비됨 -- 우리 evidence 축은 매트릭스를
만들어놓고 시뮬레이터가 그걸 무시하고 있었던 셈. (지적 계기: "TERMS-Bench는 매트릭스로
counterpart family를 만드는데, 우리는 매트릭스를 사후 분석에만 쓰는 게 맞냐"는 질문.)

**수정**: `kernel.py`의 `EVIDENCE_BONUS`(단일 상수) -> `QUADRANT_BONUS`(quadrant별 lookup
dict) + `_DEFAULT_BONUS`(quadrant=None 폴백, 기존 2.0 유지)로 교체. `evidence_term_for`가
인용된 결함의 quadrant를 조회해서 그에 맞는 크기의 보정항을 `accept_prob`에 반영하도록
수정 -- `_lookup_by_stance`와 같은 "카테고리 -> 파라미터" 패턴.

값(임의, 프로토타입 단계 -- EVIDENCE_BONUS=2.0이 그랬던 것처럼 튜닝 필요, 방향성만 의도적):
- `obvious_aligned=1.0`: 누구나 찾을 수 있는 결함 -- 기본 보너스
- `subtle_aligned=3.0`: 찾기 어려운데 실제로 심각 -- detection 성공 보상
- `obvious_misaligned=0.5`: 눈에 띄지만 실제 영향 적음 -- 과잉반응을 경제적으로 보상하면
  안 되므로 최소 보너스
- `subtle_misaligned=4.0`: 가장 어려운 사분면(detection+grounding 둘 다 성공) -- 최대 보너스

**다중 인용 처리**: 한 턴에 실제 결함을 여러 개 인용하면 그 중 **최댓값** 보너스를 씀.
`severity_calibration`이 다중 인용 턴을 통째로 제외하는 것과 다른 선택인데, 이유는 성격이
다르기 때문 -- `severity_calibration`은 사후 통계라 애매한 데이터를 그냥 버릴 수 있지만,
`evidence_term_for`는 매 턴 `accept_prob` 계산에 실제로 쓰이는 **실시간 값이라 반드시
뭔가를 반환해야 함**.

**검증**: 4개 quadrant 각각 인용했을 때 정확히 해당 보너스가 나오는지, quadrant=None(태깅
안 된 결함)이 `_DEFAULT_BONUS`로 폴백하는지, 다중 인용시 최댓값이 나오는지, role별 부호
반전(BUYER +/SELLER -)이 여전히 맞는지, 인용 없음/할루시네이션만 있을 때 0.0인지 전부
손으로 만든 케이스로 확인. 기존 6-family 회귀테스트 + severity_calibration/
subtle_misaligned_gap 테스트도 재실행해서 안 깨짐 확인 (규칙기반 dummy agent는
`cited_defect_ids`를 안 채우므로 `evidence_term_for`가 항상 0.0을 반환해 원래부터 이
경로와 무관 -- 그래서 숫자가 이전과 완전히 동일하게 나온 것도 기대한 대로).

## 2026-07-25 (계속) — stance_prior_for 연결 누락 버그 수정

**발견**: `kernel.py`의 `stance_prior_for(family_name)`(ADVERSARIAL의 aggressive-skewed
확률 0.05/0.15/0.80 포함)이 정의만 되어 있고 실제로 호출하는 코드가 어디에도 없었음
(`grep`으로 확인). `run_negotiation.py`가 `sample_episode(rng)`를 stance 관련 인자 없이
호출해서, `env.py`의 기본값(균등 1/3씩, `rng.choice`)이 항상 쓰이고 있었음 -- **어떤
family를 골라도 stance가 항상 균등하게 뽑혀서, Adversarial family의 핵심 특성 하나가
시뮬레이터에 반영이 안 되던 실제 버그.**

**수정**:
- `env.py`의 `sample_episode`에 `stance_weights: tuple[float,float,float] | None = None`
  파라미터 추가. `stance_B = rng.choice(stance_prior)` -> `rng.choices(stance_prior,
  weights=stance_weights, k=1)[0]`로 교체 -- 기존 `regime_weights`와 완전히 같은 패턴
  (`weights=None`이면 균등, `random.choices`의 기본 동작 그대로 활용).
- `run_negotiation.py`가 `stance_prior_for(args.family)`를 한 번 계산해서
  `sample_episode(rng, stance_weights=stance_weights)`로 매 episode에 전달하도록 연결.

**검증**: `stance_prior_for("Adversarial")`의 가중치로 3000 episode 샘플링 -> aggressive
비율 80.3%(기대 80%와 일치, 이전엔 33%였을 것). Candid는 여전히 균등(각 ~33%) 유지.
`stance_weights` 인자 없이 호출해도(하위호환) 정상 동작. 기존 6-family 회귀테스트 +
severity_calibration/subtle_misaligned_gap/quadrant-bonus 테스트 전부 재실행해서 안
깨짐 확인.

## 2026-07-25 (계속) — citing_dummy_agent.py 추가: evidence/quadrant 기계 전체 시연

**동기**: 실제 이미지+VLM이 없는 지금, citation 관련 코드(`evidence_term_for`의 quadrant
lookup, `subtle_misaligned_gap`)가 진짜 LLM agent로는 거의 발동을 안 함 -- 이유는 (1)
이미지가 없어 agent가 결함을 볼 수 없고 (2) 있어도 결함의 정확한 id 문자열을 알 방법이
없음(idea.md 2026-07-15 미해결 문제). "코드가 실제로 맞게 짜였는지"를 이미지 없이
먼저 확인하기 위해, `episode.item.ground_truth_defects`를 직접 읽는(정보 비공개
규약을 의도적으로 깨는) **테스트 전용 치팅 dummy agent**를 추가.

`benchmark/citing_dummy_agent.py` 신규:
- `make_citing_dummy_agent`: 매 OFFER 턴 80% 확률로 인용(그 중 15%는 지어낸 가짜 id),
  실제 결함 인용 시 그 결함의 price_impact에 비례해서 가격을 더 크게 움직임
  (severity_calibration이 양의 상관을 보이도록 의도적 설계).
- 단일 episode transcript: 인용이 실제로 나올 때까지 최대 30회 재시도(운 나쁘게 인용이
  하나도 안 나오는 episode를 뽑을 수 있어서) 후, 매 인용 턴마다 `evidence_term_for` 값을
  직접 계산해서 같이 출력.
- 배치(기본 300 episode): `compute_metrics()` 전체 출력.

**결과 확인 (2026-07-25)**: 단일 episode에서 같은 아이템 안의 서로 다른 quadrant 결함을
인용했을 때 `evidence_term_for`가 실제로 다른 크기로 나옴을 확인 -- subtle_aligned 인용
시 -3.00(QUADRANT_BONUS=3.0, SELLER라 부호반전), obvious_misaligned 인용 시 -0.50
(QUADRANT_BONUS=0.5). 배치 300 episode: `citation_precision=0.828`,
`hallucination_rate=0.170`(설정 hallucinate_prob=0.15 근처), `severity_calibration_rho=
0.427`(n=188, 심각도 비례 양의 상관), `subtle_misaligned_gap=0.189`(n_caught=44,
n_missed=81) -- 전부 None/0이 아닌 실제 값으로 나와 배관 자체는 정상 작동함을 확인.
voice는 이 데모에서 끔(evidence/quadrant 기계 확인이 목적이라 OpenAI 의존 불필요).

## 2026-07-12 — VLM 연동 전 남은 작업 정리
- 워크샵급 contribution 여부를 점검하다가, "팀원이 이미지+결함 데이터를 주면
  끝"이 아니라는 게 드러남 — 그 데이터를 실제로 소비하는 코드가 아직 없음.
  본인(사용자) 담당으로 아래 두 가지를 확정:
  1. **Loader**: `benchmark/data_spec.md` 스펙(JSON/JSONL)을 읽어서
     `benchmark/env.py`의 `Item`/`Defect` dataclass로 변환하는 함수. 아직
     미구현 (`data_spec.md:97`에도 명시).
  2. **Vision 호출 배관**: `benchmark/llm_agent.py`가 지금 완전히 텍스트
     전용(`_build_prompt`가 이미지 필드를 안 만들고, `chat.completions.create`
     호출에도 `image_url` content가 없음, 파일 상단 주석에도 "이미지 없어서
     텍스트만 본다"고 명시)이라, 이걸 멀티모달 호출로 바꿔야 함. 결정할 것:
     이미지 전달 방식(URL vs base64), `gpt-4o`가 이미지+forced tool call을
     동시에 잘 처리하는지, agent에게는 `image_ref`(합성 후)만 주고
     `image_ref_original`은 절대 안 준다는 것.
  - 설계 논의는 아직 시작 안 함 — 팀원 이미지 도착 시점에 맞춰 진행 예정.
  - (참고, 사용자 담당 아님) salience는 여전히 스코프 아웃 상태 유지 — 제대로
    하려면 제3자 블라인드 코더 3인 이상 + ICC가 필요해서 개인이 "짬 내서" 할
    수 있는 작업이 아니라는 점 재확인.
  - **`CritViol%` 감지 누락 (실제 버그, benchmark/ 안에 있음) — 2026-07-13 수정
    완료**: `env.py`의 `run_episode`가 `implementation/env.py`와 동일하게 위반을
    `accept_before_any_offer` 한 종류만 감지하던 문제. `_price_out_of_bounds`/
    `_worse_than_own_reservation` 두 헬퍼를 추가해 Appendix B.3의 (i) 가격범위
    (`p_min<=p_k<=p_max`) 위반과 (ii) IR 위반(자기 reservation보다 나쁜 가격을
    offer/accept)을 각각 `"price_out_of_bounds"`/`"ir_violation"`으로 기록하도록
    `run_episode`에 추가.
    - **처리 방식**: `accept_before_any_offer`와 달리 즉시 DISAGREEMENT로 끊지
      않고 `violations`에 기록만 하고 협상을 계속 진행시킴 — "결과가 정의
      불가능"한 경우(받아들일 대상 자체가 없음)와 달리 가격범위/IR 위반은
      "이상하지만 결과는 정의 가능한 제안"이라 자연스럽게 흘러가게 둠 (LLM이
      이상한 값을 내도 벤치마크가 죽지 않아야 한다는 기존 철학과 동일).
    - **적용 범위**: agent/counterpart 양쪽 턴 모두에 적용(대칭적인 일반
      유틸리티로 작성) — `kernel.py`가 counterpart의 offer/accept를 이미
      [p_min,p_max]/IR로 클리핑해서 만들기 때문에 counterpart 쪽에서 실제로
      걸릴 일은 구조적으로 없지만, 나중에 kernel이 바뀌어도 안전하게 잡히도록
      일반화해둠. 논문 Appendix B.3 문구의 `r_A`는 A/B를 일반화한 표기로 해석.
    - **검증**: (a) IR 지키는 정상 agent + `Candid` 커널로 300 episode → 위반
      0건(오탐 없음), (b) 범위 밖 offer를 내는 dummy agent + 무조건 accept하는
      dummy counterpart 조합으로 `price_out_of_bounds`/`ir_violation` 둘 다
      정상적으로 기록되는 것 확인.

## 2026-07-26 — 매핑 로버스트니스 실험 설계 논의 + env.py 인프라 준비

**배경**: 교수님 pulse.pptx 슬라이드3 — "결함→가치 매핑"(예: scratch/screen_crack/
missing_part별 listing_price 대비 퍼센트)을 보수적/중간/공격적/비선형 등 여러 버전
만들어서, 각 매핑으로 전체 벤치마크(여러 LLM agent)를 돌려 agent 순위표를 뽑고,
순위표끼리 Spearman's ρ로 얼마나 일치하는지 재는 로버스트니스 체크 절차를 논의.

**설계 갈림길과 결정**: `price_impact`는 채점(severity_calibration)뿐 아니라
`fair_price(item)`을 통해 ZOPA 중심 자체도 결정한다 (env.py 2026-07-23 결정). 그래서
"매핑을 바꾼다"가 (a) 채점 기준만 바뀌는 것인지 (b) 협상 환경(ZOPA) 자체가 같이
바뀌는 것인지 갈림길이 생김. **(b)로 결정** — 슬라이드3 문구 "② 각 매핑으로 전체
벤치마크를 돌린다"가 "전체"라고 명시한 것을 근거로, 환경 재생성부터 다시 하는 게
의도에 맞다고 판단. 이게 confound(매핑 때문에 난이도 자체가 달라짐)를 만들 수
있다는 우려에는, **매핑 간 RNG seed를 고정**해서 대응하기로 함 — `sample_item`이
`rng.sample(_DEFECT_IDENTITIES, ...)`으로 결함 kind를 뽑는 게 리스트 개수/순서에만
의존하고 각 원소의 가치 크기엔 의존하지 않으므로, 정체성 리스트를 매핑 간 고정해두면
같은 seed에서 "어떤 결함이 뽑히는지"는 항상 동일, price_impact 숫자만 매핑 따라
달라짐 -- 난이도 변화를 결함 가치 변화 하나로만 좁히는 통제.

**중요한 스코프 판단**: 이 로버스트니스 실험의 진짜 목적(agent 순위가 매핑 선택에
안 흔들리는지)은 **실 이미지가 있어야만 의미가 생김** — 지금 mock 데이터(placeholder
이미지)로 실제 LLM/VLM agent를 돌리면 `cited_defect_ids`가 항상 비어있어(이미지가
없어 볼 게 없음) `evidence_term_for`가 항상 0, citation 관련 metric도 전부 정의
불가 상태라 "시각 증거 활용 순위"라는 것 자체가 발생하지 않음. 그래서 지금은 **실험
본체가 아니라 인프라만** 미리 준비하기로 함 (citing_dummy_agent.py로 실 이미지 없이
배관만 검증했던 2026-07-25 패턴과 동일).

**env.py 구현**: `_DEFECT_CATALOG`(kind/description/price_impact/quadrant 한 튜플에
혼재)를 `_DEFECT_IDENTITIES`(kind/description/quadrant, 매핑과 무관) +
`SEVERITY_MAPPINGS`(매핑 이름 -> kind -> listing_price 대비 퍼센트, 4개 프리셋)로
분리. `sample_item(rng, listing_price, num_defects=2, mapping=None)` — `mapping=None`이면
`SEVERITY_MAPPINGS["B_mid"]` 기본값, `price_impact = mapping[kind] * listing_price`로 계산.

- **3종으로 범위 축소 결정**: 원래 카탈로그는 5종(scratch/tear/stain/dent/missing_part)
  이었으나 교수님 슬라이드3 매핑 표엔 3종(스크래치/화면균열/부품누락)의 값만 있어서,
  나머지 2종에 매핑값을 임의로 유추해 채우기보다 표 그대로 3종(scratch/screen_crack/
  missing_part)으로 좁히기로 함 (2026-07-26 결정).
- quadrant 배치: scratch=obvious_aligned(기존 유지), screen_crack=obvious_aligned
  (슬라이드2의 Obvious-Aligned 예시 "화면이 크게 깨졌고, 그만큼 값이 떨어지는 게 맞음"을
  그대로 반영), missing_part=subtle_misaligned(기존 유지, 이 벤치마크의 "킬러 사분면").
- 매핑 값: 슬라이드3 표 그대로 A_conservative(-5%/-15%/-20%), B_mid(-10%/-25%/-35%),
  C_aggressive(-15%/-40%/-50%), D_nonlinear(-8%/-30%/-45%).

**검증**: seed=42로 4개 매핑 전부 돌려서 뽑히는 결함(`missing_part_0`, `scratch_1`)이
매핑 무관하게 항상 동일하고 price_impact/`fair_price`만 이동(150→110→70→94)하는 것
확인 -- 의도한 "confound 없는 통제" 정확히 재현. `mapping=None`이 `B_mid`와 동일 결과
내는 것도 확인. `citing_dummy_agent.py` 300 episode 회귀(Adversarial family) 재실행 --
크래시 없음, citation/severity_calibration/subtle_misaligned_gap 전부 정상 값 -- 5종
→3종 축소가 quadrant/evidence 기계를 깨지 않은 것 확인.

**미구현 상태로 남은 것**: 매핑 로버스트니스 실험의 실제 오케스트레이션(여러 매핑 x
여러 agent 배치 실행 x agent 순위 산출 x 순위표 간 Spearman's ρ 계산 x 리포트)은 아직
없음 -- 오늘은 그 실험이 갈아끼울 "매핑" 인프라만 준비. 또한 여러 LLM provider(GPT
외 Claude/Gemini/Qwen 등)를 붙이는 배관도 `llm_agent.py`가 OpenAI 전용이라 아직
없음 -- 슬라이드3(매핑 로버스트니스) + 슬라이드6(visual 유/무 gap) 두 실험 모두의
공통 전제조건으로 남아있음.

## 2026-07-26 (계속) — CLAUDE.md / decisions_log.md 분리

`benchmark/CLAUDE.md`가 날짜별 세션 로그가 계속 누적되면서 500줄 넘게 길어짐 --
매번 세션 시작할 때 읽는 "운영 원칙 + 지금 상태" 문서와 "왜 이렇게 결정했는지의
전체 경위/검증 기록" 아카이브 두 성격이 섞여 있던 게 원인. 이 파일(decisions_log.md)로
날짜별 로그 전체(구현 범위 체크리스트의 상세 근거 포함)를 그대로 옮기고, CLAUDE.md는
"지금 상태(코드가 이미 원본이라 코드 읽으면 재구성 가능하므로 생략) + 확정된
스코프아웃(코드의 부재만 봐서는 알 수 없는 정보라 유지) + 다음 작업"만 남기기로 함.
"확정된 스코프아웃"을 남기기로 한 이유: `현재 상태`는 코드를 읽으면 재구성되지만,
"BE_type을 왜 안 만들었는지"(교수님 코멘트 때문 vs 그냥 아직 안 만듦)는 코드의 부재
자체로는 구분이 안 돼서, 나중에 이미 끝난 논의를 다시 여는 걸 막기 위해 유지.

마감도 이 시점에 2026-07-11(이미 지남, 프로토타입 마감이었음) → 2026-08-05로 갱신.

## 2026-07-26 (계속) — 교수님 멘토링 반영: 벤치마크 프레이밍/사분면 utilization/마감 시각

사용자가 교수님 멘토링 원본 메모를 공유해서, 지금까지 pulse.pptx 슬라이드 기준으로
정리한 내용과 대조·보강함.

**벤치마크 프레이밍 재확인**: "점수를 보는 게 아니라 모델간 흔들리지 않고 빵꾸나는
지점을 평가하는 벤치마크" — 순위표(누가 1등이냐) 자체가 목적이 아니라 **조건에
따라 특정 모델 성능이 무너지는 지점을 진단**하는 게 핵심 가치라는 재확인. 이게
CLAUDE.md 시나리오 절에 "측정 프레이밍" 항목으로 명시적으로 추가됨(2026-07-23
BE_type 드롭 결정 때도 나왔던 "우리는 정보가 주는 영향을 본다"는 방향성과 일관).

**매핑 로버스트니스 vs 사분면 간 일관성 — 시행착오 경위**: 사용자 메모의 "4가지
지표에서 스피어만 구하기"를 처음엔 "4개의 서로 다른 metric"으로 오해했다가,
"4가지 지표 = 2x2 quadrant"라는 정정을 듣고 이번엔 "매핑 로버스트니스(슬라이드3)와
사분면 간 일관성이 서로 다른 두 개의 별도 실험"이라고 잘못 재구성함(사분면별로
순위를 매기고 **사분면 쌍끼리** rho를 잰다고 이해). 사용자가 슬라이드3 원문(①~④)을
다시 붙여주면서 "이거 다른 두 개 축은 아닌 것 같다"고 정정 — 실제로 슬라이드3
원문 어디에도 "사분면"이라는 말이 없다는 게 근거.

**최종 정리(정정된 이해)**: 매핑 로버스트니스(①~④)는 **하나의 절차**고, "사분면"은
별도 실험이 아니라 **그 절차를 어느 단위로 적용하냐**의 문제. ②번 단계("매핑 A로
모든 에이전트 평가 → 순위 하나")를 전체 episode 합산으로 순위 하나만 뽑을 수도
있고, 사분면별로 쪼개서 순위 4개(사분면당 하나)로 뽑을 수도 있음 — 후자가 "빵꾸나는
지점" 진단을 더 세밀하게 잡아냄: 전체 합산 ρ는 높게 나와도 특정 사분면(예:
subtle_misaligned)만 ρ가 낮으면, 그 사분면 조건에서만 매핑 선택에 순위가 민감하다는
뜻이라 정확히 어디가 "구멍"인지 짚어낼 수 있음. 두 번 잘못 재구성했다가 슬라이드
원문 재대조로 바로잡은 경위 자체를 남겨둠 — 다음에 이 실험을 실제 코드로 옮길 때
"사분면별 순위 산출"이 매핑 로버스트니스 오케스트레이션의 **한 옵션**이지 별도
파이프라인이 아니라는 걸 헷갈리지 않기 위함.

**과잉반응 방지(Obvious-Misaligned) — 슬라이드 원문 확인**: 사용자가 처음엔 "교수님이
과잉반응 방지라고 딱히 말한 적 없다"고 했으나, pulse_extract.txt 슬라이드2 원문에
정확히 그 문장이 있음을 재확인("에이전트가 겉모습에 과잉반응해서 과도하게 깎으려
드는지 시험. grounding의 반대 방향 실패를 잡음") — 슬라이드 텍스트가 출처. 사용자가
이 프레이밍을 반영하기로 최종 확인. 결론: `subtle_misaligned_gap`을 4개 사분면
전부로 기계적으로 복붙 확장하면 안 되고, **obvious_misaligned는 "인용 시 가격을
실제 price_impact보다 과하게 깎는가"를 재는 severity_calibration류 지표가 더 맞는
모양** — 나머지 3개 사분면도 각자 실패 모드에 맞는 지표 모양을 따로 설계해야 함
(구체 설계는 미착수, CLAUDE.md "다음 작업" 참고).

**신규 액션 아이템**: 평가 대상 agent 리스트를 연구실 프라이싱 사이트에 제출 필요
(교수님 멘토링, 구체 절차/사이트 정보는 아직 없음 — 사용자가 URL 등 알려주면
reference 메모리에 남길 것).

**마감 시각 확정**: 2026-08-05 14:00 (기존 날짜만 있던 것에 시각 추가).

## 2026-07-26 (계속) — fixed-concession baseline 구현 완료

"다음 작업" 우선순위(fixed-concession baseline → multi-provider agent 배관 → quadrant 기반
utilization 지표) 합의 후 1번부터 착수.

**스펙 확인**: `terms-bench.txt:456` "conceding 1%, 10%, and 30% of the remaining distance to
reservation per Offer" — 이 공식이 `kernel.py`의 기존 `counter_offer(prev_own_price, r_B,
role_B, lam, noise_std, rng)`와 완전히 동일(`candidate = prev_own_price - lam*(prev_own_price
- r_B)`), 새 수식 없이 `lam=rate`(0.01/0.10/0.30)만 고정해서 재사용.

**accept 규칙 — Simplification**: 논문 어디에도 FC baseline 전용 accept 알고리즘 박스가
없음(H.1.1~H.1.7, `wrapper` 언급 2곳까지 전부 재검색해서 확인). 대신 H.1.5(Agent Interface,
`terms-bench.txt:3421-3424`)가 LLM JSON 파싱 실패 시 쓰는 deterministic fallback을 명시:
"accept if the standing counterpart offer is weakly preferred to walking away" — 논문이
보여주는 유일한 결정론적 accept 규칙이라 FC baseline에도 동일 적용
(`favorability(offer) >= 0`이면 즉시 수락). 논문이 FC baseline에 이 규칙을 쓴다고 명시한
건 아니므로 정확한 재현이 아니라 최선의 추정 — 파일 docstring에도 명시.

**opening offer의 urgency/stance — Simplification**: `opening_offer`는 counterpart의
urgency/stance로 첫 제안 강도를 조절하는데 agent 쪽엔 그런 필드가 없음(Episode.t_B만
있고 t_A 없음). FC baseline은 심리 상태 개념 자체가 없는 기계적 규칙이므로
urgency=0.0/stance="neutral"(=보정 없는 중립값)을 넣어 episode.harshness만으로 첫 제안
강도가 정해지게 함.

**신규 파일**: `benchmark/fixed_concession_agent.py` — `make_fixed_concession_policy(rate)`
+ `FC_RATES = {"FC-1": 0.01, "FC-10": 0.10, "FC-30": 0.30}`. `citing_dummy_agent.py`와
다른 성격: 그쪽은 evidence 배관 검증용 "치팅" dummy였지만, 이건 애초에 evidence를 전혀
안 봄(`cited_defect_ids`는 항상 `None`) — 정보를 아예 안 쓰는 진짜 바닥선. REJECT는 절대
안 함(`env.py`의 REJECT는 즉시 DISAGREEMENT로 끝나는 종단 액션이라, "포기 판단" 자체가
이 baseline 설계 범위 밖 — K턴 안에 못 정하면 round-limit DISAGREEMENT로 자연스럽게 끝남).

**run_negotiation.py 연결**: `--fc-rate {FC-1,FC-10,FC-30}` 플래그 추가 — 주어지면
`--model` 대신 fixed-concession policy를 agent로 씀. OpenAI API 키 없이도 배관 테스트
가능(이 baseline 자체가 LLM을 아예 안 쓰므로).

**검증**: (a) 단일 episode(seed=7, FC-1, BUYER)에서 offer가 $25.83→26.04→26.24로 움직이는
걸 손으로 검산 -- `26.04 = 25.83 + 0.01*(46.39-25.83)` 정확히 일치, 매 턴 reservation까지
남은 거리의 1%씩 감쇠하는 궤적 확인. (b) 300 episode 배치(Candid family, seed=42)를
FC-1/FC-10/FC-30 각각 실행 -- `SE+`(0.355→0.400→0.473)와 `AGR+`(0.811→0.877→0.966)가
rate가 커질수록 단조 증가, 논문 Table 2의 정성적 패턴(Fixed 30% > Fixed 10% > Fixed 1%)과
방향 일치. `AGR-`/`CritViol%` 전부 0.000(IR 위반 없음, 다른 IR 지키는 dummy agent들과
동일 패턴), citation 관련 지표 전부 `None`/0(설계대로 -- 정보를 아예 안 씀).

## 2026-07-26 (계속) — multi-provider agent 배관: OpenRouter로 아키텍처 확정

우선순위 2번(multi-provider agent 배관) 착수. `llm_agent.py`가 `from openai import
OpenAI`로 완전히 결합돼 있어 Claude/Gemini/Qwen을 못 붙이던 문제.

**핵심 발견**: `terms-bench.txt:457` "LLMs are called via **OpenRouter**" -- 논문 자체가
이 문제를 이미 풀어놓음. OpenRouter는 여러 provider 모델을 OpenAI 호환 API 형식
(`chat.completions.create`, `tools`/`tool_choice`, `image_url` 컨텐츠 블록)으로 통일해서
프록시하는 게이트웨이라, 각 provider SDK를 따로 배울 필요 없이 **기존 OpenAI SDK
클라이언트의 `base_url`만 바꾸면** 됨 -- `_NEGOTIATE_TOOL`/`_build_prompt`/
`_user_content`/응답 파싱 전부 그대로 재사용 가능.

**OpenRouter 결제 구조 확인 (웹서치, 2026-07-26)**: 계정 하나/API 키 하나/크레딧 잔액
하나로 OpenRouter에 올라온 모든 provider 모델 사용 가능(각 provider별 개별 계정 불필요).
선불 크레딧제(달러 단위, 실시간 차감). 토큰 가격 자체엔 마크업 없음(provider 가격 그대로
통과) but **크레딧 충전 시 5.5% 수수료** (예: $10 크레딧 사려면 카드에서 $10.55 청구).
오픈소스 모델 20개+는 영구 무료, $10+ 충전하면 무료 모델 일일 한도 1,000회로 상승. 출처:
layer3labs.io/guides/openrouter-pricing, ofox.ai/blog/openrouter-pricing-hidden-markup-
breakdown-2026, truefoundry.com/blog/openrouter-pricing.

**결정**: OpenRouter로 통합. `provider="openai"`(기본값, 기존 동작 100% 유지) /
`provider="openrouter"`(새 경로) 두 가지를 `make_llm_agent_policy`에 추가. OpenRouter
쪽 모델 문자열은 `"anthropic/claude-opus-4.6"` 같은 "provider/model" 네이밍 -- TERMS-Bench
로스터의 최신 모델명이 지금 실제로 OpenRouter에 그 이름 그대로 있는지는 별개 확인 필요
(아키텍처 결정과는 무관, 나중에 실제 연동 시 모델 slug 단위로 검증).

**코드 변경**: `llm_agent.py`의 `make_llm_agent_policy(model, provider="openai")`에
`provider` 분기 추가 -- `"openai"`는 기존 `OpenAI()` 그대로, `"openrouter"`는
`OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])`,
그 외 값은 `ValueError`.

**검증 (API 키 없이, 과금 없는 구조 검증만)**: (a) `provider="openai"`(기본값/명시 둘 다)가
에러 없이 클라이언트 생성되는 것 확인 -- 기존 동작 안 깨짐. (b) 잘못된 `provider` 값이
명확한 `ValueError` 내는 것 확인. (c) `OPENROUTER_API_KEY`가 `.env`에 없을 때
`provider="openrouter"`가 `KeyError`로 즉시 명확하게 실패하는 것 확인(조용히 넘어가지
않음). 실제 `chat.completions.create` 호출(과금 발생)은 안 함 -- 실제 Claude/Gemini/Qwen
응답 형식이 이 코드 경로와 완전히 맞물리는지는 키 도착 후 스모크테스트 필요.

**남은 것**: `OPENROUTER_API_KEY` 미발급 -- 연구실 결제로 교수님께 메일 예정
("평가 대상 agent 리스트 제출" 액션아이템과 같은 메일로 처리 가능, CLAUDE.md 참고).
키 도착 후: (1) 실제 호출 스모크테스트(tools/image_url이 Claude/Gemini/Qwen에서도
동일하게 동작하는지), (2) TERMS-Bench 로스터 모델명의 실제 OpenRouter slug 확인.

## 2026-07-26 (계속) — quadrant 기반 utilization 지표: 전체 설계 + 카탈로그 갭 발견/수정

우선순위 3번(quadrant 기반 utilization 지표) 착수. "사분면마다 새 공식"이 아니라
"전체 그림"부터 맞추는 benchmark/CLAUDE.md 작업 방식대로 진행.

**핵심 통찰**: 재려는 능력은 사실 두 개뿐 -- **Detection**(결함을 알아챘는가)과
**Calibration**(알아챘다면 크기를 진짜 가치에 맞게 반영했는가). 2x2의 두 축은 이 둘의
난이도를 각각 조절한다: **가시성(Obvious/Subtle) -> Detection 난이도**,
**정합성(Aligned/Misaligned) -> Calibration 난이도**(방향이 아니라 난이도! -- 처음엔
"정합성 축이 concession 방향을 정한다"고 잘못 정리했다가, "Aligned 셋 다 방향이 같은데
무슨 소리냐"는 지적으로 정정: 진짜 갈리는 건 방향이 아니라 "착시가 있냐 없냐"(Aligned=
보이는 대로가 진실이라 착시 없음=쉬움, Misaligned=보이는 것과 진실이 어긋나 착시
있음=어려움, 방향만 둘이 반대). "과잉반응 방지"(obvious_misaligned)는 별개의 세 번째
능력이 아니라 Calibration을 억제 방향에서 본 것.

**사분면별 정리**:

| 사분면 | Detection | Calibration | 실제 시험 |
|---|---|---|---|
| Obvious-Aligned | 쉬움 | 쉬움(착시 없음) | 기준선/sanity check |
| Subtle-Aligned | 어려움 | 쉬움(찾으면 착시 없음) | 순수 Detection |
| Obvious-Misaligned | 쉬움 | 어려움(과대평가 착시) | 순수 Calibration |
| Subtle-Misaligned | 어려움 | 어려움(과소평가 착시) | Detection+Calibration 동시 -- 킬러 조건 |

**설계**: 사분면마다 새 공식을 만들지 않고 기존 두 metric을 사분면으로 슬라이스:
- `quadrant_detection_rate(records, quadrant)` -- 기존 `citation_coverage`(전체 결함
  대상 recall)를 quadrant==Q인 결함만으로 좁힌 버전.
- `quadrant_calibration(records, quadrant)` -- 기존 `severity_calibration`(전체 결함
  pooled Spearman ρ)을 인용된 결함의 quadrant==Q인 단일-인용 턴만으로 좁힌 버전.
- `subtle_misaligned_gap`은 그대로 유지(outcome 레벨이라 다른 사분면엔 기계적으로 안
  넓힘 -- misaligned 쪽은 "잡았다고 항상 좋은 게 아님"이라 gap 해석이 안 맞음).

4개 사분면 x 2개 축 = 8개 + `subtle_misaligned_gap` 보너스 1개 = 9개 지표, 새 공식
없이 기존 `citation_coverage`/`severity_calibration` 로직 재사용만으로 구성.

**매핑과의 연결(사용자 지적, 중요)**: `quadrant_calibration`이 비교하는 "진짜 가치"
(`Defect.price_impact`)가 곧 `SEVERITY_MAPPINGS`에서 고른 퍼센트 값이라는 걸 확인 --
즉 calibration의 "정답"이 우리가 고른 매핑에 의해 정의됨. 함의 두 가지: ①매핑
로버스트니스 체크(사분면별 ρ)가 `quadrant_calibration`/`severity_calibration`에도
그대로 적용돼야 함(전에는 fair_price/ZOPA에만 해당한다고 생각했는데 calibration류
전체로 확장됨). ②"잘 교정됨"이 절대적 진실이 아니라 우리가 고른 매핑 대비 상대적
진실이라는 한계를 상속받음(price_impact 자체가 프로토타입 임의값이라는 기존 한계와
동일선상). 코드 변경은 불필요(Defect.price_impact는 생성 시점에 이미 확정된 값이라
metric이 매핑을 몰라도 됨) -- 해석상 주의사항으로 문서화.

**카탈로그 갭 발견 및 수정**: 위 설계를 구체적인 숫자 예시로 만들어보다가, `_DEFECT_
IDENTITIES`(당일 앞서 3종으로 좁혀둔 것)에 obvious_misaligned/subtle_aligned 사분면
결함이 아예 없다는 걸 발견 -- 이 두 사분면의 `quadrant_detection_rate`/
`quadrant_calibration`이 영원히 `None`으로 죽는 문제. `hairline_crack`(subtle_aligned)
/`stain`(obvious_misaligned) 2종을 카탈로그에 추가해 4개 사분면 전부 커버하도록 수정
(2026-07-26, 사용자 결정 "1번으로 가자" -- 슬라이드3에 없는 값이라 방향성만 맞춰
임의로 채움: hairline_crack은 "진짜 심각한" 축이라 scratch~screen_crack 사이 스케일,
stain은 "사소한" 축이라 5종 중 가장 작은 값. 예시는 슬라이드2 원문의 실제 예시
그대로 재사용 -- "모서리 미세 균열"/"쉽게 닦이는 얼룩"). `SEVERITY_MAPPINGS` 4개
프리셋 전부에 두 kind의 값 추가.

**검증**: (a) `_DEFECT_IDENTITIES`의 quadrant 집합이 정확히 4개(전 사분면 커버) 확인.
(b) seed=42로 5개 매핑 전부 돌려서 뽑히는 결함(`scratch_0`/`missing_part_1`)이 매핑
무관하게 동일 유지되는 불변량 재확인(5종으로 늘어나도 안 깨짐). (c) `citing_dummy_
agent.py` 300 episode 회귀(Adversarial family) 재실행 -- 크래시 없음, 모든 지표
정상 값(`severity_calibration_rho=0.441` 등), transcript에서 `scratch`(obvious_aligned)
인용 시 `evidence_term_for=-1.00`(QUADRANT_BONUS=1.0, SELLER 부호반전) 정확히 확인.

**아직 미정**: 8개 지표를 4개 사분면 다 균일하게 낼지, 아니면 "실제 시험" 열에서 고른
것만 선택적으로 낼지 -- 다음 논의 대상. `quadrant_detection_rate`/`quadrant_calibration`
함수 자체도 아직 미구현(카탈로그 갭 수정까지가 오늘 완료분).

## 2026-07-26 (계속) — quadrant_detection_rate/quadrant_calibration 구현 완료

"8개 다 균일하게 낼지" 질문에 사용자가 "8개 다 내자"로 결정 (선택적으로 거르면 우리가
미리 정한 가설에 맞춰 데이터를 필터링하는 셈이 되어, "빵꾸나는 지점 진단" 철학과
어긋난다는 근거).

**구현 (`metrics.py`)**:
- `_recall(records, quadrant=None)`: `citation_coverage`가 하던 계산(recall 평균 + n)을
  quadrant로 슬라이스 가능하게 일반화. `citation_coverage`는 `_recall(records)[0]`
  호출로 리팩터(기존 시그니처/동작 100% 유지), `quadrant_detection_rate(records,
  quadrant)`는 `_recall(records, quadrant=quadrant)` 그대로 노출.
- `_severity_calibration_pairs(records, quadrant=None)`: `severity_calibration`이 하던
  (price_impact, concession) 쌍 수집을 quadrant로 슬라이스 가능하게 일반화. 같은 방식으로
  `severity_calibration`/`quadrant_calibration(records, quadrant)` 둘 다 이 헬퍼를 공유.
  (두 리팩터 다 "새 함수 추가할 때 기존 함수와 로직이 완전히 겹치면 공유 헬퍼로 추출" --
  단순 3줄 반복이 아니라 함수 전체가 겹쳐서 추출이 타당했던 경우.)
- `_QUADRANTS = ("obvious_aligned", "subtle_aligned", "obvious_misaligned",
  "subtle_misaligned")` 상수 추가, `compute_metrics`가 이 4개를 순회하며
  `detection_rate_{q}`/`detection_rate_{q}_n`/`calibration_rho_{q}`/
  `calibration_rho_{q}_n` 16개 키를 추가로 노출 (기존 severity_calibration_rho/_n,
  subtle_misaligned_gap 트리오는 그대로 유지).
- `severity_calibration` docstring에 매핑 의존성 경고 추가(위 "매핑과의 연결" 절 내용을
  코드 주석으로도 남김 -- decisions_log뿐 아니라 코드 읽을 때도 바로 보이게).

**검증 (citing_dummy_agent.py, 500 episodes, Adversarial, seed=11)**: `detection_rate`는
4개 사분면 전부 non-None(0.335~0.405, n=188~353) -- 카탈로그 갭 수정이 실제로
detection 지표를 살렸음을 확인. `calibration_rho`는 obvious_aligned만 값이 나오고
(0.354) 나머지 3개는 `None`(n은 62~71로 충분한데도).

**발견 -- 버그 아님, mock 데이터의 구조적 한계**: `_spearman_rho`는 x(price_impact) 또는
y에 분산이 0이면 정의상 `None`을 반환한다(`_pearson`의 `var_x==0` 체크, 2026-07-23
`severity_calibration` 만들 때 이미 의도적으로 넣은 동작 -- "상수 입력엔 None"). 3개
사분면(subtle_aligned/obvious_misaligned/subtle_misaligned)은 카탈로그에 kind가
**하나뿐**이고, `sample_episode` 기본 호출의 `listing_price=(p_min+p_max)/2`가 모든
episode에서 **상수(50.0)**라서, 그 유일한 kind의 price_impact도 매번 완전히 동일한
상수가 나옴 -- 비교할 대상이 하나뿐이라 상관관계 자체가 수학적으로 정의 불가.
obvious_aligned만 값이 나오는 건 카탈로그에 kind가 2개(scratch/screen_crack, 서로 다른
price_impact)라 우연히 분산이 있어서.

**결정 (2026-07-26, 사용자 판단) -- 손 안 대고 넘어감**: 카탈로그를 더 늘리거나 mock
listing_price를 다양화하는 건 어차피 실 데이터 오면 버려질 mock 인프라를 더 정교하게
다듬는 데 시간을 쓰는 오버엔지니어링이라고 판단. 실 데이터가 오면 (a) 아이템마다
listing_price가 실제로 다양하고 (b) 같은 사분면 태그의 결함도 아이템마다 다른
price_impact를 가지므로(지금처럼 "kind 하나"가 아니라 여러 아이템에 걸친 여러 개별
결함) 분산 문제가 자연히 해소될 것으로 예상 -- 코드는 이미 올바르게 작동 중(분산 0일
때 None을 내는 게 맞는 동작), 손댈 게 없음. **실 데이터 도착 후 확인할 것**: 만약 그때도
특정 사분면의 calibration_rho가 계속 None이면, 그건 코드 문제가 아니라 그 사분면에
해당하는 실 데이터가 충분히 다양하지 않다는 신호이므로 데이터 준비 팀원에게 확인.

## 2026-07-26 (계속) — 6-family를 메인 실험 축에서 제외, Candid 고정으로 번복

사용자 질문: "TERMS-Bench는 상대방 믿음에 대한 협상/다이내믹을 보려고 6-family를
만들었는데, 우리는 quadrant 기반으로 우리만의 '축'을 이미 만들지 않았나 -- 굳이
family를 여러 개 유지할 필요가 있나?"

**2026-07-23 결정 재검토**: 그날 "6-family 유지(구현 예정)"라고 결정하면서 근거로
"협상 자체의 로버스트니스 검증(다양한 counterpart 성향에 agent가 안정적으로
대응하는가)은 여전히 유효한 질문"이라고 남겼었음. 근데 이 근거를 다시 보면 같은 날
내렸던 BE_type 드롭 결정("TERMS-Bench는 상대방 믿음 기반, 우리는 정보가 주는 영향을
본다")과 논리적으로 같은 카테고리에 속함 -- family도 결국 "상대방(counterpart) 쪽
변수를 체계적으로 바꾸는 것"이라, 우리가 보려는 "item/evidence 쪽 변수"와는 다른 축.
그때는 이 연결을 놓치고 family만 예외적으로 살려뒀던 것.

**새로 추가된 근거(2026-07-23 이후 생긴 정보)**: 그 결정 당시엔 quadrant가
`subtle_misaligned_gap`용 사후 라벨일 뿐이었음. 그런데 2026-07-25(quadrant를 실시간
커널 파라미터로 승격, `QUADRANT_BONUS`)와 2026-07-26(quadrant_detection_rate/
quadrant_calibration)을 거치며 quadrant가 지금 정확히 TERMS-Bench에서 family가 하던
역할(변수 하나를 체계적으로 바꿔가며 agent 반응이 어떻게 갈리는지 보는 축)을 차지하게
됨 -- 2026-07-23 시점엔 없었던 근거라 그때는 이 결론에 도달할 수 없었음(번복이 아니라
새 정보에 따른 정당한 재검토).

**실용적 근거**: family를 메인 실험 축으로 계속 끌고 가면 앞으로 할 모든 실험(매핑
로버스트니스/visual 유무/모델 비교)이 family x quadrant x mapping x model 4중 조합으로
폭발 -- 해석도 어려워지고(SE+ 하락이 어려운 family 때문인지 어려운 quadrant 때문인지
confound) OpenRouter 실 API 호출 비용도 family 개수만큼 곱해짐.

**결정**: 시각증거 관련 메인 실험(매핑 로버스트니스/visual 유무/quadrant 검증/모델
비교)은 앞으로 전부 **counterpart family를 Candid로 고정**해서 진행. `run_negotiation.py
--family` 기본값이 이미 `Candid`라 코드 변경 불필요. 6-family 코드(`kernel.py`의
`FAMILIES`/`ECON_PRESETS`)는 삭제하지 않음 -- 이미 구현/검증 완료된 자산이고, 시간이
남으면 "다른 counterpart 성향에서도 결론이 유지되는가" 정도의 선택적 로버스트니스
부록 체크로 나중에 재활용 가능.

## 2026-07-28 — 실 데이터 도착 확인 + `SEVERITY_MAPPINGS` 축 오류 발견/수정

**배경**: `git log`에서 `ae55c11 add data` 커밋으로 실 데이터(`benchmark/data/dataset_v1/
full_run/`, 결함-합성 이미지 324개 + `results.jsonl`)가 이미 도착해 있는 것을 확인.
"데이터 로더 만들어서 10턴 정도 돌려보자"는 요청에 따라 로더를 짜기 전에 실제 레코드를
까봄.

**발견 1 — `data_spec.md` 스펙과 실제 데이터가 다름**: 스펙 문서는 `defect`(단수,
nullable) + `kind` 필드를 명시했는데, 실제 `results.jsonl`은 `defects`(배열, 0~1개) +
`defect_type`/`visibility`/`alignment`/`severity` 4개 필드로 옴 (`quadrant`가 직접 오지
않고 `visibility`+`alignment`로 쪼개져 있고, `severity`는 스펙에 아예 없던 새 필드).
`defect_type` 카탈로그도 실제로는 7종(scratch/dent/rust/crack/stain/tear/missing_part)
으로, `env.py`의 `_DEFECT_IDENTITIES` 5종(scratch/screen_crack/hairline_crack/stain/
missing_part)과도 다름.

**원인 조사**: 데이터 폴더에 같이 있던 `full_run/run_full.py`와 `summary_v1_0726.md`(팀원이
직접 남긴 문서)를 읽어보니, 팀원이 데이터를 생성할 때 **자기 로컬에 별도의 `env.py`**
(`FAMILY_SEVERITY_RULE`, `PRICE_IMPACT_MAPPINGS`, `make_defect()`, `FAMILY_NAMES` 포함)를
만들어 썼고, 이게 우리 레포에 커밋된 `benchmark/env.py`(`_DEFECT_IDENTITIES`/
`SEVERITY_MAPPINGS`)와 독립적으로 갈라져 있었음이 드러남. 두 파일 다 "가시성x정합성
4분면"이라는 같은 pulse.pptx 슬라이드2/3 설계를 각자 구현했는데, **핵심 로직이 갈렸다**:

- 우리 레포(2026-07-26 작성): `price_impact = SEVERITY_MAPPINGS[mapping][kind] *
  listing_price` — 결함 **종류(kind)**가 가격을 결정한다고 봄.
- 팀원 실제 생성 코드(같은 날 작성, `summary_v1_0726.md` 2.2절): `price_impact =
  PRICE_IMPACT_MAPPINGS[mapping][severity] * listing_price`, `severity`는
  `FAMILY_SEVERITY_RULE[(visibility, alignment)]`로 quadrant가 결정. pulse.pptx 원표의
  열(스크래치/화면균열/부품누락)을 "결함 3종"이 아니라 **"경/중/심 예시 인스턴스"**로
  재해석했다고 명시.

**검증 (2026-07-28, 레코드 단위 전수 검사)**: 실 데이터 270개 결함 전부에 대해
`FAMILY_SEVERITY_RULE[(visibility, alignment)]` 예측값과 데이터의 실제 `severity`를
비교 — **불일치 0건**. 즉 severity는 데이터 안에서 quadrant에 의해 100% 결정되는 값이고,
`defect_type`과는 무관함이 실측으로 확인됨 (아이템당 결함은 항상 0개 또는 1개,
`data_spec.md` "아이템 하나당 결함은 하나뿐" 규칙과 일치).

**결정**: 우리 레포 `env.py`가 틀렸던 쪽이므로 팀원의 실제 생성 로직에 맞춘다.
- `SEVERITY_MAPPINGS`(kind 축, 5종) → `PRICE_IMPACT_MAPPINGS`(severity 축: mild/moderate/
  severe)로 교체. **숫자는 안 바뀜** — A_conservative의 `scratch=.05/screen_crack=.15/
  missing_part=.20`이 정확히 `mild=.05/moderate=.15/severe=.20`과 같은 숫자였음. 두 팀이
  같은 pulse.pptx 표를 다른 축으로 읽었을 뿐, 값 자체는 계속 맞았던 것.
- `FAMILY_SEVERITY_RULE`(`(visibility,alignment) -> severity`)과 `quadrant_to_severity()`,
  `price_impact_for(listing_price, severity, mapping)` 헬퍼 신규 추가.
- `Defect`에 `severity: str | None`, `defect_type: str | None` 필드 추가 (둘 다 기본값
  `None`이라 하위호환 — 기존 `Defect(...)` 생성 코드는 안 깨짐). `defect_type`은 지금
  가격 계산에 안 쓰이지만 실 데이터가 갖고 있는 필드라 `item_id`와 같은 취급으로 보관.
- mock 생성기 `sample_item()`도 같은 `price_impact_for()`를 타도록 수정 (기존엔
  `mapping[kind]`를 직접 곱했음) — **이번에 mock/실 데이터 두 경로가 서로 다른 축으로
  갈라졌던 문제가 재발하지 않도록**, 두 경로가 항상 같은 함수를 공유하게 하는 게
  핵심 목적. `_DEFECT_IDENTITIES`가 이미 갖고 있는 quadrant에서
  `quadrant_to_severity()`로 severity를 유도해서 넘김.
- `benchmark/CLAUDE.md`/`data_spec.md`/`metrics.py`의 `SEVERITY_MAPPINGS` 문구 참조도
  `PRICE_IMPACT_MAPPINGS`로 갱신 (문서가 코드와 어긋나지 않도록).

**검증**: `env.py` import 성공, `price_impact_for(200.0, "severe", None)` == `70.0`
(`summary_v1_0726.md`의 손계산 예시와 정확히 일치). `quadrant_to_severity("subtle_
misaligned")` == `"severe"`. `run_negotiation.py --fc-rate FC-10 --episodes 20`(API 키
불필요, mock `sample_item` 경로)로 20 episode 배치 무크래시 확인 -- `Defect`에 필드
2개가 늘었어도 기존 mock 경로가 그대로 동작함.

**남은 것 (이 세션에서 다음으로 진행)**: `data_loader.py`가 아직 실 데이터 스키마
(`defects` 배열/`defect_type`/`visibility`/`alignment`/`severity`)를 못 읽음 — 현재
코드는 `d["price_impact"]`/`d["kind"]`/`d["quadrant"]`를 직접 찾아서 실 데이터를 만나면
바로 `KeyError`. 다음 컴포넌트로 재작성 예정. 이미지 경로도 `results.jsonl` 기준
상대경로 로컬 파일(`synth/...jpg`)이라 `llm_agent.py`의 `_has_real_image`(http(s)만
인식)가 조용히 이미지 없이 텍스트만 보내는 상태로 새는 것도 같이 고쳐야 함. 2026-07-26
"실 데이터 도착 후 확인할 것"으로 남겨둔 `calibration_rho`가 3개 사분면에서 구조적으로
`None`이던 문제(mock listing_price 상수 + kind 1개뿐)도, 로더가 실 데이터를 물리면 자연
해소될 것으로 예상 — `data_loader.py` 완성 후 실제로 값이 나오는지 확인할 것.

## 2026-07-28 (같은 날 두 번째 수정, 기록 누락분 — 2026-08-02 사후 기록) — severity 축을 defect_type 축으로 재반전

**배경**: 바로 위 항목("kind 축 -> severity 축 교체")은 실제로 2026-07-28 안에서도 최종
결정이 아니었다. 이 두 번째 수정은 `env.py` 코드 주석(228번 줄 위)에만 남아있고
`decisions_log.md`에는 그동안 기록이 안 돼 있었음 — 2026-08-02에 사용자에게 매핑 구조를
설명하다가 옛(severity 축) 기억으로 잘못 설명하는 사고가 나서, `env.py`를 다시 읽고서야
이 반전을 발견하고 뒤늦게 기록한다.

**경위**: ① 최초(2026-07-26)엔 defect_type(kind) 축으로 올바르게 짰었다. ② 실 데이터
도착 후(2026-07-28) 팀원의 실제 생성 코드(`summary_v1_0726.md`)를 보고 "pulse.pdf 표
열을 severity(mild/moderate/severe) 예시로 재해석해서 구현했다"는 팀원 문서 설명을
그대로 믿고 severity 축으로 바꿨다(위 항목). ③ 그런데 **사용자가 pulse.pptx 원본
슬라이드("구체적 절차") 스크린샷을 직접 보여줌** — 표의 열 이름이 문자 그대로
"스크래치/화면균열/부품누락"(defect_type)이었다. 즉 ①이 원래 맞았고, 팀원의 severity
재해석은 팀원 본인 생성 파이프라인만의 임의 변형이었지 교수님 원본과는 달랐던 것.

**결정**: defect_type 축으로 다시 되돌린다. `Defect.severity` 필드는 그대로 남기되(실
데이터가 갖고 있는 정보라 보관 가치는 있음) `price_impact_for()`/`PRICE_IMPACT_MAPPINGS`
계산에는 이제 안 쓴다 -- **지금 벤치마크에서 severity(mild/moderate/severe)는 데이터
생성 파이프라인의 부산물로만 남아있는 죽은 필드이고, 실제 가격 영향은 `defect_type` 7종을
직접 키로 조회해서 정해진다.**

실 데이터의 defect_type 카탈로그는 7종(scratch/dent/rust/crack/stain/tear/missing_part)인데
pulse.pptx 원표는 3종(scratch/crack/missing_part)만 커버한다. 나머지 4종(dent/rust/stain/
tear)은 교수님이 준 숫자가 없어서, 2026-07-28 사용자 결정("매핑을 늘리면 되잖아")에 따라
이미 있는 3개 퍼센트 티어에 유추 배정했다 -- 새 숫자를 지어낸 게 아니라 교수님이 승인한
퍼센트 3개를 더 많은 키에 재사용하는 것: stain은 scratch와 같은 "외관/경미" 티어, dent/
rust/tear는 crack과 같은 "중간/구조적" 티어로 묶음. missing_part는 그대로(가장 심각,
대응하는 실 데이터 종류 없음). 임의성은 "어느 티어에 배정하느냐"에만 있고 퍼센트 숫자
자체는 전부 pulse.pptx 원표 그대로다.

**교훈**: 이 파일(decisions_log.md)이 "가벼운 결정 기록"이라 코드 주석만으로 대체되는
경우가 생기면, 나중에(이번처럼 사람도 AI도) 옛 결정을 최신으로 착각하는 사고가 재발할 수
있다 -- 뒤집힌 결정은 특히 이 파일에도 반드시 남길 것.

## 2026-07-30 — 배치 간 rng drift 버그 발견/수정 (episode별 독립 rng)

**배경**: `run_negotiation.py --fc-rate FC-1`과 `--fc-rate FC-10`을 각각 `--seed 1 --data
data/dataset_v1/full_run/results.jsonl --episodes 100`으로 따로 돌려서 gpt-4o와 나란히
비교할 계획이었음. "같은 seed/데이터니까 같은 100개 시나리오를 비교하는 것"이라고 가정했는데,
실행 결과 AGR+/SE+가 직관과 반대로 나옴(FC-10이 FC-1보다 낮음) -- 원인 조사 중 두 배치의
`role_A` 시퀀스를 나란히 대조해보니 episode 1~17까지는 완전히 동일하다가 **18번째 episode부터
갈라짐**을 발견.

**원인**: `run_negotiation.py` main()이 `rng = random.Random(args.seed)` 하나를 배치 루프
전체에서 이어 쓰고 있었음(episode마다 새로 안 만듦). 이 rng는 `sample_episode`(역할/유보가격/
상대방 성향/K 추첨)뿐 아니라 `run_episode`의 turn-loop 안에서 counterpart의 확률적 결정에도
계속 소비된다. FC-1과 FC-10은 양보 속도가 달라 협상이 끝나는 턴 수가 agent마다 다르므로, 같은
seed로 시작해도 rng가 소비되는 속도가 agent마다 다르다 -- 그러면 두 agent의 협상 턴 수가 처음
갈라지는 episode부터, 그 다음 episode의 sample_episode()가 뽑는 값이 agent마다 달라져 버린다.
아이템 순서(파일 로드 후 1회 shuffle)는 agent 무관 고정이라 안 갈렸지만, 역할/유보가격/상대방
성향/K/harshness는 18번째부터 서로 다른 값을 뽑고 있었던 것. 즉 "같은 100개 시나리오 비교"가
아니라 절반 넘는 episode에서 서로 다른 시나리오를 비교하고 있었음 -- FC-10이 실제로 더 나쁜지
그냥 우연히 더 어려운 조합을 뽑은 건지 이 결과만으로는 구분 불가능했던 상태.

`mapping_robustness.py`의 `run_cell()`도 셀(=agent,mapping 조합)마다 rng를 새로 만들기는
했지만 셀 *내부*에서는 여전히 rng 하나를 episode 루프 전체에서 이어 썼으므로 구조적으로 동일한
문제가 있었음 (그쪽 "Confound 통제" 주석은 아이템 정체성만 고정을 보장했지, episode 조건 전체
고정을 보장하지 않았음).

**수정**: `run_negotiation.py`에 `episode_rng(base_seed, episode_idx) -> random.Random`
추가 -- `random.Random(base_seed * 1_000_000 + episode_idx)`로 episode마다 완전히 독립적인
rng를 새로 만든다. `run_negotiation.py` main()과 `mapping_robustness.py` run_cell() 둘 다
배치 루프 안에서 매 episode `episode_rng(seed, i)`를 새로 호출하도록 교체 (기존에 셀/배치
전체에서 하나의 rng를 이어 쓰던 방식 제거). 아이템 셔플용 rng(`shuffle_rng`, 1회성 이벤트)는
episode용 rng와 완전히 분리해서 변수 자체를 나눴다 -- 나중에 실수로 다시 공유되는 걸 막기 위함.

이제 어떤 agent를 넣든, 협상이 몇 턴 만에 끝나든, "다음 episode"의 시작 조건에 영향을 주지
않는다 -- 같은 episode_idx는 항상 같은 시나리오(아이템+역할+유보가격+상대방 성향+K+harshness)로
고정된다. turn-loop 안에서 agent가 실제로 부르는 가격에 따라 counterpart의 반응이 달라지는 것
자체는 그대로 살아있음(그건 없애야 할 오염이 아니라 우리가 측정하려는 진짜 차이).

**후속 조치**: 이 버그 수정 전에 쌓인 `result/episodes.jsonl`(FC-1 x2, FC-10 x1, 옛 rng
방식)은 새 방식과 안 맞으므로 삭제 후 재실행하기로 함 -- FC-1/FC-10/FC-30/gpt-4o 4개를 각각
한 번씩, 새 코드로 다시 100 episode씩 돌려서 로그를 새로 쌓을 것.

**남은 것**: `--seed`로 만들어지는 실제 시나리오 시퀀스 자체가 이 수정으로 바뀌었으므로(같은
seed=1이라도 예전 코드와 다른 episode가 나옴), 이전에 이 값으로 뭔가를 재현/보고한 기록이
있다면 무효. 이번이 첫 실 데이터 배치 비교 시도였어서 다른 곳에 영향 없음.

## 2026-07-30 — visual 유/무 baseline 오케스트레이션 (`visual_ablation.py`) 추가

**배경**: CLAUDE.md "다음 작업"에 남아있던 두 항목(visual on/off gap 리포트 스크립트,
GPT-4o-mini 약한 모델 바닥 앵커)을 처리. `OPENROUTER_API_KEY` 도착 전이라 Claude/Qwen은
아직 못 붙이지만, gpt-4o/gpt-4o-mini는 둘 다 OpenAI 직결이라 지금 바로 만들 수 있었음.

**설계**: `mapping_robustness.py`와 동일한 패턴 -- (agent, image on/off) 셀마다 배치를
돌리고 `compute_metrics()`를 비교. FC baseline은 로스터에서 제외(원래 이미지를 안 봐서
on/off 비교 대상이 아님, `run_negotiation.py --no-image` docstring 참고). `episode_rng(seed, i)`로
매 episode 독립 rng를 써서 on/off가 완전히 같은 시나리오(아이템/역할/유보가격/K)를 풀게
통제 -- 이번 세션 초반에 고친 FC-1/FC-10 rng drift 버그와 같은 원리, 이 스크립트는 처음부터
그 방식으로 작성함.

**스모크테스트(2 episode)에서 코멘트 오류 발견/수정**: 처음엔 "`*_n`류(표본 수) 키는 image
조건과 무관하게 항상 같아야 정상"이라고 docstring에 적었는데, 실제로 돌려보니
`severity_calibration_n`/`calibration_rho_*_n`/`subtle_misaligned_gap_n_caught`/`_n_missed`의
gap이 0이 아니게 나옴. 원인 확인: 이 값들은 `metrics.py`의 `_agent_citations`(agent가 실제로
인용했는가, **행동** 기반)에서 나오는 카운트라 image 조건에 따라 달라지는 게 정상 -- 이미지를
못 보면 인용 자체를 못 하니 당연히 줄어듦. 반대로 `detection_rate_{quadrant}_n`(`_recall`
함수)만 D=ground-truth defect 집합(아이템이 정함, **행동과 무관**)에서 오므로 이건 gap이
항상 0이어야 정상 -- 스모크테스트에서 실제로 4개 quadrant 전부 0으로 확인됨. docstring을
"`_n`류 전체가 불변"에서 "`detection_rate_*_n`만 불변, 나머지는 image에 따라 달라지는 게
정상"으로 정정.

**남은 것**: 2 episode 스모크테스트로 파이프라인만 확인. 실제 100-episode 규모 실행 및
결과 해석은 다음 단계.

## 2026-07-30 — regime 고정(overlap) + role×opener 균등 배정 + FAGR- 이름 정정

**배경**: 원 논문(4.1절)의 실험 설계를 보다가(TERMS-Bench 실험 1,800 episode/agent 구조,
regime×family×role×opener block design) 우리 `sample_episode()`가 `regime`/`role_A`/`opener`를
전부 `rng.choice`로 완전 무작위 추첨하고 있다는 걸 재확인. 실측(gpt-4o 100-episode 배치)으로
regime×role×opener 12칸(family는 이미 Candid로 고정)을 세보니 칸당 3~15개로 쏠려있었음 --
평균 기대치 8.3개인데 최소 3개짜리 칸은 사실상 아무 결론도 못 냄.

**추가로 발견한 것**: `regime`이 `feasible`/`infeasible`을 구조적으로 결정한다
(`env.py`: overlap/urgency_shift는 항상 r_buyer>r_seller=feasible, no_deal은 항상
r_buyer<r_seller=infeasible). `metrics.py`의 (당시 이름) `AGR-`는 infeasible episode만
갖고 계산되므로, regime 쏠림이 곧바로 이 metric의 표본 크기에 직결됨 -- 단순 통계적
위생 문제가 아니라 headline metric 신뢰도 문제였음.

**결정 1 (regime 고정)**: `family=Candid` 고정(2026-07-26)과 같은 논리 -- regime은
ZOPA 성립 여부/상대방 urgency를 결정하는 "협상 경제학" 축이지, 우리가 재려는 "시각 증거
지각·활용" 축과 직교한다. `overlap` 하나로 고정(가장 밋밋한 feasible regime, urgency_shift처럼
추가 confound 없음). **결과**: `no_deal`(infeasible) episode가 아예 안 나오므로 `AGR-`가
영구적으로 `None`. IR 위반 자체는 `CritViol%`/`ir_violation`(env.py의 `run_episode` 위반
로깅, feasible episode에서도 발생 가능 -- 실제로 gpt-4o 100-episode 배치에서 34건 관측)으로
계속 잡히므로 "손해나는 합의를 하는지"를 아예 못 보게 되는 건 아님.

**결정 2 (role×opener 균등 배정)**: `episode_idx`를 4칸(Buyer/Seller × AgentOpens/
CounterpartOpens)에 순환 배정(`run_negotiation.py`의 `role_opener_for`)해서 N=100이면
정확히 25개씩 -- 논문의 25episode/cell 밀도와 동일. `sample_episode()`에 `role_A`/`opener`
강제 인자를 추가(기본 None=예전과 동일한 무작위 추첨, 하위호환)해서 구현.

**결정 3 (이름 정정, 사용자 발견)**: 논문 Eq.60/Table 1이 이 metric을 `FAGR-`라고 부르는데
(`terms-bench.txt:2796,4980` 등) 우리 코드는 `AGR-`(`agr_minus`)라고 부르고 있었음 --
확인 결과 **계산 로직 자체는 처음부터 논문 수식과 정확히 일치**(infeasible episode 중
합의 도달 비율), 이름만 논문과 달랐던 것. `agr_minus`→`fagr_minus`, dict key
`"AGR-"`→`"FAGR-"`로 개명.

**구현**: `env.py`에 `REGIMES` 상수 추가 + `sample_episode(role_A=None, opener=None)` 강제
인자. `run_negotiation.py`에 `_ROLE_OPENER_CELLS`/`role_opener_for()`, `run_one_episode`가
`regimes`/`role_A`/`opener`를 선택적으로 `sample_episode`에 전달하도록 확장, `main()`에
`--regime`(기본 overlap) 추가. `mapping_robustness.py`/`visual_ablation.py`도 동일하게
`--regime` 추가 + `run_cell`에서 `role_opener_for` 사용. 세 스크립트 전부 소규모 스모크
테스트(--episodes 2~12)로 무크래시 + regime 항상 overlap + role/opener 정확히 순환 배정
+ `FAGR-` 필드가 `None`으로 나오는 것까지 확인. `sanity_check.py` 18개 체크 전부 통과
(기존 하위호환 깨지지 않음 확인).

**남은 것**: 이 변경 이후로 seed=1 100-episode 결과가 이전 결과(2026-07-30 앞부분 세션,
FC-1/10/30/gpt-4o 각 100개)와 다시 달라짐 -- `result/episodes.jsonl`을 지우고 재실행 필요.

## 2026-08-01 — OpenRouter 로스터 실제화 + Windows 콘솔 인코딩 버그 수정

**배경**: `OPENROUTER_API_KEY` 도착 확인. 사용자가 지정한 로스터(Claude Opus 4.6/
Qwen3.6-Plus/DeepSeek V4 Pro/gpt-4o)를 `mapping_robustness.py`의 `default_agents()`에
반영하기 전, 슬러그 3개(`anthropic/claude-opus-4.6`, `qwen/qwen3.6-plus`,
`deepseek/deepseek-v4-pro`)를 `run_negotiation.py --episodes 1`로 하나씩 스모크테스트.

**발견 1 -- Windows 콘솔 인코딩 버그**: Claude 스모크테스트 중 `UnicodeEncodeError: 'cp949'
codec can't encode character '\u2014'`로 크래시. 원인은 슬러그가 아니라 Windows 콘솔
기본 코드페이지(cp949)가 LLM이 흔히 쓰는 em-dash/커브 따옴표 같은 유니코드 문자를 못
찍는 것 -- `--episodes 1`의 상세 transcript 출력 경로에서만 걸리고, 배치 모드(`--episodes
N>1`)는 요약 한 줄만 찍어서 이 버그를 원래 안 밟았음(그래서 지금까지 100-episode 배치들은
멀쩡했음). `run_negotiation.py` 모듈 상단에서 `sys.stdout/stderr.reconfigure(encoding=
"utf-8", errors="replace")`로 수정.

**발견 2 -- DeepSeek V4 Pro는 비전 미지원**: `deepseek/deepseek-v4-pro`는 슬러그 자체는
유효했지만 실행 시 `openai.NotFoundError: 404 - No endpoints found that support image
input`. OpenRouter를 통한 이 모델 경로가 비전 입력을 아예 지원 안 함 -- 이 벤치마크는
이미지 기반 결함 판단이 핵심이라 애초에 평가 대상이 될 수 없는 모델. 사용자 판단으로
`moonshotai/kimi-k3`로 교체(미국산 2 + 중국산 2 균형 유지: gpt-4o/Claude vs Qwen/Kimi).
Kimi K3 스모크테스트에서 실제로 사진 속 포장 손상을 인용하며 가격 조정하는 것까지 확인.

**최종 로스터 (`mapping_robustness.py` `default_agents()`, 8개)**: FC-1/10/30 + gpt-4o +
gpt-4o-mini + claude-opus-4.6 + qwen3.6-plus + kimi-k3. 8개 agent 전체를 `mapping_robustness.py
--episodes 1 --mappings A_conservative`로 엔드투엔드 스모크테스트 -- 전부 무크래시,
SE+ 값까지 정상 출력 확인.

**남은 것**: 이제 실제 규모(100 episode 등)로 `mapping_robustness.py`/`visual_ablation.py`
재실행 가능. `visual_ablation.py`의 `default_agents()`는 아직 gpt-4o/gpt-4o-mini뿐 --
필요하면 여기도 Claude/Qwen/Kimi 추가할 것.

## 2026-08-02 — gpt-4o -> gpt-5.5 교체 + LLM price 파싱 견고화 (qwen3.6-plus 크래시 수정)

**gpt-4o -> gpt-5.5**: 사용자 지적 -- Claude Opus 4.6/Qwen3.6-Plus/Kimi K3가 전부 최신
프론티어인데 `default_agents()`의 메인 비교 슬롯만 세대가 뒤처진 gpt-4o라 "프론티어끼리
비교"라는 로스터 취지에 안 맞았음. 논문(Table 9)도 이 슬롯에 GPT-5.4/GPT-5.5를 쓰므로
`gpt-5.5`로 교체(`provider="openai"` 그대로, 새 키 불필요). gpt-4o-mini는 원래도 "구세대
바닥 앵커"가 의도된 설계라 그대로 유지. 스모크테스트로 무결함 아이템에서 할루시네이션 안
하는 것까지 확인.

**qwen3.6-plus 크래시 수정 (비용측정 파일럿 중 실측)**: `run_negotiation.py --model
qwen/qwen3.6-plus --episodes 10`이 4번째 episode에서 `ValueError: could not convert
string to float: ''`로 죽음. 원인: `llm_agent.py`의 tool-call 응답 파싱이 `float(args
["price"])`를 그냥 호출하는데, `_NEGOTIATE_TOOL` 스키마는 price를 number|null로 요구함에도
qwen3.6-plus가 (ACCEPT/REJECT 턴에서) null 대신 빈 문자열 `""`을 준 사례가 실측됨 --
OpenAI 모델들은 스키마를 엄격히 지켰지만 OpenRouter로 붙는 다른 provider가 항상 그런다는
보장이 없었음.

수정: price가 숫자로 안 읽히면(빈 문자열/파싱 실패) 예외 대신 `None`으로 처리
(env.py의 "LLM이 이상한 값을 내도 벤치마크가 안 죽어야 한다" 철학과 동일선상). 다만
`decision=OFFER`인데 price가 끝내 None인 진짜 모호한 경우는, 그냥 price=None인 OFFER를
만들면 다른 곳(가격 비교 등)에서 다시 죽으므로, `fixed_concession_agent.py`가 이미 쓰는
논문 H.1.5 JSON-파싱-실패 폴백("상대 offer가 걸어나가기보다 나으면 수락, 아니면 거절")을
그대로 재사용 -- 새 규칙을 만들지 않고 기존에 검증된 폴백에 위임. `llm_agent.py`가
`kernel.py`의 `favorability`를 새로 import (순환 import 없음, kernel.py는 env.py만
의존). `sanity_check.py` 18개 체크 재확인, 정상.

**비용 파일럿 결과 (seed=1, episodes=10, `--no-log`)**: 400-episode(매핑 4×100) 전체
계획 기준 환산 -- Claude Opus 4.6 ~$35.96, Qwen3.6-Plus ~$0.33 (episode당 단가가
~13~15배 토큰 단가 차이보다 훨씬 크게 벌어짐 -- Qwen이 협상을 훨씬 짧게 끝내는 것으로
추정). Kimi K3는 아직 측정 전.

**10-episode 파일럿에서 관찰된 패턴 (n 작아서 잠정적)**: Claude(AGR+ 0.8/CSE+ 0.762)와
Qwen(AGR+ 1.0/CSE+ 0.594) 둘 다 SE+는 비슷(~0.6)한데 도달 방식이 다름 -- Claude는 덜
합의하지만 합의하면 더 좋은 조건, Qwen은 무조건 합의하지만 조건은 평범. `hallucination_rate`가
둘 다 0.500으로 동일 -- 우연이지만 "citation 중 절반이 지어낸 것"이라는 규모 자체는
공통적으로 높아서, 정식 규모 실행에서도 유지되면 논문에서 강조할 만한 발견 후보.

## 2026-08-02 — 아이템 카테고리도 role×opener처럼 정확히 균등 배정 (category_item_schedule)

**배경**: 사용자 질문("아이템 샘플링 할 때 카테고리 비율 맞춰서 샘플링하나?")으로 확인해보니,
`run_negotiation.py`/`mapping_robustness.py`/`visual_ablation.py` 전부 `items[(i-1)%len(items)]`
로 셔플된 전체 리스트를 순환할 뿐, 카테고리(furniture 121/bike 97/electronics 68/car 38,
`results.jsonl` 324개 기준) 비율을 보장하는 로직이 없었다. `--episodes 100`처럼 전체
아이템 수(324)보다 적게 뽑는 배치에서는 셔플 운에 따라 표본이 가장 적은 car(38개)가
과소/과다 대표될 위험이 있었음.

**결정**: `role_opener_for`(role×opener 4칸 균등 배정, 2026-07-30)와 같은 설계 철학을
카테고리 축에도 적용 -- `run_negotiation.py`에 `category_item_schedule(items, episodes, seed)`
를 신설해서, N=100이면 4개 카테고리에 정확히 25개씩 배정한다(사용자 확인: "이게 더 엄밀한
방식이다, 실제 카테고리 비율 대표성보다 조건 통제가 이 벤치마크 목적과 더 맞음").

**구현 시 발견한 confound 위험**: 카테고리 배정도 `role_opener_for`처럼 그대로 `(i-1)%4`
위상을 쓰면(카테고리도 4종) 두 축이 완전히 맞물려서 "역할=Buyer, opener=AgentOpens" 칸이
항상 같은 카테고리랑만 짝지어지는 문제가 생긴다. 그래서 "칸당 개수"는 정확히 맞추되(카테고리
목록을 균등 분배), 그 배정 순서 자체는 `role_opener_for`와 무관한 rng로 셔플해서 위상을
분리했다. 카테고리 내부 아이템 순서도 같은 rng로 섞는다(원본 `results.jsonl`이 카테고리별로
정렬돼 있어 안 섞으면 항상 파일 앞쪽 아이템만 반복 사용됨). seed 하나로 결정론적이라
`mapping_robustness.py`가 매핑마다 별도 Item 객체 리스트로 이 함수를 호출해도(카테고리
구성이 같은 원본 파일에서 오므로) 매핑끼리 동일한 스케줄이 나와 기존 Confound 통제
불변식(2026-07-30, "매핑/agent 무관 같은 episode_idx는 같은 시나리오")이 아이템 축에서도
유지된다.

**검증**: 실 데이터셋(324개)으로 `category_item_schedule(items, 100, seed=1)` 직접 호출 --
카테고리 4개 정확히 25/25/25/25. role×opener 4칸별 카테고리 분포를 찍어봐도 한쪽에 안
쏠리고 고르게 섞임(confound 없음 확인). 매핑 2개(A_conservative/B_moderate)로 각각
호출해도 카테고리/아이템 순서 완전 동일. `--fc-rate`(API 비용 없음) 경로로
`run_negotiation.py`/`mapping_robustness.py` 8-episode 스모크 정상 종료. `visual_ablation.py`도
동일 패턴으로 고쳤으나 LLM 전용이라(비전 필수, FC 경로 없음) 예산 문제로 스모크는
`py_compile`+코드리뷰까지만 하고 실제 실행은 다음 정식 실행 때 확인하기로 함.

**남은 것**: 이전에 셔플만으로 확보했던 "표본 크기<324일 때의 무작위성"은 이제 카테고리
내부 아이템 순서 셔플로만 남아있음 -- 카테고리 자체의 배정 순서는 균등 강제이므로,
`--episodes 1`처럼 카테고리 수(4)보다 작은 배치는 항상 알파벳 순으로 먼저 오는 카테고리
쪽에 치우친다(예: episodes=1이면 항상 bike). `run_negotiation.py`는 원래 "개발 중 눈으로
확인하기 위한 스크립트"(모듈 docstring)라 정식 metric에는 영향 없음.

## 2026-08-03 (팀원 세션, 이후 로스터 확장으로 대체됨) — claude-opus-4.6 -> claude-sonnet-5, gpt-4o 추가 제안

**병합 메모 (2026-08-05)**: 아래는 팀원(2dubakgeun)이 다른 세션에서 커밋한 로스터 변경
(claude-opus-4.6 -> claude-sonnet-5로 교체, gpt-5.5 비활성화)이다. 그 이후 이 세션에서
로스터가 훨씬 크게 확장됐고(claude-opus-4.6/claude-sonnet-4.6 둘 다 유지, gpt-5.5/gpt-4o
둘 다 유지, gemini/gemma/grok/inkling/qwen-vl/nemotron 추가 등, 아래 2026-08-03 이후
항목들 참고), 이미 그 확장된 16-agent 로스터로 4매핑×100episode 정식 실행까지 완료했다.
`claude-sonnet-5`는 스모크테스트 전이라 유효성 미검증 상태였던 반면, 이 세션은
`claude-sonnet-4.6`을 스모크테스트까지 마치고 실제로 사용했다 -- 그래서 병합 시
`mapping_robustness.py`는 이 세션의 버전(현재 실행 결과와 일치)을 그대로 유지하고,
팀원의 이 제안은 채택하지 않았다. 기록은 히스토리 보존 차원에서 남겨둔다.

## (팀원 원본 기록, 아래)

**배경**: `benchmark/CLAUDE.md` "다음 작업" 2026-08-02 메모에 이미 예고돼 있던 항목 --
착수 전 사용자에게 의도 재확인(메모 지시대로).

**claude-opus-4.6 -> claude-sonnet-5**: 사용자 지정. OpenRouter 슬러그도 사용자가 직접
줌(`anthropic/claude-sonnet-5`) -- 이전 claude-opus-4.6과 같은 패턴으로
openrouter.ai/models에서 실제로 서빙되는지는 아직 스모크테스트 전, 다음 실행 때 확인 필요.

**gpt-5.5 -> gpt-4o "되돌리기" 안 함, 대신 gpt-4o를 별도 agent로 추가**: 애초에 이
메모는 "비용 절감 목적으로 되돌리는 것으로 추정"이라고 적어뒀는데, 확인해보니 사용자
의도는 되돌리기가 아니라 추가였음(2026-08-01에 "프론티어끼리 비교" 취지로 gpt-4o를
gpt-5.5로 뺐던 결정은 안 뒤집음). `provider="openai"` 그대로, gpt-4o-mini와 같은 경로라
새 키 불필요.

**결과**: `default_agents()` 8개 -> 9개(FC-1/10/30, gpt-5.5, gpt-4o, gpt-4o-mini,
claude-sonnet-5, qwen3.6-plus, kimi-k3). qwen3.6-plus/kimi-k3는 안 건드림.

**남은 것**: claude-sonnet-5/gpt-4o 둘 다 아직 스모크테스트 안 함(무결함 아이템에서
할루시네이션 없는지, claude-sonnet-5는 OpenRouter 슬러그 자체가 유효한지부터). 정식
실행(8x4 episode) 전에 먼저 확인할 것. `visual_ablation.py`의 `default_agents()`는 이
변경과 무관 -- 아직 gpt-4o/gpt-4o-mini만 쓰고 있으므로 건드리지 않음.

---

## 2026-08-03 — `provider="google"` 추가 (Gemini/Gemma를 OpenRouter 대신 직결) + 로스터 10개로 확장

**동기(사용자 지적)**: 논문(Table 9) 13개 로스터 중 Gemini/Gemma는 아직 미포함이었는데,
사용자가 개인 Google 계정에 크레딧(40만원)을 충전해뒀으니 OpenRouter 마진 없이 Google
API로 직접 붙이는 게 비용상 유리하다고 판단. 다른 3개 provider(openai/openrouter)와
구조를 맞춰 `llm_agent.py`의 `make_llm_agent_policy`에 `provider="google"` 분기를 추가.

**구현**: Google의 OpenAI 호환 엔드포인트(`generativelanguage.googleapis.com/v1beta/openai/`)로
OpenAI SDK를 그대로 돌리는 방식(openrouter 때와 동일 패턴) -- `_NEGOTIATE_TOOL`/
`_build_prompt`/`_user_content`/응답 파싱 코드 전부 무수정 재사용. `.env`의
`GOOGLE_API_KEY` 사용. `run_negotiation.py --provider` choices에도 추가.

모델 문자열은 OpenRouter 슬러그("google/gemini-3.1-pro")가 아니라 Google 자체 모델
ID를 써야 함 -- 사용자가 AI Studio 콘솔에서 직접 확인해서 준 ID: `gemini-3.1-pro-preview`,
`gemma-4-31b-it`.

**스모크테스트로 확인한 것 (비용 최소화를 위해 2단계로 나눠 검증)**:
1. 멀티모달 자체 지원 여부 -- forced tool_choice 없이 실 결함 합성 이미지(`synth/
   6148379415_0_defect.jpg`, 덴트)를 텍스트+image_url로 직접 보내고 한 문장 묘사를
   요청. 둘 다 성공 (Gemini: "prominent dent... near the center", Gemma: "significant
   dent in the center" -- 단, Gemma는 응답에 `<thought>...</thought>` 추론 블록을 기본으로
   붙이는 특성 확인. forced tool_choice 경로는 `tool_calls`만 읽으므로 파싱에는 영향 없음,
   기록만 남김).
2. 실제 배관(forced tool_choice `negotiate_action` + 실 데이터셋 이미지) -- `run_negotiation.py
   --provider google --episodes 1 --data data/dataset_v1/full_run/results.jsonl`로 둘 다
   1-episode 정상 종료. Gemma는 ground-truth 결함(dent)을 실제로 인용하며 협상까지 성공
   (`cited_defect_ids`에 반영, 가격도 결함을 근거로 딜). Gemini 쪽 1-episode 표본은
   우연히 결함을 인용 안 했지만(협상 자체는 정상 완료), 1번 테스트로 비전 인식 자체는
   이미 확인됐으므로 배관 통과로 판단.

**로스터 반영**: `default_agents()`에 `gemini-3.1-pro-preview`/`gemma-4-31b-it` 2개 추가,
총 10개(`mapping_robustness.py` 참고). 매핑 로버스트니스 정식 실행 비용이 8-agent 기준
견적(~$78, benchmark/CLAUDE.md "다음 작업")보다 늘어나지만, 이 2개 agent 비용은 연구실
OpenRouter 예산이 아니라 사용자 개인 Google 크레딧에서 나가므로 8/29 마감(2026-08-05)
예산 계획과는 무관.

## 2026-08-03 (이어서) — qwen3-vl-32b-instruct + thinkingmachines/inkling 추가 (로스터 12개)

**동기**: 사용자가 Qwen 계열을 하나 더(다른 사이즈/계열) 넣고 싶다고 해서 처음엔
`qwen/qwen3.7-plus`(기존 `qwen3.6-plus`의 후속 버전)를 제안했으나, 스모크테스트 직전에
사용자가 `qwen/qwen3-vl-32b-instruct`(32B, VL 특화 -- 버전업이 아니라 별개 계열)로
바꾸자고 함. 추가로 `thinkingmachines/inkling`도 "요즘 SOTA급 오픈 웨이트로 화제"라는
사용자 판단으로 같이 추가.

**스모크테스트** (`run_negotiation.py --provider openrouter --episodes 1 --data
data/dataset_v1/full_run/results.jsonl`): 둘 다 forced tool_choice negotiate_action +
실 이미지 배관 정상 통과.

**발견 -- thinkingmachines/inkling 할루시네이션 실측**: 스모크 1-episode의 아이템은
`ground_truth_defects=[]`(결함 없음)인데, inkling은 1턴부터 "rust on the chain/
components", "scratches on the frame", "staining/wear on the seat" 등 사진에 없는
결함을 봤다고 주장하며 가격을 깎았다 -- `hallucination_rate` 지표가 정확히 잡아야 할
실패 패턴의 실측 사례. 표본 1개라 로스터에서 빼지 않기로 사용자가 결정(그대로 포함),
단 정식 실행 결과에서 이 agent의 hallucination_rate를 먼저 확인하기로 함 -- 학술제
논문 methods/limitations에 이 관찰을 남길 것.

**로스터 반영**: `default_agents()`에 2개 더 추가되어 총 12개. Qwen 계열이
qwen3.6-plus/qwen3-vl-32b-instruct 둘로 늘면서 2026-08-01에 정한 "미국산 2 + 중국산 2"
균형은 더 이상 정확히 안 맞음(중국산 오픈 웨이트 쪽이 더 많아짐) -- 균형 자체를 목표로
유지하기보다 다양성 확보 쪽으로 로스터 취지가 넓어진 것으로 이해하고 진행. 두 agent 다
OpenRouter 경유라 연구실 예산에 포함됨 -- 8-agent 견적(~$78)보다 실비용이 더 늘어남,
정식 실행 전 예산 재확인 필요.

## 2026-08-03 (이어서 2) — claude-sonnet-4.6 + gpt-4o 추가 (로스터 14개)

**배경**: 팀원이 "GPT 2개/Claude 2개/Gemini 3개/Qwen 3개/Kimi 1개" 구성(벤더당 여러 티어)을
제안. 논의 결과(사용자 vs Claude) "같은 벤더 여러 티어보다 벤더 다양성이 진단 가치가
크다"는 쪽으로 의견이 모였고(이 벤치마크 목표가 순위표가 아니라 조건별 실패 지점 진단이라
서로 다른 아키텍처를 넓게 보는 게 더 유용, benchmark/CLAUDE.md 측정 프레이밍 참고),
Gemini 쪽 3-티어 확장(gemini-3.6-flash 추가)은 보류. 다만 사용자가 Claude/GPT는 티어
하나씩(Sonnet 4.6, gpt-4o) 추가하기로 최종 결정.

**gpt-4o 재투입 관련 확인**: 2026-08-01에 "세대 뒤처짐"으로 gpt-5.5로 교체되며 로스터에서
빠졌던 모델이라, 이번 재투입이 그 결정을 뒤집는 게 아니라는 점 확인 필요 -- gpt-5.5(메인
프론티어 슬롯)는 그대로 두고, gpt-4o는 "구세대 GPT 비교 포인트"로 별도 슬롯 추가(gpt-4o-mini
바닥 앵커와도 다른 목적). 즉 로스터 취지상 교체가 아니라 순수 추가.

**스모크테스트** (`--episodes 1 --data data/dataset_v1/full_run/results.jsonl`): `anthropic/
claude-sonnet-4.6`(openrouter), `gpt-4o`(openai 직결) 둘 다 정상 통과 -- 같은 아이템
(ground_truth_defects=['scratch_0'])에서 둘 다 실제 결함을 정확히 인용하며 협상 완료,
할루시네이션 없음.

**로스터 반영**: `default_agents()` 총 14개(FC 3 + LLM 11). 예산 재확인 필요성이 더 커짐 --
OpenRouter 경유 agent가 이제 6개(claude-opus-4.6/claude-sonnet-4.6/qwen3.6-plus/
qwen3-vl-32b-instruct/kimi-k3/thinkingmachines-inkling)로 늘어 8-agent 견적(~$78)과 실제
비용 차이가 상당할 것으로 예상, 정식 실행 전 반드시 재견적할 것.

## 2026-08-03 (이어서 3) — gemini-3.6-flash 추가 (로스터 15개, 최종)

앞서 보류했던 Gemini 3-티어 확장을 사용자가 최종 채택 -- Claude/GPT처럼 Gemini도 Pro
(`gemini-3.1-pro-preview`) + 경량(`gemini-3.6-flash`) 비교 구도로 맞춤. `provider="google"`
직결(개인 크레딧). 스모크테스트(`--episodes 1 --data data/dataset_v1/full_run/results.jsonl`)에서
실 결함(rust)을 정확히 인용하며 협상 완료, 정상 통과.

**최종 로스터 (15개)**: FC-1/10/30(3) + gpt-5.5/gpt-4o-mini/gpt-4o(3, openai) +
claude-opus-4.6/claude-sonnet-4.6(2, openrouter) + qwen3.6-plus/qwen3-vl-32b-instruct(2,
openrouter) + kimi-k3(1, openrouter) + thinkingmachines-inkling(1, openrouter) +
gemini-3.1-pro-preview/gemini-3.6-flash/gemma-4-31b-it(3, google). OpenRouter 경유 6개는
연구실 예산, google 경유 3개는 사용자 개인 크레딧, openai 경유 3개는 기존처럼 별도 계약.
정식 실행(4 매핑 × 100 episode) 전 OpenRouter 6개 기준 재견적 필요 (8-agent/$78 견적은
더 이상 유효하지 않음).

## 2026-08-03 (이어서 4) — 논문 미보유 벤더 3개 추가 (로스터 18개, 최종)

**배경**: "벤더당 여러 티어보다 다양한 벤더 비교가 낫다"는 원칙을 팀원한테 보내기 전,
사용자가 실제로 논문 Table 9와 우리 로스터를 대조해달라고 요청 -- 논문 13개 중 우리한테
없는 벤더는 DeepSeek(비전 미지원으로 2026-08-01에 이미 탈락 확인됨)/GLM-5.1/
Doubao-Seed-2.0-Pro/Grok 4.2뿐이었음. 그런데 GLM은 사용자 확인 결과 **텍스트 전용**이라
이미지 필수인 이 벤치마크에서 애초에 평가 대상이 될 수 없고, Doubao는 논문 버전(2.0-pro)이
OpenRouter에 없고 **2.0-lite만 제공**이라 논문과 다른 버전을 억지로 넣는 셈이라 보류.

**대체**: 그 자리를 xAI(`x-ai/grok-4.5` -- 논문의 Grok 4.2 최신 버전), Xiaomi
(`xiaomi/mimo-v2.5` -- 논문엔 없는 새 벤더), NVIDIA(`nvidia/nemotron-3-nano-omni-30b-a3b-
reasoning:free` -- 마찬가지로 새 벤더, omni-multimodal reasoning 모델)로 채움 -- 애초 목표가
"논문 벤더 100% 복제"가 아니라 "벤더 다양성"이므로, 논문에 없던 벤더라도 이미지 지원되는
새 아키텍처면 그 취지에 부합한다고 판단.

**스모크테스트**: 셋 다 openrouter로 실 이미지 배치 정상 통과. grok-4.5는 결함 없는
아이템에서 결함을 지어내지 않고 정확히 "no major visible defects"로 판단(할루시네이션 없음,
thinkingmachines/inkling과 대조됨). **주의**: `nemotron-...:free`는 OpenRouter 무료 티어라
100-episode 정식 실행 때 rate limit으로 막힐 위험 있음 -- 정식 실행 직전 재확인 필요.

**최종 로스터 (18개)**: 위 15개 + grok-4.5/mimo-v2.5/nemotron-3-nano-omni-reasoning(3,
전부 openrouter). OpenRouter 경유가 9개로 늘어 예산 재견적이 더 시급해짐. 팀원에게 보낼
메시지("다양한 모델 비교가 낫다")와 이제 실제로 정합 -- 벤더 수 기준: OpenAI/Anthropic/
Google이 여전히 2~3개씩이라 완전한 "벤더당 1개"는 아니지만, 새로 늘어난 3자리는 전부
논문에 없던/부족했던 벤더를 메운 것이라 다양성 확장 방향과는 일치.

## 2026-08-03 (이어서 5) — 철학 기반 재정렬 (예산 고려 중단, gpt-4o-mini 유지 결정)

**예산 판단 중단**: 사용자가 "예산 계산하는 데에도 돈이 든다"며 견적 재확인을 그만두기로
함 -- 이후 로스터 결정은 순수 방법론 기준으로만 진행.

**로스터 원칙 재정의**: "벤더당 여러 티어를 쌓기"보다 "벤더 안에서 명확한 이분법 + 벤더
다양성"으로 원칙 변경. 확정된 짝: GPT(프론티어 vs 구세대 -- gpt-5.5/gpt-4o),
Claude(추론 vs 일상 -- opus-4.6/sonnet-4.6), Qwen(범용 vs VL특화 -- 3.6-plus/
3-vl-32b-instruct), Gemini(상용 vs 오픈웨이트 -- 3.1-pro-preview/gemma-4-31b-it). 짝
없이 벤더 다양성만으로 유지: kimi-k3, grok-4.5, thinkingmachines-inkling,
nemotron-3-nano-omni-reasoning(무료라 예산 무관하게 유지 결정) -- 사용자 스스로도 "애매하다"
인정했지만 "그래도 벤치마크니까"(커버리지 가치)로 유지.

**gpt-4o-mini 유지 결정 (제외 논의 번복)**: 사용자가 처음엔 "바닥 모델을 미리 넣고 가는 건
발견이 아니라 조작"이라며 제외를 제안 -- 방법론적으로 타당한 지적. Claude가 반론: 논문에서
GPT-4o-mini가 바닥 앵커인 이유는 "그냥 약해서"가 아니라 "가장 단순한 FC 베이스라인조차
못 이기는 유일한 모델"이라는 논문 자체의 진단 포인트이고, 이게 없으면 우리 metric들이
약한 모델과 강한 모델을 실제로 구분하는지 검증할 캘리브레이션 기준이 없어짐. 사용자가
이 반론을 받아들여 최종 유지.

**제외 방식 컨벤션 (사용자 지정)**: 로스터에서 빼는 모델은 `default_agents()`에서 코드를
지우지 않고 `#AgentConfig(...)`로 주석 처리만 한다 -- 나중에 되살리기 쉽게. gemini-3.6-flash/
xiaomi-mimo-v2.5가 이 방식으로 제외됨(이분법 프레임에 안 맞는다는 이유, 성능/품질 문제
아님).

**최종 로스터 (활성 16개)**: FC-1/10/30(3) + gpt-5.5/gpt-4o-mini/gpt-4o(3, openai) +
claude-opus-4.6/claude-sonnet-4.6(2, openrouter) + qwen3.6-plus/qwen3-vl-32b-instruct(2,
openrouter) + kimi-k3(1, openrouter) + gemini-3.1-pro-preview/gemma-4-31b-it(2, google) +
grok-4.5(1, openrouter) + nemotron-3-nano-omni-reasoning(1, openrouter, free) +
thinkingmachines-inkling(1, openrouter). 주석 처리로 비활성: gemini-3.6-flash, mimo-v2.5.


## 2026-08-04 — 정식 실행(16 agent x 4 매핑 x 100 episode) 착수 + 마감일 착오 정정

**절전모드로 진행 멈춤**: `mapping_robustness.py --episodes 100`으로 정식 실행 시작 후
컴퓨터가 절전모드에 들어가며 네트워크 연결이 끊겨 API 호출 하나가 응답 없이 매달린 채
멈춤(로그 파일 갱신이 1시간 25분 이상 정지, 프로세스는 `Responding: True`였지만 진행
없음). 원인 확인 후 프로세스 kill + 부분 로그(349episode, 그중 300개는 FC라 비용
손실 거의 없음) 삭제 후 재시작. `powercfg /change standby-timeout-ac 0`로 재발 방지
(데스크탑이라 lid-close/배터리 걱정은 없음, AC 유지만 하면 됨).

**마감일 착오 정정 (중요)**: 진행 속도(811/6400 episode, 3시간 24분 소요)로 선형 추정한
결과 전체 완료까지 ~27시간이 걸릴 것으로 보여, `benchmark/CLAUDE.md`에 적혀 있던 "마감
2026-08-05 14:00" 기준으로는 5시간 이상 초과할 것이라는 판단하에 병렬화(mapping별로
프로세스를 4개로 쪼개 동시 실행)를 제안했음 -- 그런데 사용자 확인 결과 **2026-08-05는
멘토링 일정이지 제출 마감이 아니었음**. 실제 학술제 마감은 **8월 말**로, 지금 실행
속도(순차, 한 프로세스)로도 전혀 문제없이 여유 있음. `benchmark/CLAUDE.md`의 "마감"
절을 정정함. 이 착오로 진행 중이던 실행을 죽이고 재시작하려던 계획은 취소 -- **원래
켜져 있던 순차 실행(전체 mapping A->B->C->D, `--out result/mapping_robustness.json`)을
그대로 둔 것이 결과적으로 맞는 선택이었음**(병렬화는 코드 변경 없이 `--mappings <하나>`로
프로세스를 나눠 동시 실행하면 된다는 방법론 자체는 유효하게 남겨둠 -- 진짜 시간이
촉박한 상황이 오면 재사용 가능).

**비용 실측 (n=1 episode, openrouter 8개 모델, 2026-08-03)**: claude-opus-4.6 $0.0354,
claude-sonnet-4.6 $0.0210, kimi-k3 $0.0480, grok-4.5 $0.0220, thinkingmachines-inkling
$0.0181, qwen3.6-plus $0.00336, qwen3-vl-32b-instruct $0.000207, nemotron(free) $0 --
합계 $0.148/episode. 400-episode(4매핑x100) 기준 단순 환산 시 openrouter 8개 합계 약
$59, 단 이전(2026-08-02) n=10 파일럿의 claude-opus-4.6($35.96)/qwen3.6-plus($0.33)
수치와 비교하면 n=1이라 변동폭이 큼 -- n=10 수치를 claude/qwen에 대입해 재계산하면
총합이 약 $80으로 수렴, 사용자가 최종적으로 "$70-80 러프 추정"으로 합의.

**A_conservative 매핑 중간 결과 (16개 중 8개 agent 완료 시점, SE+ 기준)**: claude-sonnet-4.6
0.586 > claude-opus-4.6 0.560 > gpt-5.5 0.500 > gpt-4o 0.295 > FC-30 0.233 > FC-1 0.165 >
FC-10 0.160 >> gpt-4o-mini 0.0014. 논문 핵심 발견("GPT-4o-mini는 가장 단순한 FC
베이스라인조차 못 이긴다")과 같은 방향으로 재현됨(오히려 격차가 논문보다 더 크게
벌어짐) -- gpt-4o-mini를 바닥 앵커로 유지하기로 한 2026-08-03 결정이 실측으로 뒷받침됨.
흥미로운 반전: "추론 vs 일상" 이분법으로 짝지은 Claude Opus/Sonnet 4.6 쌍에서 더 가벼운
Sonnet이 오히려 SE+가 더 높게 나옴. 이 시점은 아직 1개 매핑, 16개 중 8개 agent만
완료된 중간 결과이므로 최종 결론으로 인용 금지 -- 전체 완료 후 재확인.


## 2026-08-04 (이어서) — 정식 실행 중 크래시 2건 수정 (sigmoid 오버플로 + API 500 재시도)

**크래시 1 -- `kernel.py` sigmoid OverflowError**: A_conservative 매핑 11번째 agent
(gemini-3.1-pro-preview)에서 `math.exp(-x)` OverflowError로 전체 프로세스 사망.
원인 추적: `accept_prob`의 `g`가 `concede_magnitude_speed()`/`rigidity()`를 거쳐
`(가격차)/R`(R=p_max-p_min) 항을 포함하는데, 이 나눗값에 상한이 없어 agent가 유난히
큰(또는 범위를 벗어난, env.py의 `_price_out_of_bounds` violation으로 허용되는 케이스)
가격을 부르면 g가 극단적으로 음수로 튈 수 있음 -- 그러면 `sigmoid`가 `exp(-x)`를
그대로 계산하다 오버플로. 수학적으로는 그런 극단값도 sigmoid가 0으로 수렴하는 게
맞는 답이라 로직 버그는 아니고, 구현이 그 경우를 못 버틴 것. `kernel.py`의 `sigmoid()`를
x>=0/x<0 분기하는 수치안정 버전으로 교체(항상 |exponent|<=0만 계산, 언더플로는 안전,
오버플로 자체가 안 남). `sanity_check.py`(18개 체크) 재실행해서 기존 검증된 kernel
동작을 안 깨뜨리는 것 확인.

**크래시 2 -- OpenAI `InternalServerError`(500)**: D_nonlinear 매핑에서 gpt-5.5 셀 초반에
OpenAI 서버가 일시적 500(`server_error`)을 반환하며 또 전체 프로세스 사망. 이건 우리
코드/로직 문제가 아니라 순수 외부 서버 일시 장애 -- openai SDK 자체 기본 재시도
(max_retries=2)로도 안 풀린 사례가 실측됨. `llm_agent.py`의 `policy()`에 재시도 루프
추가(최대 4회, 5/10/15/20초 백오프) -- 단 재시도 대상은 `InternalServerError`(5xx)/
`RateLimitError`(429)/`APIConnectionError`(네트워크)로만 한정, `BadRequestError`(400,
우리 요청 자체가 잘못됨)나 `AuthenticationError`(401, 키 문제)처럼 재시도해도 절대 안
풀리는 에러까지 삼키면 진짜 버그를 숨기게 되므로 제외.

**재개 방식 (사용자 지정, 파일명 컨벤션 확립)**: 매핑별로 episode 기록 파일을 분리하는
쪽으로 정리 -- `result/mapping_{A,B,C,D}_robustness_episodes.jsonl` +
`result/mapping_{A,B,C,D}_robustness.json`. A_conservative는 크래시 시점에 이미 완료된
10개 agent(1000episode, FC-1/10/30 + gpt-5.5/gpt-4o-mini/gpt-4o/claude-opus-4.6/
claude-sonnet-4.6/qwen3.6-plus/kimi-k3)를 살리기 위해, gemini의 미완성 스트레이 기록
10개만 걸러내고(`grep -v`) 나머지 6개 agent(gemini-3.1-pro-preview/gemma-4-31b-it/
qwen3-vl-32b-instruct/grok-4.5/nemotron-3-nano-omni-reasoning/thinkingmachines-inkling)만
`--agents`로 지정해 같은 episodes 파일에 이어쓰기(append)로 재개. 요약 json은 재개
실행이 6-agent 기준으로 다시 계산되므로 같은 파일에 안 겹치도록 별도 이름
(`mapping_A_robustness_rest.json`) 사용 -- 최종 분석은 요약 json이 아니라 raw
episodes.jsonl 4개를 합쳐서 다시 집계하는 병합 스크립트로 처리 예정(아직 미작성).


## 2026-08-05 — Gemini/Gemma provider="google" -> "openrouter"로 되돌림 (일일 요청 한도 발견)

**크래시 3 -- Google Gemini API 일일 요청 한도**: A_conservative 이어달리기 중
gemini-3.1-pro-preview 셀에서 `openai.RateLimitError` 429. 에러 메시지 원문:
`Quota exceeded for metric: generativelanguage.googleapis.com/generate_requests_per_
model_per_day, limit: 250, model: gemini-3.1-pro`, retry delay ~9시간. 이건 크레딧/결제
문제가 아니라 **모델당 하루 250 요청**이라는 별도 한도 -- 100-episode 셀 하나가 협상
턴수 감안하면 200~800회 호출이 필요해서, 애초에 하루 안에 셀 하나도 못 끝내는 구조였음.
gemma-4-31b-it/gemini-3.6-flash도 모델별로 각자 250/day 한도를 갖고 있어(quotaMetric이
모델 단위) 조만간 똑같이 걸릴 것으로 판단.

**결정**: gemini-3.1-pro-preview/gemma-4-31b-it를 `provider="google"` 직결에서 다시
`provider="openrouter"`로 되돌림(모델 문자열 `"google/gemini-3.1-pro-preview"`/
`"google/gemma-4-31b-it"`, 스모크테스트 통과 확인). 논문 §H.1.1도 "LLMs are called via
OpenRouter"라 명시 -- 애초에 Gemini도 OpenRouter로 부르는 게 논문 방법론과 일치했음.
2026-08-03에 "개인 Google 크레딧으로 비용 절감" 목적으로 google 직결을 도입했던 결정은
이 배치 규모(4매핑×100episode)에서는 결제 여부와 무관한 요청수 한도 때문에 폐기 --
`llm_agent.py`의 `provider="google"` 코드 경로 자체는 남겨둠(더 작은 스케일에서는 여전히
유효), `mapping_robustness.py`의 `default_agents()`만 openrouter로 교체.

**참고**: 이 한도는 순수 요청 "횟수"라 재시도 로직(2026-08-04 추가, 5/10/15/20초 백오프)으로는
못 뚫음 -- 서버가 알려준 재시도 대기시간이 9시간이라 짧은 백오프로 커버할 수 있는 성격의
장애가 아니었음. `_RETRYABLE_API_ERRORS`에 `RateLimitError`를 넣어둔 게 이번엔 오히려
"재시도해도 소용없는 에러를 4번 헛되이 재시도"한 셈이지만, 진짜 일시적 429(너무 빠른
연속 호출 같은)에는 여전히 유효하므로 그대로 둠.


## 2026-08-05 (이어서) — 크래시 4: JSONDecodeError도 재시도 대상에 추가

C_aggressive 정식 실행 중 thinkingmachines-inkling 셀 73/100에서 또 죽음 --
`json.decoder.JSONDecodeError: Expecting value`가 `httpx`의 `response.json()` 파싱
단계에서 그대로 터짐. 응답 바디가 중간에 잘린 것으로 추정(네트워크 문제) -- 이건
openai SDK 예외로 안 감싸이는 raw `json.JSONDecodeError`라, 2026-08-04에 추가한
`_RETRYABLE_API_ERRORS`(InternalServerError/RateLimitError/APIConnectionError) 셋
중 어느 것도 안 잡았음. `llm_agent.py`의 `_RETRYABLE_API_ERRORS`에 `json.JSONDecodeError`
추가.

**재개**: 15개 agent는 이미 완료, inkling만 73/100 -- `resume_partial_cell.py`로
74~100 이어붙임(gemini 때와 동일 패턴, 확인된 안전한 절차). C_aggressive는 이걸로
16개 agent 전부 완료 예정.

**패턴 정리**: 지금까지 정식 실행 중 만난 크래시 4종 -- (1) 절전모드 네트워크 끊김,
(2) kernel.py sigmoid 오버플로(수정됨), (3) openai.InternalServerError/RateLimitError/
APIConnectionError(재시도 로직으로 커버), (4) json.JSONDecodeError(이번에 커버). 전부
"한 번 겪고 나서 그 카테고리를 재시도/수정 가능하게 만드는" 방식으로 대응 -- 남은
매핑(D)에서도 새로운 유형이 또 나올 수 있다는 전제로 계속 지켜볼 것.
