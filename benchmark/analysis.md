# 분석 계획 (H1~H8)

`analysis.ipynb`에서 진행 중인 가설 검증 목록. "H"는 Hypothesis(가설)의 약자 — 통계학의
H0/H1(귀무/대립가설) 관례를 빌려서, 검증하려는 질문 하나하나에 번호만 붙인 것.

각 H는 "먼저 어떤 대비(contrast)가 뭘 드러낼지 설계 → 데이터로 확인 → 필요하면 대조군 넣어서
재검증"하는 순서로 진행한다 (TERMS-BENCH 논문이 findings를 검증한 방식과 동일 —
`Paper/CLAUDE.md`/`paper-code-map.md` 참고).

---

## H1: Detection vs Grounding 실패 분리 — **완료**

**질문**: 2x2 설계(가시성 Obvious/Subtle × 정합성 Aligned/Misaligned)가 실제로 두 개의
독립된 능력 — Detection(결함을 알아챘는가)과 Grounding(알아챘다면 가치를 제대로
반영했는가) — 을 분리해서 잴 수 있는가?

**측정**:
- Detection drop = recall(Obvious) − recall(Subtle), `quadrant_detection_rate` 풀링
- Grounding = signed(할인액) − price_impact, quadrant별 bias (최초엔 calib_ρ로 시도했으나
  price_impact가 quadrant와 무관하게 defect_type로만 정해지고 quadrant는
  `QUADRANT_BONUS`를 통해 accept_prob에만 영향을 준다는 게 드러나서 bias 방식으로 교체)
- obvious_diff / subtle_diff = misaligned − aligned bias (aligned를 baseline으로 둔 통제된 비교)

**결과**:
- Detection: 13/13 agent 전원 유의 (Obvious보다 Subtle에서 못 잡음). 단, drop 크기 자체를
  순위로 읽으면 안 됨 — gemma-4-31b-it/kimi-k3는 recall_obv 자체가 낮은 바닥효과, 반면
  claude-opus-4.6은 recall_obv=0.612인데 recall_sub=0.217로 최하위인 진짜 capability drop.
  1위(qwen3.6-plus)도 recall_obv=0.673에 불과 — 가장 쉬운 조건의 ceiling 자체가 낮음.
- Grounding: obvious_diff는 13개 중 2개만 유의(그마저도 반대 방향) — obvious 축엔
  misalignment 고유 효과 없음, baseline 보수성일 뿐. subtle_diff는 11개 중 10개 유의(전부
  음수) — subtle이면서 misaligned일 때만 진짜 misalignment 효과가 나타남. 이 quadrant의
  price_impact/R 평균이 0.0486인데 diff가 −0.02~−0.05 수준이라, 받아야 할 할인의
  절반 가까이(많게는 전부)를 못 받는 큰 효과.
- claude-opus-4.6, nemotron은 subtle_misaligned 단일 인용 turn 자체가 없어(n=0) grounding
  측정이 아예 불가 — detection 실패가 너무 심해서 grounding 평가 기회조차 없어지는
  사례로 별도 주목할 만함.

**결론**: Detection은 보편적 실패(모든 agent, Subtle이면 다 실패), Grounding은 subtle_misaligned
하나에만 국한된 실패 — 두 축이 단순히 더해지는 게 아니라 특정 교집합에서만 새로운 실패
유형이 나타난다는 증거.

---

## H2: Subtle-Misaligned 킬러조건 super-additivity 검증 — **완료 (판단 불가)**

**질문**: 가장 어려운 사분면(subtle_misaligned)의 협상 성과(SE+) 하락이 두 난이도 축을
단순히 더한 것보다 더 큰가? H1이 결함 인용 단위(recall, bias)에서 본 걸 여기서는 episode
outcome(SE+) 단위에서 본다.

**측정**:
- 아이템당 결함이 0~1개라 episode를 quadrant로 직접 버킷팅 가능 (citation 여부 무관)
- 가산적 예측 = (obvious_aligned−subtle_aligned) + (obvious_aligned−obvious_misaligned)
- super-additivity gap = 실제(obvious_aligned−subtle_misaligned) − 가산적 예측
- quadrant별 독립 bootstrap(B=2000)으로 gap의 CI 계산
- critical violation(가격범위/IR 위반 등)이 있었던 episode는 SE+ 계산에서 제외 (아래
  발견 참고 — mean 기반 비교가 극단값 하나로 완전히 깨지는 걸 막기 위함. protocol
  compliance는 CritViol%가 따로 재는 별개 축이라 여기 안 섞음)

**첫 시도에서 발견한 문제**: violation 제외 없이 돌렸더니 gemini-3.1-pro-preview의
`se_obvious_aligned`가 -43.632로 나옴 (정상 범위는 대략 -1~1). 원인 추적 결과
`A_conservative` mapping episode_idx=42에서 gemini가 listing price $550인 아이템에
**$300,000 offer**($p_max=4200`, 자기 reservation의 590배)를 낸 게 확인됨 —
`price_out_of_bounds`+`ir_violation`으로 정확히 기록됐지만 협상은 그대로 진행돼 counterpart가
수락, `outcome=300000`이 그대로 SE+ 계산에 들어가 단일 episode가 80개 평균을 완전히
왜곡시킴. 이 조사 과정에서 H8이 파생됨(아래).

**결과 (violation 제외, quadrant 전체 episode)**: `gap_sig`가 13개 중 gpt-5.5 하나뿐이고
CI 하한이 0.004로 거의 0에 붙어있음 — 13번 검정 중 우연히 하나 걸리는 수준. 부호도 양수
6개/음수 7개로 방향성 없음 → population 수준에서 super-additivity 근거 없음. 단, 이건
H1과 모순 아님 — quadrant 안에 detection을 아예 못 한(QUADRANT_BONUS가 안 걸린) episode가
섞여서 H1이 잡은 좁은 효과가 평균에서 희석된 것으로 해석.

**결과 (citation 조건부로 좁힘)**: `n_sm`이 5~28로 급감(한 자릿수 다수). `gap_sig=True`가
4개(gemma, gpt-4o-mini, kimi-k3, nemotron)로 늘었지만 그중 3개는 **음수**(예측보다 덜
나쁨, H2 가정과 반대 방향). 유일한 양수(gpt-4o-mini, +0.267)는 n_sm=6짜리라 신뢰 어려움.

**결과 (H9 발견 후 role 나눠서 재검증)**: H5가 role/quadrant confound로 폐기된 뒤 H2도
같은 위험이 있어 재검증. agent×role 26칸 중 `gap_sig=True` 2개뿐(gemma-4-31b-it BUYER
-0.398, gpt-4o BUYER -0.235, 둘 다 음수=반대 방향). **원래 유일했던 gpt-5.5의 유의한
결과는 role로 나누니 사라짐**(BUYER 0.022, SELLER 0.174 둘 다 안 유의) — H5의 gpt-4o
사례와 같은 패턴, role 섞여서 생긴 착시였음. **그러나 population 수준 결론("증거
없음")은 안 바뀜** — 2/26이면 여전히 우연 수준(기대값 ~1.3)이고 새로 나온 2개도
반대 방향. H5와 달리 H2는 role confound에 상대적으로 안전했던 것으로 판정.

**결론**: 표본 부족으로 **판단 불가**. "효과가 없다"가 아니라 "지금 데이터(4 mapping ×
100 episode/agent)로는 citation-조건부 검정을 감당할 만큼 subtle_misaligned citation이
안 쌓인다"는 뜻. episode 수를 늘릴 예산이 생기면 재시도할 항목으로 보류.

---

## H3: severity_calibration_rho의 mapping 안정성 — **완료**

**질문**: 4개 mapping(A_conservative/B_moderate/C_aggressive/D_nonlinear) 각각에서
agent별 calibration이 안정적인가? mapping 하나에서만 무너지면 "특정 price curve를
패턴매칭"한 것이고, 전부에서 안정적이면 "진짜 가치를 읽는다"는 근거.

**측정**: severity_calibration_rho(quadrant 구분 없이, 단일 인용 OFFER turn의 price_impact
vs |Δprice|/R)를 4개 mapping 각각 따로(풀링 안 함) 계산, episode 단위 bootstrap(B=2000)으로
mapping별 CI 산출.

**결과**:
- 9개 agent(claude-opus-4.6, claude-sonnet-4.6, gemini-3.1-pro-preview, gpt-4o, gpt-5.5,
  grok-4.5, kimi-k3, qwen3.6-plus, thinkingmachines-inkling)는 4개 mapping 전부 유의,
  rho 0.55~0.93로 안정적. gemma-4-31b-it은 3/4 유의.
- gpt-4o-mini: 4개 mapping 전부 CI가 0을 가로지름(점추정치도 -0.34~0.38로 부호까지 흔들림) —
  mapping별로 불안정한 게 아니라 애초에 어느 mapping에서도 calibration이 없음.
- qwen3-vl-32b-instruct: C_aggressive 하나만 유의, 나머지 3개는 약함(0.08~0.45, CI가 0 포함).
- nemotron: n=7~13로 표본 부족, CI 상한이 전부 1.000에 붙어 판단 불가.
- mapping간 rho 크기 차이(A가 낮고 C가 높은 경향)는 agent 능력 차이가 아니라 confound —
  A_conservative는 price_impact 절대 크기가 작아(5~20%) 고정된 가격 노이즈 대비 신호가
  약함. mapping별 rho 차이를 곧바로 "특정 mapping 패턴매칭"으로 해석하면 안 됨.

**결론**: 9~10/13 agent는 mapping이 달라져도 진짜 가치를 일관되게 추적함 — mapping
overfitting 우려는 대부분 기각. gpt-4o-mini는 전 mapping에서 근본적으로 calibration이
없고, qwen3-vl-32b는 부분적으로만 있음. nemotron은 데이터 부족으로 보류.

---

## H4: Hallucination과 surplus의 관계 — **완료**

**질문**: 지어낸 결함(hallucination)을 인용해서 가격을 깎으려 드는 agent가 실제로 더 많은
surplus(SE+/CSE+)를 가져가는가 — 부정직한 전략이 이 커널 구조에서 먹히는지.

**측정**: Level 1(agent간 대략적 확인, hallucination_rate vs SE+/CSE+ 상관, n=13) + Level 2
(같은 agent 내, episode를 `fake_only`(지어낸 것만 인용)/`real`(실제 결함 인용)/`none`(무인용)
3그룹으로 나눠 SE+/AGR+/CSE+를 fake_only 대 나머지 둘로 비교, episode 단위 bootstrap). 전부
critical violation 있는 episode는 제외(H2에서 발견한 이유와 동일 — gemini 등에서 극단치
episode가 섞여있었음, 두 번째 사례도 이 조사 중 발견).

**결과**:
- Level 1: rho(hallucination_rate, SE+)=0.17, rho(·, CSE+)=-0.022 — n=13에 유의선 ~0.55라
  노이즈 범위, agent간 대략 비교로는 무관.
- Level 2 SE+: fake_only vs none 13개 중 4개 유의(claude-opus-4.6, gemini-3.1-pro-preview,
  kimi-k3, qwen3-vl-32b-instruct), 전부 음수. fake_only vs real은 gemini 하나만 유의.
- Level 2 AGR+(성사율): fake_only vs none 7개 유의, 6개 음수(성사율 하락) + gpt-4o-mini만
  강하게 양수(+0.246). fake_only vs real은 claude-opus-4.6/gemini만 음수로 유의, gpt-4o-mini는
  여기서도 양수로 유의.
- Level 2 CSE+(성사된 경우만): claude-opus-4.6/gemini는 완전히 무관(0 근처) — 이 둘의 SE+
  penalty는 100% AGR+(성사가 덜 됨)에서 옴. gpt-4o-mini/qwen3-vl-32b-instruct는 CSE+가
  음수로 유의 — 성사는 되는데 가격이 나쁨.
- **gpt-4o-mini는 다른 12개와 반대 패턴**: AGR+ 상승(더 쉽게 성사) + CSE+ 하락(성사돼도
  가격 나쁨) — 지어낸 결함을 대며 빨리, 대신 나쁜 조건에 합의해버리는 것으로 보임.
  H1(바닥 detection)/H3(calibration 전무)에서도 계속 특이 케이스였던 모델.

**결론**: 거짓말이 이득을 준다는 증거는 어디에도 없음 — 커널 설계(`evidence_term_for`가
지어낸 결함을 무보상 처리)가 의도대로 작동. 오히려 대부분 agent에서 손해(주로 AGR+ 하락)로
이어지며, 그 경로는 agent마다 다름. gpt-4o-mini는 정반대 기전(과잉 양보형)이라 H6에서
개별적으로 더 볼 것.

---

## H5: Agent별 quadrant 약점의 mapping 안정성 — **폐기 (H5_retrial로 대체)**

**질문**: agent마다 특정 quadrant에서 약한 패턴("지문")이 4개 mapping에 걸쳐 고정적으로
나타나는가, 아니면 mapping마다 흔들리는가 — 흔들리면 특정 price curve에 대한
overfitting을 의심할 근거.

**측정**: agent x mapping x quadrant(role은 안 나누고 풀링)로 SE+/AGR+/CSE+ 계산, 4개
mapping 중 "가장 나쁜 quadrant"(worst)를 찾고 mode/agree count로 안정성 판단. 순수
무작위 대비(4개 카테고리에서 4번 무작위 추출 시뮬레이션, N=200,000)로 관측된 agree
분포가 유의미하게 안정적임을 확인(agree=4가 무작위 기대치의 거의 10배).

**폐기 이유**: H6로 넘어가며 gpt-5.5/nemotron의 transcript를 직접 읽다가, counterpart가
SELLER일 때(=agent가 BUYER) 첫 제안이 자기 reservation에서 평균 36%(R 대비) 벗어나는데
counterpart가 BUYER일 때(=agent가 SELLER)는 평균 5.4%만 벗어난다는 걸 발견(전체 episode
풀링, 7배 차이). 원인은 `p_min=0`(고정 하한) vs `p_max`(느슨하게 큰 상한)의 구조적 비대칭 —
코드 어디에도 의도됐다는 근거 없음. 게다가 **quadrant 배정 자체가 role과 균등하지
않았음**: obvious_aligned=BUYER 60.0%, subtle_aligned=38.8%, subtle_misaligned=35.3%,
obvious_misaligned=51.1%(거의 균형). 즉 H5가 role을 안 나누고 quadrant를 비교한 게
근본적인 설계 결함으로 드러나 폐기, role을 나눈 재분석(H5_retrial)으로 대체.

**참고**: H2도 quadrant별 SE+를 role 안 나누고 계산했으므로 같은 confound가 있을 수
있음 — 나중에 재검증 필요(미시작).

---

## H5_retrial: role(BUYER/SELLER)을 통제한 quadrant 약점 재분석 — **완료**

**측정**: H5와 동일하되 agent x **role** x mapping x quadrant로 쪼개서 worst quadrant의
mode/agree를 role별로 따로 계산.

**결과**:
- **BUYER와 SELLER의 worst quadrant가 같은 agent는 gpt-4o 하나뿐** — 나머지 12개는
  역할마다 다른 quadrant가 약점으로 나옴. 즉 대부분 agent는 "quadrant 지문 하나"가 아니라
  "역할별로 다른 지문 두 개"를 가짐.
- 원래 H5 결과가 role 하나와는 맞아떨어지는 경우 11/13. 이 중 일부는 role로 나누니 신호가
  더 강해짐(claude-sonnet-4.6: 3/4→BUYER 4/4, gemma-4-31b-it: 2/4→BUYER 4/4 & SELLER
  4/4 둘 다 완벽) — 원래는 두 개의 강한 신호가 섞여서 뭉개져 있었던 것.
- nemotron: 원래 "obvious_aligned 4/4 완벽"이었는데, 알고 보니 BUYER는 quadrant 안 가리고
  전부 바닥(사실상 무작위)이고 **SELLER에서만 obvious_aligned(3/4)이 진짜 신호** — 원래
  결과는 BUYER의 전반적 무능이 만든 착시가 상당 부분 섞여있었음.
- **gpt-4o, qwen3-vl-32b-instruct는 원래 H5 결과가 role 어느 쪽과도 안 맞음** — 순수
  confound가 만든 유령 신호였던 것으로 판정.

**결론**: 원래 H5의 "quadrant 지문" 결론은 신뢰할 수 없고, role별로 나눈 이 결과가
정확한 버전. agent마다 BUYER/SELLER 각각의 약점 quadrant를 따로 봐야 함.

---

## H6: Agent별 편차/이상치 진단 — **진행중**

**질문**: population-level 검정("몇 개 agent가 유의한가")이 아니라, 이 벤치마크의 원래
목적("특정 모델이 특정 조건에서 무너지는 구멍 진단", `benchmark/CLAUDE.md` 참고)에 맞게
개별 agent 단위 편차를 본다 — 예: 대부분 obvious_aligned에서 baseline bias가 비슷한데
유독 튀는 agent가 있는지, gemma-4-31b-it/kimi-k3처럼 가장 쉬운 조건에서부터 이미
실패하는 agent가 detection 외에 grounding 쪽에도 있는지.

**연관**: claude-opus-4.6, nemotron의 subtle_misaligned n=0 케이스를 여기서 개별적으로
더 파볼 것.

**진행 경위**: gpt-5.5/nemotron의 obvious_aligned 만성 약점을 transcript로 파다가
BUYER/SELLER 역할 비대칭을 발견 → H9로 독립, H5_retrial 촉발(위 참고).

### H6-1: claude-opus-4.6/nemotron의 subtle_misaligned n=0 원인 규명 — **완료**

**정정**: 애초에 "둘 다 n=0"이라는 전제가 틀렸음. H1 원본 데이터를 다시 보면
nemotron은 `n_sm=13`으로 데이터가 있었고 CI가 0을 살짝 걸쳐서 유의하지 않았던
것뿐 — n=0인 건 claude-opus-4.6 하나뿐.

**결과**:
- **claude-opus-4.6**: 64 episode 중 진짜 결함을 인용한 건 5개(7.8%) — 뭔가
  인용한 25개 중 20개는 엉뚱한 걸 인용(할루시네이션). 그 5개조차 "단일 인용 +
  OFFER turn"이라는 H1 기준에 하나도 안 걸려서(`n_single_offer_hit=0`) grounding
  측정이 아예 불가능했음. **원인은 압도적으로 detection 실패** — 측정 방식의
  사각지대는 부수적.
- **nemotron**: 46 episode 중 24개(52%)가 진짜 결함을 정확히 인용 — 오히려 detection이
  꽤 좋음. `n_single_offer_hit=9`로 데이터도 존재. 애초에 이 agent를 "n=0 문제"로
  분류한 것 자체가 착오였음.

**결론**: subtle_misaligned grounding 측정 실패는 claude-opus-4.6에게만 해당하는
문제고, 원인은 순수 detection 실패(진짜 결함을 거의 못 알아챔)임.

**진행 경위**: gpt-5.5/nemotron의 obvious_aligned 만성 약점(H5)을 transcript로 직접
파다가 BUYER/SELLER 역할 비대칭이라는 훨씬 큰 구조적 문제를 발견 → H9로 독립시킴,
H5_retrial 촉발. **원래 H6 후보들(claude-opus-4.6/nemotron n=0 케이스, gpt-4o-mini
할루시네이션 패턴, H5의 CSE+ obvious_misaligned 미스터리)은 아직 착수 안 함** —
H9/H2 재검증 끝나면 이어서 진행.

---

## H7: 결함 유무 자체가 협상 결과에 영향을 주는가 — **미시작**

**질문**: quadrant 비교와는 다른 층위 — 아이템에 결함이 아예 없는 episode(1600개 중
336개)와 결함이 있는 episode 사이에 SE+/AGR+ 등이 예상대로 차이 나는지. 일종의 완전
baseline 체크.

---

## H8: Protocol violation(가격범위/IR 위반) 비율의 agent간 편차 — **미시작**

**계기**: H2 재계산 중 gemini-3.1-pro-preview의 SE+ 극단치를 추적하다가 발견 (위 H2
"첫 시도에서 발견한 문제" 참고). 단발성 사고가 아니라 훨씬 큰 패턴이었음.

**예비 수치** (4 mapping 풀링, 13 LLM agent × 400 episode = 5,200개 기준):
- `price_out_of_bounds` 또는 `ir_violation`이 낀 episode: 532개 (~10%)
- agent별 편차가 매우 큼 — qwen3-vl-32b-instruct 152/400(38%), gpt-4o-mini 118/400(29.5%),
  nemotron 86/400(21.5%), gpt-4o 78/400(19.5%) vs claude-opus-4.6 7/400(1.75%),
  gemini-3.1-pro-preview 5/400(1.25%), gemma-4-31b-it 6/400(1.5%)

**질문**: 이 위반율이 TERMS-BENCH의 4대 진단 축 중 protocol compliance(`CritViol%`)에
해당하는데, agent간 편차(1.25%~38%)가 detection/grounding 축보다 오히려 더 뚜렷한
차별화 포인트일 수 있음. `CritViol%`가 이 편차를 실제로 얼마나 반영하는지(지금 위반
episode 비율과 metrics.py의 CritViol% 정의가 일치하는지)부터 확인 필요. 그 다음 이
위반율이 SE+ 변동성(위 H2 케이스처럼)이나 다른 지표와 상관있는지도 볼 만함.

---

## H9: BUYER/SELLER 역할 비대칭 — 벤치마크 환경 설계 문제 — **진행중**

**계기**: H6에서 gpt-5.5/nemotron의 obvious_aligned 만성 약점을 transcript로 직접 읽다가
발견. BUYER transcript 2건은 상대가 listing price의 4~6배로 첫 제안을 부르고 agent가
라운드 끝까지 근처도 못 가는 반면, SELLER transcript 2건은 거의 다 왔는데(2~7% 차이)
마지막에 못 넘는 전혀 다른 실패 양상을 보임.

**측정 1 — counterpart 첫 제안이 자기 reservation(r_B)에서 얼마나 벗어나는지, R로
정규화, counterpart 역할별 비교** (13 agent 전체 episode 풀링, violation 제외):
- counterpart가 SELLER일 때(=agent가 BUYER): 평균 0.363, 중앙값 0.358, n=2269, **min이
  0.101부터 시작**(한 번도 안 겸손함)
- counterpart가 BUYER일 때(=agent가 SELLER): 평균 0.054, 중앙값 0.022, n=2216
- **7배 차이.** 원인은 `p_min=0`(고정 하한) vs `p_max`(느슨하게 큰 상한, 예: listing
  $550인데 p_max=$4200)의 구조적 비대칭 — SELLER 역할 counterpart의 유리한 쪽 slack
  (`p_max - r_B`)이 BUYER 역할 counterpart의 slack(`r_B - p_min = r_B`)보다 훨씬 큼.
  코드 어디에도(다른 설계 결정들과 달리) 이게 의도됐다는 날짜 붙은 코멘트가 없음 —
  의도치 않은 부작용으로 판단.

**측정 2 — 이 비대칭이 agent 성과에 실제로 반영되는지** (SE+, 13 agent 전체):
- **13/13 전원 BUYER < SELLER, 예외 없음.** 격차는 qwen3-vl-32b-instruct(-0.405, 최대)부터
  gpt-4o-mini(-0.062, 최소, 근데 이건 원래 양쪽 다 낮아서(0.14/0.20))까지 다양.
  이 정도로 완벽하게 전원일치인 패턴은 H1~H8 통틀어 처음.

**측정 3 — quadrant 배정이 role과 균등한지** (13 agent 전체 episode 풀링):
- obvious_aligned: BUYER 60.0%, subtle_aligned: BUYER 38.8%, obvious_misaligned: BUYER
  51.1%(거의 균형), subtle_misaligned: BUYER 35.3% — 25%p까지 벌어짐. quadrant 배정이
  role과 독립적으로 균등하지 않았음 → H5(원래)가 폐기되고 H5_retrial로 대체된 근거.

**agent간 비교 왜곡 여부**: role/opener는 모든 agent에게 동일한 seed로 균등 배정되므로
(`benchmark/CLAUDE.md` "4칸 균등 배정"), **agent 순위 비교는 이 confound로 왜곡되지
않음** — 모두가 동일하게 불리한 조건을 받았기 때문. 다만 (1) 절대적인 SE+ 수치가
전반적으로 낮게 나오고, (2) quadrant처럼 role과 우연히 얽힐 수 있는 다른 축을 role
없이 해석하면 착시가 생김(H5 사례).

**미결 — 다음 확인/결정 필요**:
1. H2(quadrant별 SE+ super-additivity 검정)도 role 안 나누고 계산했음 — 재검증 필요
2. 이걸 실제로 고칠지(p_max를 listing_price에 비례하게 낮추는 등, 데이터 재수집 필요)
   vs 논문에 known limitation으로 명시만 할지 결정 필요 — 마감(8월 말) 감안해서 판단
