"""정식 sanity check 스크립트 (benchmark/CLAUDE.md "검증 방식" 항목, 2026-07-29 정리).

지금까지 kernel.py/env.py/metrics.py 변경 때마다 citing_dummy_agent.py 같은 즉석 스크립트로
눈으로만 확인해왔다. 이 파일은 그 중 "매번 다시 확인해야 하는 불변조건"들을 한 곳에 모아
자동으로 pass/fail을 내는 회귀 게이트다. 두 파트로 나뉜다:

Part A -- TERMS-Bench 충실도: implementation/kernel.py는 논문 PDF 페이지 이미지와 직접 대조
검증된 레퍼런스다(implementation/kernel.py 모듈 docstring, implementation/CLAUDE.md "진행
상황" 참고). benchmark/kernel.py는 그걸 복사+확장(evidence_term 축 추가)한 것인데, 두 파일을
직접 코드 대조해보면 핵심 상수(ALPHA/BETA/GAMMA/LAMBDA*/OMEGA*/ECON_PRESETS 등)와 공식이
전부 문자 그대로 동일하고, 유일한 구조적 차이는 accept_prob에 더해지는 evidence_term
하나뿐이다(2026-07-29 확인, decisions_log.md 참고). 이 스크립트는 "코드가 같다"를 다시
증명하는 게 아니라 -- 그건 이미 사람이 눈으로 대조 완료했다 -- **그 일치가 실행 결과로도
재현되는지, 그리고 앞으로 누가 실수로 어긋나게 고치면 잡히는지**를 확인하는 회귀 가드다.

Part B -- visual/evidence 축 의도 검증: benchmark/kernel.py가 새로 추가한 evidence_term_for/
QUADRANT_BONUS가 benchmark/CLAUDE.md·decisions_log.md에 적어둔 설계 의도(부호, quadrant별
크기, 할루시네이션 무시, 다중 인용시 최댓값, IR 게이트 불가침)를 실제로 지키는지 property
단위로 확인한다. 이건 원 논문에 없는 이 프로젝트 고유 설계라 대조할 외부 레퍼런스가 없다 --
대신 우리 자신이 문서화한 의도를 코드가 지키는지가 기준이다.

실행: `bash run.sh sanity_check.py` 대신 그냥 `.venv/Scripts/python.exe sanity_check.py`
(run.sh는 run_negotiation.py 전용 래퍼라 이 파일엔 안 맞음). 실패가 하나라도 있으면 exit code
1.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # benchmark/
IMPL = ROOT.parent / "implementation"

_failures: list[str] = []
_checked = 0


def check(name: str, cond: bool) -> None:
    global _checked
    _checked += 1
    if not cond:
        _failures.append(name)
        print(f"  FAIL: {name}")


# ---------------------------------------------------------------------------
# 모듈 로더 -- implementation/과 benchmark/이 각자 `from env import ...`를 하므로,
# 그냥 나란히 import하면 'env'라는 이름이 충돌한다. kernel.py를 로드하는 그 순간에만
# sys.modules['env']가 원하는 버전을 가리키게 바꿔치기한다.
# ---------------------------------------------------------------------------


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_pair(root: Path):
    env = _load_module("env", root / "env.py")
    kernel = _load_module(f"kernel_{root.name}", root / "kernel.py")
    del sys.modules["env"]  # 다음 _load_pair(다른 root)가 깨끗하게 다시 잡도록
    return env, kernel


impl_env, impl_kernel = _load_pair(IMPL)
bench_env, bench_kernel = _load_pair(ROOT)


# ---------------------------------------------------------------------------
# Part A -- implementation/kernel.py vs benchmark/kernel.py, evidence_term=0 조건
# ---------------------------------------------------------------------------


def _deterministic_agent_action(turn_idx: int, role_A, r_A: float, p_min: float, p_max: float, env_mod):
    """rng를 전혀 안 쓰는 고정 양보 스케줄 -- 두 loop이 완전히 같은 agent 가격을 보게 만드는
    용도. Part A가 보려는 건 "counterpart 쪽 rng 소비 순서/공식이 어긋나는가"이므로, agent
    쪽 무작위성은 아예 없애서 변수를 하나로 좁힌다. cited_defect_ids=None -> evidence_term은
    항상 0.0 (benchmark 쪽도 이 조건에선 최초 implementation과 동일 공식이 됨)."""
    anchor = p_min if role_A == env_mod.Role.BUYER else p_max
    frac = min(1.0, 0.15 * turn_idx)
    price = anchor + frac * (r_A - anchor)
    kwargs = dict(decision=env_mod.Decision.OFFER, price=price)
    if env_mod is bench_env:
        kwargs["cited_defect_ids"] = None  # implementation.env.Action엔 이 필드가 아예 없음
    return env_mod.Action(**kwargs)


def _run_counterpart_sequence(kernel_mod, env_mod, *, family: str, role_A_name: str, stance: str, opener: str, n_turns: int = 8, seed: int = 12345):
    role_A = env_mod.Role.BUYER if role_A_name == "BUYER" else env_mod.Role.SELLER
    r_A = 120.0 if role_A_name == "BUYER" else 80.0
    r_B = 80.0 if role_A_name == "BUYER" else 120.0

    kwargs = dict(
        regime="overlap", p_min=0.0, p_max=200.0, role_A=role_A, r_A=r_A,
        t_B=env_mod.TypeB(r=r_B, urgency=0.4, stance=stance),
        opener=opener, K=2 * n_turns + 2, harshness=0.6,
    )
    if env_mod is bench_env:
        # Episode.item은 benchmark 쪽에만 있는 필수 필드 -- ground_truth_defects=()라
        # evidence_term_for가 이 테스트에서 항상 0.0을 내는 이중 안전장치(agent가 아예 인용을
        # 안 하는 것과 별개로).
        kwargs["item"] = bench_env.Item(
            category="test", title="t", description="d", listing_price=200.0,
            image_ref="placeholder://x", ground_truth_defects=(),
        )
    episode = env_mod.Episode(**kwargs)

    policy = kernel_mod.make_counterpart_policy(family)
    rng = __import__("random").Random(seed)
    history: list[tuple[int, str, object]] = []
    results = []
    for i in range(1, n_turns + 1):
        agent_action = _deterministic_agent_action(i, role_A, r_A, kwargs["p_min"], kwargs["p_max"], env_mod)
        history.append((2 * i - 1, "agent", agent_action))
        cp_action = policy(episode, history, "counterpart", rng)
        results.append((cp_action.decision.name, cp_action.price, cp_action.sentiment, cp_action.posture))
        history.append((2 * i, "counterpart", cp_action))
    return results


def _sequences_match(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (dec_a, price_a, sent_a, post_a), (dec_b, price_b, sent_b, post_b) in zip(a, b):
        if dec_a != dec_b or sent_a != sent_b or post_a != post_b:
            return False
        if (price_a is None) != (price_b is None):
            return False
        if price_a is not None and not math.isclose(price_a, price_b, rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def check_part_a() -> None:
    print("=== Part A: implementation/kernel.py vs benchmark/kernel.py (evidence_term=0) ===")
    check("두 모듈의 FAMILIES 키셋 일치", set(impl_kernel.FAMILIES.keys()) == set(bench_kernel.FAMILIES.keys()))

    scenarios = 0
    mismatches = []
    for family in impl_kernel.FAMILIES:
        for role_A_name in ("BUYER", "SELLER"):
            for stance in ("conciliatory", "neutral", "aggressive"):
                for opener in ("AgentOpens", "CounterpartOpens"):
                    scenarios += 1
                    impl_res = _run_counterpart_sequence(impl_kernel, impl_env, family=family, role_A_name=role_A_name, stance=stance, opener=opener)
                    bench_res = _run_counterpart_sequence(bench_kernel, bench_env, family=family, role_A_name=role_A_name, stance=stance, opener=opener)
                    if not _sequences_match(impl_res, bench_res):
                        mismatches.append((family, role_A_name, stance, opener))

    check(f"{scenarios}개 시나리오(6 family x 2 role x 3 stance x 2 opener) 전부 turn-by-turn 일치", len(mismatches) == 0)
    for m in mismatches[:5]:
        print(f"    mismatch: family={m[0]} role_A={m[1]} stance={m[2]} opener={m[3]}")


# ---------------------------------------------------------------------------
# Part B -- evidence_term_for / QUADRANT_BONUS가 설계 의도대로 동작하는가
# ---------------------------------------------------------------------------


def _item_with(*defects) -> object:
    return bench_env.Item(category="c", title="t", description="d", listing_price=100.0, image_ref="x", ground_truth_defects=tuple(defects))


#  2026-07-29 재조정(kernel.py 참고: ALPHA*delta_bar의 현실적 상한 ~1.0~1.5에 맞춘 값,
#  비율 1:3:0.5:4는 원래 설계 그대로 유지)의 결과값을 여기 하드코딩해둔다 -- 아래
#  check_part_b의 "evidence_term_for가 QUADRANT_BONUS를 올바르게 읽는가" 체크는
#  QUADRANT_BONUS 자체를 그대로 재사용하기 때문에(자기참조), 값 자체가 실수로
#  바뀌어도 못 잡는다. 이 상수가 있어야 "숫자 자체가 의도한 값인가"까지 잡힌다.
EXPECTED_QUADRANT_BONUS = {
    "obvious_aligned": 0.3,
    "subtle_aligned": 0.9,
    "obvious_misaligned": 0.15,
    "subtle_misaligned": 1.2,
}
EXPECTED_DEFAULT_BONUS = 0.6


def check_part_b() -> None:
    print("=== Part B: evidence_term_for / QUADRANT_BONUS 설계 의도 검증 ===")
    Defect, Action, Decision, Role = bench_env.Defect, bench_env.Action, bench_env.Decision, bench_env.Role
    evidence_term_for = bench_kernel.evidence_term_for
    QUADRANT_BONUS = bench_kernel.QUADRANT_BONUS

    check("QUADRANT_BONUS 값이 2026-07-29 재조정 표와 정확히 일치", QUADRANT_BONUS == EXPECTED_QUADRANT_BONUS)
    check("_DEFAULT_BONUS가 재조정된 값(0.6)과 일치", bench_kernel._DEFAULT_BONUS == EXPECTED_DEFAULT_BONUS)

    real_defect = Defect(id="scratch_0", description="d", price_impact=5.0, quadrant="obvious_aligned", defect_type="scratch")
    item = _item_with(real_defect)

    fake_cite = Action(decision=Decision.OFFER, price=90.0, cited_defect_ids=("missing_part",))  # item엔 없음
    check("할루시네이션(실제 없는 결함 인용) -> evidence_term=0.0", evidence_term_for(fake_cite, item, Role.BUYER) == 0.0)

    no_cite = Action(decision=Decision.OFFER, price=90.0, cited_defect_ids=None)
    check("인용 없음 -> evidence_term=0.0", evidence_term_for(no_cite, item, Role.BUYER) == 0.0)

    for quadrant, bonus in QUADRANT_BONUS.items():
        d = Defect(id="x_0", description="d", price_impact=5.0, quadrant=quadrant, defect_type="x")
        it = _item_with(d)
        cite = Action(decision=Decision.OFFER, price=90.0, cited_defect_ids=("x",))
        check(f"{quadrant}: BUYER 인용 -> +{bonus} (가격 근거 강화)", evidence_term_for(cite, it, Role.BUYER) == bonus)
        check(f"{quadrant}: SELLER 인용 -> -{bonus} (가격 근거 약화)", evidence_term_for(cite, it, Role.SELLER) == -bonus)

    d1 = Defect(id="a_0", description="d", price_impact=1.0, quadrant="obvious_misaligned", defect_type="a")
    d2 = Defect(id="b_0", description="d", price_impact=1.0, quadrant="subtle_misaligned", defect_type="b")
    it2 = _item_with(d1, d2)
    multi_cite = Action(decision=Decision.OFFER, price=90.0, cited_defect_ids=("a", "b"))
    check(
        "한 턴에 여러 결함 인용 -> 최댓값 quadrant(subtle_misaligned) 채택",
        evidence_term_for(multi_cite, it2, Role.BUYER) == QUADRANT_BONUS["subtle_misaligned"],
    )

    d_none = Defect(id="c_0", description="d", price_impact=1.0, quadrant=None, defect_type="c")
    it3 = _item_with(d_none)
    cite_none = Action(decision=Decision.OFFER, price=90.0, cited_defect_ids=("c",))
    check("quadrant=None(미태깅) -> _DEFAULT_BONUS로 폴백", evidence_term_for(cite_none, it3, Role.BUYER) == bench_kernel._DEFAULT_BONUS)

    check(
        "IR 게이트: delta_bar<0이면 evidence_term이 아무리 커도 accept_prob=0.0",
        bench_kernel.accept_prob(-0.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, evidence_term=100.0) == 0.0,
    )
    check(
        "delta_bar>=0이면 evidence_term=0일 때 기존(implementation) 공식과 동치",
        math.isclose(
            bench_kernel.accept_prob(0.1, 0.5, 0.3, 0.0, 0.0, 0.0, 0.4, evidence_term=0.0),
            impl_kernel.accept_prob(0.1, 0.5, 0.3, 0.0, 0.0, 0.0, 0.4),
        ),
    )


def main() -> None:
    check_part_a()
    print()
    check_part_b()
    print()
    print(f"{_checked}개 체크 중 {len(_failures)}개 실패")
    if _failures:
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("전부 통과.")


if __name__ == "__main__":
    main()
