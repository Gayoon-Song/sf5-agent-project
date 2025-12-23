# ============================================================
# 12R_RULEBOOK_EXPLAIN_LOCAL_UID150_v3.py  (FULL REPLACEMENT • ASCII-safe)
#
# 역할
#   - 11R 보정 패치 CSV 기반 WHY 자동 생성 + RULE MASTER 테이블 생성
#
# 입출력
#   - 입력: 11R_GROUND_TRUTH/S_PATCH_TIME_COST_CALIB_UID150.csv (기본)
#   - 출력: 12R_RULEBOOK/S_PATCH_RULEBOOK_EXPLAIN_UID150.csv
#
# 실행 예
#   - 기본: python 12R_RULEBOOK_EXPLAIN_LOCAL_UID150_v3.py
#   - 지정: python 12R_RULEBOOK_EXPLAIN_LOCAL_UID150_v3.py --in_patch X.csv
# ============================================================

import os, json, math, textwrap, warnings, re, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------- ASCII-safe print ----------
def _s(x: str) -> str:
    return str(x).encode("ascii", "ignore").decode("ascii")
def _print(msg: str):
    print(_s(msg))

# ---------- Root resolver ----------
def resolve_root(root_cli=None):
    if root_cli: return os.path.abspath(root_cli)
    env_root = os.environ.get("SF5_ROOT", "").strip()
    if env_root: return os.path.abspath(env_root)
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

ROOT_DIR = resolve_root()

# ---------- find helpers ----------
def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def get_col(df, *names, default=None):
    for n in names:
        if n in df.columns:
            return n
    return default

def to_uid3(x):
    try:
        s = str(x).strip()
        if s.lower() == "nan": return None
        if len(s) == 3 and s.isdigit(): return s
        n = int(float(s))
        return f"{n:03d}"
    except:
        return None

def fmt_min_to_hm(m):
    if m is None or (isinstance(m, float) and np.isnan(m)): return "0분"
    m = float(m)
    if m < 0: m = 0
    h = int(m // 60); mm = int(round(m - h*60))
    return f"{h}시간 {mm}분" if h > 0 else f"{mm}분"

def safe_float(x, default=0.0):
    try:
        if pd.isna(x): return default
        return float(x)
    except:
        return default

# ---------- scale classifiers ----------
def classify_delta_scale(abs_dmax):
    v = safe_float(abs_dmax)
    if v < 0.10: return "TINY"
    if v < 0.30: return "SMALL"
    if v < 0.80: return "MEDIUM"
    if v < 2.00: return "LARGE"
    return "HUGE"

def classify_area_scale(area):
    v = safe_float(area)
    if v < 20: return "TINY"
    if v < 100: return "SMALL"
    if v < 500: return "MEDIUM"
    if v < 2000: return "LARGE"
    return "HUGE"

def classify_diag_scale(diag):
    v = safe_float(diag)
    if v < 5: return "TINY"
    if v < 15: return "SMALL"
    if v < 40: return "MEDIUM"
    if v < 100: return "LARGE"
    return "HUGE"

def _read_csv_flex(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--in_patch", default=None)
    ap.add_argument("--out_dir", default=None)
    # pipeline compat
    ap.add_argument("--uid", default=None)
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    args, _unknown = ap.parse_known_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    BASE = ROOT_DIR
    OUT_DIR = args.out_dir if args.out_dir else os.path.join(BASE, "12R_RULEBOOK")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_PATCH_EXPLAIN = os.path.join(OUT_DIR, "S_PATCH_RULEBOOK_EXPLAIN_UID150.csv")
    OUT_PATCH_EXPLAIN_ALIAS = os.path.join(OUT_DIR, "S_PATCH_RULEBOOK_EXPLAIN.csv")
    OUT_RULE_MASTER = os.path.join(OUT_DIR, "S_RULE_MASTER_TABLE_UID150.csv")
    OUT_RULE_MASTER_ALIAS = os.path.join(OUT_DIR, "S_RULE_MASTER_TABLE.csv")

    _print("====================================")
    _print("===== 12R RULEBOOK/WHY (LOCAL v3) =====")
    _print("BASE_DIR : " + BASE)
    _print("OUT_DIR  : " + OUT_DIR)

    # 1. 입력 자동 탐색
    # 파이프라인에서 11R이 생성했을 것으로 예상되는 파일
    IN_CANDIDATES = []
    if args.uid:
        # 단일 UID 처리 중이라면, 해당 UID 파일이 있는지 확인
        u_num = re.findall(r"\d+", args.uid)[-1]
        u3 = f"{int(u_num):03d}"
        single_path = os.path.join(BASE, "11R_GROUND_TRUTH", f"S_PATCH_TIME_COST_CALIB_UID{u3}.csv")
        IN_CANDIDATES.append(single_path)

    IN_CANDIDATES.extend([
        os.path.join(BASE, "11R_GROUND_TRUTH", "S_PATCH_TIME_COST_CALIB_UID150.csv"),
        os.path.join(BASE, "11R_GROUND_TRUTH", "S_PATCH_TIME_COST_CALIB_UID.csv"),
    ])

    if args.in_patch and os.path.isfile(args.in_patch):
        IN_PATCH = args.in_patch
    else:
        IN_PATCH = find_first_existing(IN_CANDIDATES)
        if IN_PATCH is None:
            # fallback
            target_name = "S_PATCH_TIME_COST_CALIB_UID"
            hits = []
            for r, d, f in os.walk(BASE):
                for fn in f:
                    if target_name in fn and fn.endswith(".csv"):
                        hits.append(os.path.join(r, fn))
            if hits:
                IN_PATCH = sorted(hits)[0]
            else:
                _print("[12R][WARN] 11R input not found. Skipping 12R.")
                return

    _print("[12R] INPUT_PATCH_CALIB : " + IN_PATCH)

    # 2. 로드
    df = _read_csv_flex(IN_PATCH)
    _print("[12R] CSV shape: " + str(df.shape))

    # 3. UID 정규화
    uid_col = get_col(df, "uid", "UID")
    if uid_col is None:
        # UID 컬럼이 없으면 생성 시도 (파일명 등에서)
        # 하지만 UID 정보 없이는 그룹핑이 어려움.
        # 단일 파일 처리인 경우 args.uid 사용
        if args.uid:
            u_num = re.findall(r"\d+", args.uid)[-1]
            u3 = f"{int(u_num):03d}"
            df["uid"] = u3
            uid_col = "uid"
        else:
            raise KeyError("[12R][ERROR] uid 컬럼을 찾지 못했습니다. (uid/UID)")
            
    df["uid"] = df[uid_col].apply(to_uid3)
    df = df[~df["uid"].isna()].copy()

    # 4. 파생/보정 특성 생성 (기존 로직 유지)
    if "delta_dir" not in df.columns:
        sign_col = get_col(df, "sign")
        if sign_col:
            def _dir_from_sign(s):
                s = str(s).lower()
                if "thick" in s or "plus" in s or s in ["1", "+1", "add"]: return "thicker"
                if "thin" in s or "minus" in s or s in ["-1", "remove"]: return "thinner"
                return "mixed"
            df["delta_dir"] = df[sign_col].apply(_dir_from_sign)
        else:
            davg = get_col(df, "delta_avg")
            if davg:
                df["delta_dir"] = df[davg].apply(lambda v: "thicker" if safe_float(v)>0 else ("thinner" if safe_float(v)<0 else "mixed"))
            else:
                df["delta_dir"] = "mixed"

    if "abs_delta_max" not in df.columns:
        dmax = get_col(df, "delta_max")
        df["abs_delta_max"] = df[dmax].abs() if dmax else 0.0
    if "abs_delta_avg" not in df.columns:
        davg = get_col(df, "delta_avg")
        df["abs_delta_avg"] = df[davg].abs() if davg else 0.0

    # BBox spans
    bbox_cols = ["bbox_xmin","bbox_xmax","bbox_ymin","bbox_ymax","bbox_zmin","bbox_zmax"]
    has_bbox = all(c in df.columns for c in bbox_cols)
    if has_bbox:
        if "bbox_z_span" not in df.columns:
            df["bbox_z_span"] = (df["bbox_zmax"] - df["bbox_zmin"]).abs()
        if "bbox_xy_area_ratio" not in df.columns:
            span_x = (df["bbox_xmax"] - df["bbox_xmin"]).abs()
            span_y = (df["bbox_ymax"] - df["bbox_ymin"]).abs()
            span_z = df["bbox_z_span"]
            df["bbox_xy_area_ratio"] = (span_x * span_y) / (span_z.replace(0,1e-6))

    # Normals
    nx = get_col(df, "normal_avg_x"); ny = get_col(df, "normal_avg_y"); nz = get_col(df, "normal_avg_z")
    if "normal_main_axis" not in df.columns:
        if nx and ny and nz:
            def _main_axis(row):
                ax = abs(safe_float(row[nx])); ay = abs(safe_float(row[ny])); az = abs(safe_float(row[nz]))
                mx = max(ax, ay, az)
                if mx == ax: return "X"
                if mx == ay: return "Y"
                return "Z"
            df["normal_main_axis"] = df.apply(_main_axis, axis=1)
        else:
            df["normal_main_axis"] = "Z"

    if "is_visible_zplus" not in df.columns:
        if nz: df["is_visible_zplus"] = df[nz].apply(lambda v: safe_float(v) > 0.3)
        else: df["is_visible_zplus"] = False

    if "side_hint" not in df.columns:
        cz = get_col(df, "center_z")
        if cz and nz:
            def _side_hint(row):
                zc = safe_float(row[cz]); nzv = safe_float(row[nz])
                if nzv > 0.2 and zc >= 0: return "CAVITY"
                if nzv < -0.2 and zc < 0: return "CORE"
                if "bbox_z_span" in row and safe_float(row["bbox_z_span"]) > 20: return "SLIDE/LIFTER"
                return "UNKNOWN"
            df["side_hint"] = df.apply(_side_hint, axis=1)
        else:
            df["side_hint"] = "UNKNOWN"

    # 5. 설명 맵 (기존 로직 유지)
    geom_map = {
        "RIB_ADD": "리브가 신규로 추가되거나 리브 상단/측면 두께가 증가한 설계변경입니다.",
        "RIB_TOP": "리브 상단부 형상(두께/높이)이 변경된 설계변경입니다.",
        "RIB_REMOVE": "리브가 삭제되거나 리브 두께가 감소한 설계변경입니다.",
        "BOSS_ADD": "보스(체결/지지부)가 추가 또는 확대된 설계변경입니다.",
        "BOSS_REMOVE": "보스가 삭제 또는 축소된 설계변경입니다.",
        "HOLE_ADD": "관통/블라인드 홀 가공부가 신규 생성된 설계변경입니다.",
        "HOLE_REMOVE": "홀 가공부가 삭제 또는 축소된 설계변경입니다.",
        "POCKET_FLOOR": "포켓(캐비티) 바닥부 형상·깊이가 변경된 설계변경입니다.",
        "POCKET_SIDE": "포켓 측면 벽 형상이 변경된 설계변경입니다.",
        "CYL_SIDE": "원통측(보어/샤프트 접촉면) 두께가 변경된 설계변경입니다.",
        "PLANE_FACE": "평면부의 국부 두께/형상이 변경된 설계변경입니다.",
        "FREEFORM_SURF": "자유곡면 영역의 형상/두께가 변경된 설계변경입니다.",
        "CURVED_SURF": "곡면 영역의 형상/두께가 변경된 설계변경입니다.",
        "PARTING_CHANGE": "파팅라인 또는 파팅 인접 영역 형상이 변경된 설계변경입니다.",
        "INSERT_ADD": "인서트/코어 삽입부가 추가 또는 확대된 설계변경입니다.",
        "INSERT_REMOVE": "인서트/코어 삽입부가 삭제 또는 축소된 설계변경입니다.",
    }
    proc_map = {
        "MILLING": "NC 밀링(조삭) 가공이 주 공정으로 판단됩니다.",
        "DRILLING": "드릴/리머/탭 등 홀 계열 가공이 주 공정으로 판단됩니다.",
        "EDM": "방전(EDM) 가공이 필요하다고 판단됩니다.",
        "WELDING": "용접(살덧댐/보강) 후 재가공이 필요한 설계변경입니다.",
        "POLISHING": "연마/사상(폴리싱) 공정이 필요합니다.",
        "HAND_FINISH": "수작업 사상·수정 공정이 포함될 가능성이 큽니다.",
    }
    machine_map = {
        "NC_SMALL": "소형 3축/4축 NC 설비",
        "NC_MID": "중형 NC 설비",
        "NC_LARGE": "대형 NC 설비",
        "NC_5AX": "5축 NC 설비",
        "EDM_SMALL": "소형 방전 설비",
        "EDM_LARGE": "대형 방전 설비",
    }
    setup_map = {
        "ONE_SIDE": "단면 셋업으로 가공 가능한 수준입니다.",
        "TWO_SIDE": "양면 셋업 또는 2회 이상 재클램핑이 필요합니다.",
        "MULTI_SIDE": "다각 셋업/회전·분해 셋업이 필요한 복합 가공입니다.",
    }
    diff_map = {
        "EASY": "난이도는 낮은 편입니다.",
        "NORMAL": "난이도는 보통 수준입니다.",
        "HARD": "난이도가 높아 공정 리스크가 존재합니다.",
        "VERY_HARD": "난이도가 매우 높아 공정/품질 협의가 선행되어야 합니다.",
    }
    nctype_map = {
        "3AXIS": "3축 NC로 가공 가능합니다.",
        "5AXIS": "5축 가공이 유리하거나 필요합니다.",
        "MANUAL": "수작업 보정/사상이 동반될 수 있습니다.",
    }

    # 컬럼 매핑
    geom_col = get_col(df, "geom_class_auto", default=None)
    proc_col = get_col(df, "process_family_auto", default=None)
    mach_col = get_col(df, "machine_group_auto", "machine_group", default=None)
    setup_col = get_col(df, "setup_type_auto", "setup_type", default=None)
    diff_col = get_col(df, "difficulty_auto", "difficulty", default=None)
    nctype_col = get_col(df, "nc_type_auto", "nc_type", default=None)
    ptype_col = get_col(df, "patch_type", default=None)

    area_col = get_col(df, "area_est"); diag_col = get_col(df, "diag_mm")
    time_pred_col = get_col(df, "time_min_pred", default=None)
    cost_pred_col = get_col(df, "cost_krw_pred", default=None)
    time_calib_col = get_col(df, "time_min_calib", default=None)
    cost_calib_col = get_col(df, "cost_krw_calib", default=None)
    time_scale_uid_col = get_col(df, "time_scale_uid", default=None)
    cost_scale_uid_col = get_col(df, "cost_scale_uid", default=None)

    # 설명 생성 함수들 (기존 로직 그대로)
    def explain_geom(row):
        geom_key = str(row.get(geom_col, "UNKNOWN")) if geom_col else "UNKNOWN"
        geom_desc = geom_map.get(geom_key, f"형상 분류상 '{geom_key}' 계열의 설계변경 패치입니다.")
        ptype = str(row.get(ptype_col, "pair_patch")) if ptype_col else "pair_patch"
        if "added" in ptype: ptype_desc = "기존 형상에 **신규 면이 추가**된 유형입니다."
        elif "removed" in ptype: ptype_desc = "기존 형상이 **삭제/제거**된 유형입니다."
        else: ptype_desc = "기존 면에서 **국부적으로 치수가 변경**된 유형입니다."
        ddir = row.get("delta_dir", "mixed")
        abs_dmax = safe_float(row.get("abs_delta_max", 0))
        abs_davg = safe_float(row.get("abs_delta_avg", 0))
        dsign_desc = {"thicker":"살붙임(+) 방향", "thinner":"살빼기(-) 방향", "mixed":"혼합/복합 방향"}.get(ddir, "복합 방향")
        delta_scale = classify_delta_scale(abs_dmax)
        area_scale = classify_area_scale(row.get(area_col, 0) if area_col else 0)
        diag_scale = classify_diag_scale(row.get(diag_col, 0) if diag_col else 0)
        side = str(row.get("side_hint","UNKNOWN"))
        vis = bool(row.get("is_visible_zplus", False))
        vis_desc = "가시면(외관면) 가능성이 있어 품질/외관 리스크 고려가 필요합니다." if vis else "비가시면(내부/코어측) 성격이 강해 기능·조립 위주로 관리됩니다."
        side_desc = {"CAVITY":"캐비티(CAVITY) 측으로 추정됩니다.", "CORE":"코어(CORE) 측으로 추정됩니다.", "SLIDE/LIFTER":"슬라이드/리프터 등 가동 코어 계열일 수 있습니다.", "UNKNOWN":"코어/캐비티 구분은 추가 확인이 필요합니다."}.get(side, side)
        return f"{geom_desc} {ptype_desc} 두께 변화는 **{dsign_desc}**이며, |Δmax|≈{abs_dmax:.3f} mm, |Δavg|≈{abs_davg:.3f} mm. 스케일 분류는 Δ:{delta_scale}, 면적:{area_scale}, 범위(diag):{diag_scale}. {side_desc} {vis_desc}"

    def explain_process(row):
        pf = str(row.get(proc_col, "UNKNOWN")) if proc_col else "UNKNOWN"
        pf_desc = proc_map.get(pf, f"공정 분류상 '{pf}' 계열 작업이 예상됩니다.")
        mach = str(row.get(mach_col, "UNKNOWN")) if mach_col else "UNKNOWN"
        mach_desc = machine_map.get(mach, f"설비 그룹은 '{mach}'로 추정됩니다.")
        setup = str(row.get(setup_col, "UNKNOWN")) if setup_col else "UNKNOWN"
        setup_desc = setup_map.get(setup, f"셋업 유형은 '{setup}'로 추정됩니다.")
        diffi = str(row.get(diff_col, "UNKNOWN")) if diff_col else "UNKNOWN"
        diff_desc = diff_map.get(diffi, f"난이도 분류는 '{diffi}'입니다.")
        nct = str(row.get(nctype_col, "UNKNOWN")) if nctype_col else "UNKNOWN"
        nct_desc = nctype_map.get(nct, f"가공 방식은 '{nct}' 형태로 판단됩니다.")
        side = str(row.get("side_hint","UNKNOWN"))
        side_proc_hint = ""
        if "SLIDE" in side or "LIFTER" in side: side_proc_hint = "가동 코어(슬라이드/리프터) 가능성 → **분해 후 단품 가공/재조립** 포함 가능."
        elif side == "CAVITY": side_proc_hint = "캐비티측 → **사상·광택·수지흐름 영향 검토** 병행 권장."
        elif side == "CORE": side_proc_hint = "코어측 → **취출/간섭/언더컷 영향** 추가 검토 필요."
        seq_hint = ""
        if pf == "WELDING": seq_hint = "일반적으로 **용접 → 조삭(NC/EDM) → 정삭 → 사상** 순입니다."
        elif pf == "EDM": seq_hint = "방전 공정은 **전극 수량·형상 복잡도**에 따른 변동성이 큽니다."
        elif pf == "DRILLING": seq_hint = "홀 가공은 위치 정합·각도·깊이에 민감하며, 탭/리머 포함 시 공수가 증가합니다."
        return f"{pf_desc} 예상 설비는 {mach_desc}. {setup_desc} {diff_desc} {nct_desc} {seq_hint} {side_proc_hint}"

    def explain_scale(row):
        abs_dmax = safe_float(row.get("abs_delta_max", 0))
        area = safe_float(row.get(area_col, 0) if area_col else 0)
        diag = safe_float(row.get(diag_col, 0) if diag_col else 0)
        delta_scale = classify_delta_scale(abs_dmax)
        area_scale = classify_area_scale(area)
        diag_scale = classify_diag_scale(diag)
        if delta_scale in ["LARGE","HUGE"] or area_scale in ["LARGE","HUGE"]: risk = "범위가 커 **가공/사상 시간 증가·품질 리스크**가 높습니다."
        elif delta_scale in ["MEDIUM"] or area_scale in ["MEDIUM"]: risk = "중간 규모 설변으로 **표준 가공+부분 사상** 수준이 예상됩니다."
        else: risk = "소규모/국부 설변으로 **단순 보정 가공** 성격이 강합니다."
        return f"Δ:{delta_scale}/면적:{area_scale}/범위:{diag_scale} → {risk}"

    def explain_time(row):
        tp = safe_float(row.get(time_pred_col, 0)) if time_pred_col else 0.0
        tc = safe_float(row.get(time_calib_col, tp)) if time_calib_col else tp
        sc = safe_float(row.get(time_scale_uid_col, 1.0)) if time_scale_uid_col else 1.0
        base = f"모델 예측 기준 **{tp:.2f}분**."
        if time_calib_col and time_scale_uid_col:
            return f"{base} UID 스케일(time_scale_uid≈{sc:.3f}) 적용 후 **{tc:.2f}분({fmt_min_to_hm(tc)})**. 규칙: time_calib = time_pred × scale."
        return f"{base} (실측 기반 보정계수 없음 → 현 값이 최종)."

    def explain_cost(row):
        cp = safe_float(row.get(cost_pred_col, 0)) if cost_pred_col else 0.0
        cc = safe_float(row.get(cost_calib_col, cp)) if cost_calib_col else cp
        sc = safe_float(row.get(cost_scale_uid_col, 1.0)) if cost_scale_uid_col else 1.0
        base = f"모델 예측 기준 **{cp:,.1f}원**."
        if cost_calib_col and cost_scale_uid_col:
            return f"{base} UID 스케일(cost_scale_uid≈{sc:.3f}) 적용 후 **{cc:,.1f}원**. 규칙: cost_calib = cost_pred × scale."
        return f"{base} (실측 기반 보정계수 없음 → 현 값이 최종)."

    # WHY 컬럼 생성
    _print("[12R] WHY 생성 ...")
    df["why_geom_ko"]    = df.apply(explain_geom, axis=1)
    df["why_process_ko"] = df.apply(explain_process, axis=1)
    df["why_scale_ko"]   = df.apply(explain_scale, axis=1)
    df["why_time_ko"]    = df.apply(explain_time, axis=1)
    df["why_cost_ko"]    = df.apply(explain_cost, axis=1)
    df["why_total_ko"]   = df.apply(lambda r: "\n".join([r["why_geom_ko"], r["why_process_ko"], r["why_scale_ko"], r["why_time_ko"], r["why_cost_ko"]]), axis=1)

    # 저장: PATCH WHY
    # 온라인 모드라면 별도 파일명 고려 가능하나, 12R은 보통 전체 파일을 참조하는 KB 구축 전 단계이므로
    # 여기서는 일단 전체 파일(또는 입력과 동일 이름)로 저장.
    # 만약 온라인 모드 전용이라면 입력 경로의 _EXPLAIN.csv 로 저장하는 게 안전.
    
    if args.uid and args.in_patch:
        # 단일 UID 처리 중
        base, ext = os.path.splitext(args.in_patch)
        out_explain = base.replace("_CALIB", "_EXPLAIN") + ".csv"
        df.to_csv(out_explain, index=False, encoding="utf-8-sig")
        _print("[12R] SAVE single: " + out_explain)
        
        # 룰 마스터는 스킵하거나 전체 갱신 (여기서는 생략)
        
    else:
        # 배치 모드
        df.to_csv(OUT_PATCH_EXPLAIN, index=False, encoding="utf-8-sig")
        df.to_csv(OUT_PATCH_EXPLAIN_ALIAS, index=False, encoding="utf-8-sig")
        _print("[12R] SAVE: " + OUT_PATCH_EXPLAIN)
        _print("[12R] SAVE: " + OUT_PATCH_EXPLAIN_ALIAS)

        # RULE MASTER 생성 (배치일 때만 유의미)
        _print("[12R] RULE MASTER 생성 ...")
        key_cols = []
        for c in [geom_col, proc_col, mach_col, setup_col, diff_col, nctype_col, "delta_dir", "side_hint"]:
            if c and c in df.columns: key_cols.append(c)
        if key_cols:
            df_sorted = df.sort_values(by=area_col, ascending=False) if area_col else df.copy()
            rep_rows = df_sorted.groupby(key_cols, dropna=False).head(1).copy()
            rep_rows = rep_rows.rename(columns={
                "why_geom_ko":"rule_geom_ko", "why_process_ko":"rule_process_ko",
                "why_scale_ko":"rule_scale_ko", "why_time_ko":"rule_time_ko",
                "why_cost_ko":"rule_cost_ko", "why_total_ko":"rule_full_ko",
            })
            rule_cols = key_cols + ["rule_geom_ko","rule_process_ko","rule_scale_ko","rule_time_ko","rule_cost_ko","rule_full_ko"]
            # 컬럼 존재 여부 체크
            valid_rule_cols = [c for c in rule_cols if c in rep_rows.columns]
            rep_rows = rep_rows[valid_rule_cols].reset_index(drop=True)
            rep_rows.to_csv(OUT_RULE_MASTER, index=False, encoding="utf-8-sig")
            rep_rows.to_csv(OUT_RULE_MASTER_ALIAS, index=False, encoding="utf-8-sig")
            _print("[12R] SAVE: " + OUT_RULE_MASTER)

    _print("===== 12R DONE =====")

if __name__ == "__main__":
    main()