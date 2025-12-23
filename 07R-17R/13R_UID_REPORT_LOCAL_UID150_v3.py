# ============================================================
# 13R_UID_REPORT_LOCAL_UID150_v3.py  (FULL REPLACEMENT • ASCII-safe)
#
# 역할
#   - 11R 보정 패치 CSV + 12R WHY + (선택) 17R GT를 결합하여
#     UID 단위 요약 리포트(S_UID_REPORT.csv) 생성
#
# 사용 예
#   - 기본: python 13R_UID_REPORT_LOCAL_UID150_v3.py
# ============================================================

import os, re, argparse
import numpy as np
import pandas as pd

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

# ---------- helpers ----------
def first_existing_path(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def read_csv_flex(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")

def norm_uid_value(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = re.sub(r"[^0-9]", "", str(x).strip())
    return s.zfill(3) if s else np.nan

def fmt_time_min_to_str(m):
    if m is None or (isinstance(m, float) and np.isnan(m)): return ""
    m = float(m)
    if m < 0: m = 0.0
    h = int(m // 60); mm = int(round(m - h*60))
    return f"{h}시간 {mm}분" if h > 0 else f"{mm}분"

def fmt_cost_to_str(c):
    if c is None or (isinstance(c, float) and np.isnan(c)): return ""
    try: return f"{int(round(float(c))):,}원"
    except: return ""

def ensure_patch_id(df, uid_col="uid"):
    pid = first_existing_col(df, ["patch_global_id", "patch_id", "patch_idx"])
    if pid is not None:
        df["patch_global_id"] = df[pid].astype(int)
        return df
    # auto assign
    df = df.copy()
    df["__row__"] = np.arange(len(df))
    df["_uid_for_pid_"] = df[uid_col].astype(str)
    df["patch_global_id"] = (
        df.sort_values(["_uid_for_pid_", "__row__"])
          .groupby("_uid_for_pid_").cumcount()
          .astype(int)
    )
    df.drop(columns=["__row__", "_uid_for_pid_"], inplace=True)
    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--patch_calib", default=None)
    ap.add_argument("--patch_why",   default=None)
    ap.add_argument("--gt_uid",      default=None)
    ap.add_argument("--out_dir",     default=None)
    # pipeline compat
    ap.add_argument("--uid", default=None)
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    args, _unknown = ap.parse_known_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    BASE = ROOT_DIR
    OUT_DIR = args.out_dir if args.out_dir else os.path.join(BASE, "13R_UID_REPORT")
    os.makedirs(OUT_DIR, exist_ok=True)

    _print("====================================")
    _print("===== 13R UID Report (LOCAL v3) =====")
    _print("BASE_DIR : " + BASE)
    _print("OUT_DIR  : " + OUT_DIR)

    # 1. 입력 자동 탐색
    calib_paths = [args.patch_calib] if args.patch_calib else []
    # 단일 UID 처리 중이라면 해당 파일 우선
    if args.uid:
        u_num = re.findall(r"\d+", args.uid)[-1]
        u3 = f"{int(u_num):03d}"
        calib_paths.append(os.path.join(BASE, "11R_GROUND_TRUTH", f"S_PATCH_TIME_COST_CALIB_UID{u3}.csv"))

    calib_paths.extend([
        os.path.join(BASE, "11R_GROUND_TRUTH", "S_PATCH_TIME_COST_CALIB_UID150.csv"),
        os.path.join(BASE, "11R_GROUND_TRUTH", "S_PATCH_TIME_COST_CALIB_UID.csv"),
    ])
    
    why_paths = [
        args.patch_why,
        os.path.join(BASE, "12R_RULEBOOK", "S_PATCH_RULEBOOK_EXPLAIN_UID150.csv"),
        os.path.join(BASE, "12R_RULEBOOK", "S_PATCH_RULEBOOK_EXPLAIN.csv"),
    ]
    gt_paths = [
        args.gt_uid,
        os.path.join(BASE, "17R_DATA", "S_UID_GT_001_150.csv"),
    ]

    CALIB_CSV = first_existing_path(calib_paths)
    WHY_CSV   = first_existing_path(why_paths)
    GT_CSV    = first_existing_path(gt_paths)

    if CALIB_CSV is None:
        _print("[13R][ERROR] 11R input CSV not found.")
        return

    # 2. 로딩
    df_calib = read_csv_flex(CALIB_CSV)
    df_why   = read_csv_flex(WHY_CSV) if WHY_CSV else pd.DataFrame()
    df_gt    = read_csv_flex(GT_CSV) if GT_CSV else pd.DataFrame()
    have_gt  = not df_gt.empty

    # 3. UID 정규화
    uid_col_calib = first_existing_col(df_calib, ["uid","UID","uid_num","uid_raw"])
    if not uid_col_calib:
        # UID 컬럼 없으면 단일 UID로 간주하거나 생성
        if args.uid:
            u_num = re.findall(r"\d+", args.uid)[-1]
            df_calib["uid"] = f"{int(u_num):03d}"
        else:
            raise KeyError("[13R] uid column not found in CALIB")
    else:
        df_calib["uid"] = df_calib[uid_col_calib].apply(norm_uid_value)

    if not df_why.empty:
        uid_col_why = first_existing_col(df_why, ["uid","UID","uid_num"])
        if uid_col_why: df_why["uid"] = df_why[uid_col_why].apply(norm_uid_value)
    
    if have_gt:
        uid_col_gt = first_existing_col(df_gt, ["uid","UID","uid_num"])
        if uid_col_gt: df_gt["uid"] = df_gt[uid_col_gt].apply(norm_uid_value)

    # 4. Patch ID
    df_calib = ensure_patch_id(df_calib, uid_col="uid")
    if not df_why.empty:
        df_why = ensure_patch_id(df_why, uid_col="uid")

    # 5. 주요 컬럼 선택
    time_pred_col  = first_existing_col(df_calib, ["time_min_pred","time_min_auto"])
    cost_pred_col  = first_existing_col(df_calib, ["cost_krw_pred","cost_krw_auto"])
    time_calib_col = first_existing_col(df_calib, ["time_min_calib"]) or time_pred_col
    cost_calib_col = first_existing_col(df_calib, ["cost_krw_calib"]) or cost_pred_col
    
    time_scale_col = first_existing_col(df_calib, ["time_scale_uid"])
    cost_scale_col = first_existing_col(df_calib, ["cost_scale_uid"])
    if not time_scale_col: df_calib["time_scale_uid"] = 1.0; time_scale_col="time_scale_uid"
    if not cost_scale_col: df_calib["cost_scale_uid"] = 1.0; cost_scale_col="cost_scale_uid"

    # 6. WHY 병합
    if not df_why.empty:
        why_col = first_existing_col(df_why, ["why_total_ko","why_total"])
        if not why_col: df_why["why_total_ko"] = ""; why_col="why_total_ko"
        df_patch = df_calib.merge(df_why[["uid","patch_global_id",why_col]], on=["uid","patch_global_id"], how="left")
    else:
        df_patch = df_calib.copy()
        df_patch["why_total_ko"] = ""

    # 7. UID 집계
    df_patch["uid_num"] = pd.to_numeric(df_patch["uid"], errors="coerce")
    uid_stat = (df_patch.groupby("uid", as_index=False)
                .agg(uid_num=("uid_num","first"),
                     n_patch=("patch_global_id","count"),
                     time_min_pred_uid=(time_pred_col,"sum"),
                     time_min_calib_uid=(time_calib_col,"sum"),
                     cost_krw_pred_uid=(cost_pred_col,"sum"),
                     cost_krw_calib_uid=(cost_calib_col,"sum"),
                     time_scale_uid=(time_scale_col,"first"),
                     cost_scale_uid=(cost_scale_col,"first")))

    # 8. Top K WHY
    def topk_why(sub_df, col, k=5):
        s2 = sub_df.sort_values(col, ascending=False).head(k)
        lines = []
        for _, r in s2.iterrows():
            pid = int(r["patch_global_id"])
            t = float(r[time_calib_col]); c = float(r[cost_calib_col])
            why = str(r.get("why_total_ko","")).split("\n")[0].strip()
            lines.append(f"- Patch {pid}: {why} (시간≈{t:.1f}분, 비용≈{c:,.0f}원)")
        return "\n".join(lines)

    top_time = df_patch.groupby("uid").apply(lambda g: topk_why(g, time_calib_col)).reset_index(name="top_patches_time_ko")
    top_cost = df_patch.groupby("uid").apply(lambda g: topk_why(g, cost_calib_col)).reset_index(name="top_patches_cost_ko")
    uid_stat = uid_stat.merge(top_time, on="uid", how="left").merge(top_cost, on="uid", how="left")

    # 9. GT 병합 & 저장
    uid_report = uid_stat
    if have_gt:
        keep = [c for c in df_gt.columns if c not in uid_report.columns or c=="uid"]
        uid_report = uid_report.merge(df_gt[keep], on="uid", how="left")

    OUT1 = os.path.join(OUT_DIR, "S_UID_REPORT_UID150.csv")
    OUT2 = os.path.join(OUT_DIR, "S_UID_REPORT.csv")
    uid_report.to_csv(OUT1, index=False, encoding="utf-8-sig")
    uid_report.to_csv(OUT2, index=False, encoding="utf-8-sig")

    _print("[13R] SAVE: " + OUT2)
    _print("===== 13R DONE =====")

if __name__ == "__main__":
    main()