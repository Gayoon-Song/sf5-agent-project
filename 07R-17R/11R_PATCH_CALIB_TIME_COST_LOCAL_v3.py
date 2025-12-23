# ============================================================
# 11R_PATCH_CALIB_TIME_COST_LOCAL_v3.py  (FULL REPLACEMENT • ASCII-safe)
#
# 역할
#   - 10R 예측 CSV의 patch-level 시간/비용을 UID 단위로 합산
#   - (선택) 17R UID 실측(GT)과 비교하여 UID별 보정 스케일 산출
#   - 스케일을 patch-level 예측에 곱해 time_min_calib / cost_krw_calib 산출
#
# 사용 예
#   - 기본: python 11R_PATCH_CALIB_TIME_COST_LOCAL_v3.py
#   - 지정: python 11R_PATCH_CALIB_TIME_COST_LOCAL_v3.py --pred_csv X.csv
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
def _norm_uid3_series(s: pd.Series) -> pd.Series:
    ss = s.astype(str)
    ss = ss.str.extract(r"(\d+)")[0].fillna("0")
    return ss.astype(int).astype(str).str.zfill(3)

def _pick_col(cols, candidates):
    for c in candidates:
        if c in cols: return c
    return None

def _read_csv_flex(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8")

def _split_paths(raw: str):
    tmp = re.split(r"[;,]", raw)
    return [p.strip() for p in tmp if p.strip()]

# ---------- core ----------
def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--pred_csv", default=None, help="보정 대상 예측 CSV")
    ap.add_argument("--gt_uid_csv", default=None, help="UID 레벨 실측(GT) CSV (선택)")
    ap.add_argument("--out_dir", default=None)
    # pipeline compat
    ap.add_argument("--uid", default=None)
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    args, _unknown = ap.parse_known_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    DEFAULT_PRED = os.path.join(ROOT_DIR, "10R_MODELS", "S_PATCH_WITH_TIME_COST_PRED_UID150.csv")
    DEFAULT_GT   = os.path.join(ROOT_DIR, "17R_DATA",   "S_UID_GT_001_150.csv")
    OUT_DIR = args.out_dir if args.out_dir else os.path.join(ROOT_DIR, "11R_GROUND_TRUTH")
    os.makedirs(OUT_DIR, exist_ok=True)

    _print("=======================================")
    _print("===== 11R PATCH CALIB (LOCAL v3) =====")
    _print("ROOT     : " + ROOT_DIR)
    _print("OUT_DIR  : " + OUT_DIR)

    # -------- 1) Load prediction CSV(s)
    pred_paths = _split_paths(args.pred_csv) if args.pred_csv else [DEFAULT_PRED]
    
    # 만약 pred_csv가 "AUTO"라면 (s_pipeline에서 보냄)
    if args.pred_csv == "AUTO" and args.uid:
        u_num = re.findall(r"\d+", args.uid)[-1]
        u3 = f"{int(u_num):03d}"
        cand = os.path.join(ROOT_DIR, "10R_MODELS", f"S_PATCH_WITH_TIME_COST_PRED_UID{u3}.csv")
        if os.path.isfile(cand):
            pred_paths = [cand]
        else:
            _print(f"[11R][WARN] AUTO pred csv not found: {cand}. Using default.")
            pred_paths = [DEFAULT_PRED]

    dfs = []
    for p in pred_paths:
        if not os.path.isfile(p):
            _print(f"[11R][WARN] pred CSV not found: {p}. Skipping.")
            continue
        dfp = _read_csv_flex(p)
        dfp["__pred_src__"] = os.path.basename(p)
        dfs.append(dfp)
        _print("[11R] load pred: " + p + " shape=" + str(dfp.shape))
    
    if not dfs:
        _print("[11R][ERROR] No valid input CSVs found.")
        return

    df_pred = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    
    # normalize uid
    uid_col = _pick_col(df_pred.columns, ["uid","UID","UID_norm","uid_num"])
    if uid_col is None:
        df_pred["uid3"] = "000"
    else:
        df_pred["uid3"] = _norm_uid3_series(df_pred[uid_col])

    # pick prediction columns
    time_pred_col = _pick_col(df_pred.columns, ["time_min_pred","time_pred_min","time_min_pred_patch"])
    cost_pred_col = _pick_col(df_pred.columns, ["cost_krw_pred","cost_pred_krw","cost_krw_pred_patch"])
    if time_pred_col is None or cost_pred_col is None:
        raise KeyError(_s("[11R][ERROR] time/cost pred columns not found in pred CSV"))

    # -------- 2) UID-level pred sum
    uid_pred_sum = (
        df_pred.groupby("uid3", dropna=True)
               .agg(time_min_pred_uid=(time_pred_col,"sum"),
                    cost_krw_pred_uid=(cost_pred_col,"sum"),
                    n_patch=("uid3","size"))
               .reset_index().rename(columns={"uid3":"uid"})
    )

    # -------- 3) Load GT (optional)
    gt_path = args.gt_uid_csv if args.gt_uid_csv else DEFAULT_GT
    have_gt = os.path.isfile(gt_path)
    df_uid_report = uid_pred_sum.copy()
    
    # [수정] 미리 컬럼을 생성하지 않고 Merge를 먼저 수행 (중복 이름 충돌 방지)

    if have_gt:
        dfg = _read_csv_flex(gt_path)
        g_uid_col = _pick_col(dfg.columns, ["uid","UID","UID_norm","uid_num"])
        if g_uid_col:
            dfg["uid"] = _norm_uid3_series(dfg[g_uid_col])
            time_gt_col = _pick_col(dfg.columns, ["time_min_measured","time_min_user","time_min_gt"])
            cost_gt_col = _pick_col(dfg.columns, ["cost_krw_measured","cost_krw_user","cost_krw_gt"])
            
            if time_gt_col and cost_gt_col:
                dfg = dfg[["uid", time_gt_col, cost_gt_col]].rename(
                    columns={time_gt_col:"time_min_measured", cost_gt_col:"cost_krw_measured"}
                )
                df_uid_report = df_uid_report.merge(dfg, on="uid", how="left")

    # [수정] Merge 후에도 컬럼이 없다면 그때 NaN으로 생성
    if "time_min_measured" not in df_uid_report.columns:
        df_uid_report["time_min_measured"] = np.nan
    if "cost_krw_measured" not in df_uid_report.columns:
        df_uid_report["cost_krw_measured"] = np.nan

    # -------- 4) compute scale (fallback=1.0)
    EPS = 1e-9
    def _scale(measured, pred_sum):
        try:
            m = float(measured) if measured is not None else np.nan
            p = float(pred_sum) if pred_sum is not None else np.nan
        except: return 1.0
        if not np.isfinite(p) or p < EPS: return 1.0
        if not np.isfinite(m) or m < EPS: return 1.0
        return float(m) / float(p)

    df_uid_report["time_scale_uid"] = df_uid_report.apply(
        lambda r: _scale(r["time_min_measured"], r["time_min_pred_uid"]), axis=1
    )
    df_uid_report["cost_scale_uid"] = df_uid_report.apply(
        lambda r: _scale(r["cost_krw_measured"], r["cost_krw_pred_uid"]), axis=1
    )

    # -------- 5) apply scale to patches
    df_out = df_pred.copy()
    df_out["uid"] = df_out["uid3"]
    df_out = df_out.merge(
        df_uid_report[["uid","time_scale_uid","cost_scale_uid"]],
        on="uid", how="left"
    )
    df_out["time_scale_uid"] = df_out["time_scale_uid"].fillna(1.0)
    df_out["cost_scale_uid"] = df_out["cost_scale_uid"].fillna(1.0)
    df_out["time_min_calib"] = pd.to_numeric(df_out[time_pred_col], errors="coerce").fillna(0.0) * df_out["time_scale_uid"]
    df_out["cost_krw_calib"] = pd.to_numeric(df_out[cost_pred_col], errors="coerce").fillna(0.0) * df_out["cost_scale_uid"]

    # -------- 6) sanity check & save
    out_uid_report  = os.path.join(OUT_DIR, "S_UID_TIME_COST_CALIB_REPORT.csv")
    out_patch_calib = os.path.join(OUT_DIR, "S_PATCH_TIME_COST_CALIB_UID150.csv")
    out_patch_alias = os.path.join(OUT_DIR, "S_PATCH_TIME_COST_CALIB_UID.csv")

    df_uid_report.to_csv(out_uid_report, index=False, encoding="utf-8")
    
    if args.uid and len(pred_paths) == 1:
        u_num = re.findall(r"\d+", args.uid)[-1]
        u3 = f"{int(u_num):03d}"
        single_out = os.path.join(OUT_DIR, f"S_PATCH_TIME_COST_CALIB_UID{u3}.csv")
        df_out.to_csv(single_out, index=False, encoding="utf-8")
        _print(f"[11R] saved single: {single_out}")
    else:
        df_out.to_csv(out_patch_calib, index=False, encoding="utf-8")
        df_out.to_csv(out_patch_alias, index=False, encoding="utf-8")

    _print("===== 11R DONE =====")

if __name__ == "__main__":
    main()