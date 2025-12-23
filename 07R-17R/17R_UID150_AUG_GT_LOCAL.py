# ============================================================
# 17R_UID150_AUG_GT_LOCAL.py  (FULL REPLACEMENT • ASCII-safe)
#
# 역할
#   - 실측(001~014) 패치 분포 기반 증강(015~max_uid)
#   - max_uid 자유화, 출력 alias + 범위표시 파일 동시 생성
#
# 실행 예:
#   python 17R_UID150_AUG_GT_LOCAL.py --max_uid 150
# ============================================================

import os, re, json, math, argparse, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ---------- ASCII-safe print ----------
def _s(x: str) -> str:
    return str(x).encode("ascii", "ignore").decode("ascii")
def aprint(msg: str):
    print(_s(msg))

# ---------- Root resolver ----------
def resolve_root(root_cli=None):
    if root_cli: return os.path.abspath(root_cli)
    env_root = os.environ.get("SF5_ROOT", "").strip()
    if env_root: return os.path.abspath(env_root)
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

ROOT_DIR = resolve_root()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

# ---------- Helpers ----------
def ensure_uid_str(x):
    if pd.isna(x): return None
    s = str(x).strip()
    s = s.replace("UID_", "").replace("uid_", "").replace("UID", "").replace("uid", "")
    m = re.search(r"\d+", s)
    if not m: return None
    return f"{int(float(m.group())):03d}"

def pick_uid_col(df):
    for c in ["uid", "UID", "uid_str", "UID_str", "uid_num", "UID_num"]:
        if c in df.columns: return c
    return None

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def fmt_stats(a):
    a = np.asarray(a, dtype=float)
    return f"min={np.nanmin(a):.1f}, mean={np.nanmean(a):.1f}, max={np.nanmax(a):.1f}"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_uid", type=int, default=150)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    global ROOT_DIR
    if args.base_dir:
        ROOT_DIR = os.path.abspath(args.base_dir)

    if args.max_uid < 15:
        raise ValueError("[17R][ERROR] max_uid must be >= 15")

    BASE = ROOT_DIR
    DIR_07S   = os.path.join(BASE, "07S_PATCH_LABEL")
    INPUT_07S = os.path.join(DIR_07S, "S_PATCH_LABEL_TEMPLATE.csv")
    EXCEL_GT  = os.path.join(BASE, "11R_GROUND_TRUTH", "공정_실측값.xlsx")

    OUT_DIR   = ensure_dir(os.path.join(BASE, "17R_DATA"))
    OUT_PATCH_ALIAS = os.path.join(OUT_DIR, "S_PATCH_LABEL_UID.csv")
    OUT_UIDGT_ALIAS = os.path.join(OUT_DIR, "S_UID_GT.csv")
    OUT_PATCH_RANGE = os.path.join(OUT_DIR, f"S_PATCH_LABEL_UID001_{args.max_uid:03d}.csv")
    OUT_UIDGT_RANGE = os.path.join(OUT_DIR, f"S_UID_GT_001_{args.max_uid:03d}.csv")

    if not args.overwrite:
        for fp in [OUT_PATCH_ALIAS, OUT_UIDGT_ALIAS, OUT_PATCH_RANGE, OUT_UIDGT_RANGE]:
            if os.path.exists(fp):
                raise FileExistsError(f"[17R][ERROR] output exists: {fp} (use --overwrite)")

    aprint("====================================")
    aprint("===== 17R UID Aug (LOCAL FULL) =====")
    aprint(f"[17R] BASE_DIR  : {BASE}")
    aprint(f"[17R] MAX_UID   : {args.max_uid}")

    rng = np.random.default_rng(args.seed)

    # 1) Load 07S
    if not os.path.exists(INPUT_07S):
        raise FileNotFoundError(f"[17R][ERROR] 07S template not found: {INPUT_07S}")
    df_patch = pd.read_csv(INPUT_07S)
    uid_col = pick_uid_col(df_patch)
    if not uid_col: raise KeyError("[17R] UID column not found in 07S")
    
    df_patch["uid"] = df_patch[uid_col].apply(ensure_uid_str)
    df_patch = df_patch[df_patch["uid"].notna()].copy()

    real_uids = [f"{i:03d}" for i in range(1, 15)]
    df_real = df_patch[df_patch["uid"].isin(real_uids)].copy()
    if df_real.empty:
        raise ValueError("[17R] No real UIDs (001-014) in 07S")

    # 2) Load GT
    if not os.path.exists(EXCEL_GT):
        raise FileNotFoundError(f"[17R][ERROR] GT Excel not found: {EXCEL_GT}")
    
    xls = pd.ExcelFile(EXCEL_GT)
    # simple heuristic for sheet
    sheet = xls.sheet_names[0]
    for s in xls.sheet_names:
        if "실측" in s or "GT" in s: sheet = s; break
        
    df_gt_raw = xls.parse(sheet)
    
    # col detect
    uid_gt_col = time_gt_col = cost_gt_col = None
    for c in df_gt_raw.columns:
        lc = str(c).lower()
        if not uid_gt_col and "uid" in lc: uid_gt_col = c
        if not time_gt_col and ("time" in lc or "hour" in lc): time_gt_col = c
        if not cost_gt_col and ("cost" in lc or "krw" in lc): cost_gt_col = c
        
    if not (uid_gt_col and time_gt_col and cost_gt_col):
        # fallback hardcoded if detection fails
        uid_gt_col = df_gt_raw.columns[0]
        time_gt_col = df_gt_raw.columns[1]
        cost_gt_col = df_gt_raw.columns[2]
        aprint("[17R][WARN] Col detection failed, using index 0,1,2")

    df_gt = df_gt_raw[[uid_gt_col, time_gt_col, cost_gt_col]].copy()
    df_gt.columns = ["uid_raw", "time_hour_measured", "cost_krw_measured"]
    df_gt["uid"] = df_gt["uid_raw"].apply(ensure_uid_str)
    df_gt["time_min_measured"] = safe_num(df_gt["time_hour_measured"]).fillna(0.0) * 60.0
    df_gt["cost_krw_measured"] = safe_num(df_gt["cost_krw_measured"]).fillna(0.0)

    # ensure real uids exist in GT
    existing = set(df_gt["uid"].dropna())
    for u in real_uids:
        if u not in existing:
            df_gt = pd.concat([df_gt, pd.DataFrame([{
                "uid": u, "time_min_measured": 0.0, "cost_krw_measured": 0.0
            }])], ignore_index=True)
            
    df_gt["uid_kind"] = "real"
    df_gt["uid_num"] = df_gt["uid"].apply(lambda x: int(x) if x else 0)
    df_gt = df_gt.sort_values("uid_num").reset_index(drop=True)

    # 3) Column sets
    time_src_col = "time_min_user" if "time_min_user" in df_real.columns else "time_min_auto"
    cost_src_col = "cost_krw_user" if "cost_krw_user" in df_real.columns else "cost_krw_auto"
    
    shape_cols = [c for c in [
        "delta_max","delta_avg","delta_min","abs_delta_max","abs_delta_avg","area_est","diag_mm",
        "bbox_z_span","bbox_xy_area_ratio"
    ] if c in df_real.columns]
    pos_cols = [c for c in ["center_x","center_y","center_z"] if c in df_real.columns]
    
    shape_std = {c: float(pd.to_numeric(df_real[c], errors="coerce").std()) for c in shape_cols + pos_cols}

    # 4) Real Calibration
    calib_frames = []
    for u in real_uids:
        sub = df_real[df_real["uid"]==u].copy()
        if sub.empty: continue
        
        gt_row = df_gt[df_gt["uid"]==u]
        t_total = float(gt_row["time_min_measured"].iloc[0]) if not gt_row.empty else 0.0
        c_total = float(gt_row["cost_krw_measured"].iloc[0]) if not gt_row.empty else 0.0
        
        w_time = safe_num(sub[time_src_col]).fillna(0.0).clip(lower=0.0).values
        w_cost = safe_num(sub[cost_src_col]).fillna(0.0).clip(lower=0.0).values
        
        if w_time.sum() <= 0: w_time = np.ones(len(sub))
        if w_cost.sum() <= 0: w_cost = np.ones(len(sub))
        
        sub["time_min_calib"] = w_time / w_time.sum() * t_total
        sub["cost_krw_calib"] = w_cost / w_cost.sum() * c_total
        calib_frames.append(sub)
        
    df_real_calib = pd.concat(calib_frames, ignore_index=True)

    # 5) Augmentation
    real_counts = df_real_calib.groupby("uid").size()
    mean_cnt = float(real_counts.mean())
    min_p = max(1, int(0.6 * mean_cnt))
    max_p = max(min_p+1, int(1.4 * mean_cnt))
    
    real_uids_list = real_counts.index.tolist()
    
    aug_frames, aug_uid_rows = [], []
    
    def add_noise(df_in, cols):
        for c in cols:
            v = safe_num(df_in[c]).fillna(0.0).values
            sigma = 0.03 * np.abs(v) + 0.1 * shape_std.get(c, 0.0)
            noise = rng.normal(0.0, np.maximum(sigma, 1e-6))
            df_in[c] = v + noise
        return df_in

    for new_num in range(15, args.max_uid + 1):
        new_uid = f"{new_num:03d}"
        
        # Pick source UID
        src_uid = rng.choice(real_uids_list)
        src_sub = df_real_calib[df_real_calib["uid"]==src_uid].copy()
        
        # Sample patches
        n_sample = int(rng.integers(min_p, max_p + 1))
        # if source has fewer patches, sample with replacement
        replace = (len(src_sub) < n_sample)
        sampled = src_sub.sample(n=n_sample, replace=replace, random_state=new_num).copy()
        
        sampled["uid"] = new_uid
        sampled["uid_num"] = new_num
        
        # Noise
        sampled = add_noise(sampled, shape_cols + pos_cols)
        
        # Total metrics
        gt_row = df_gt[df_gt["uid"]==src_uid]
        t_base = float(gt_row["time_min_measured"].iloc[0]) if not gt_row.empty else 0.0
        c_base = float(gt_row["cost_krw_measured"].iloc[0]) if not gt_row.empty else 0.0
        
        scale = rng.lognormal(0.0, 0.25)
        t_aug = t_base * scale
        c_aug = c_base * scale
        
        # Distribute
        w_time = safe_num(sampled["time_min_calib"]).fillna(0.0).clip(lower=0.0).values
        w_cost = safe_num(sampled["cost_krw_calib"]).fillna(0.0).clip(lower=0.0).values
        if w_time.sum() <= 0: w_time = np.ones(len(sampled))
        if w_cost.sum() <= 0: w_cost = np.ones(len(sampled))
        
        sampled["time_min_calib"] = w_time / w_time.sum() * t_aug
        sampled["cost_krw_calib"] = w_cost / w_cost.sum() * c_aug
        
        aug_frames.append(sampled)
        aug_uid_rows.append({
            "uid": new_uid, "uid_num": new_num, "uid_kind": "aug",
            "time_min_measured": t_aug, "cost_krw_measured": c_aug
        })

    df_aug = pd.concat(aug_frames, ignore_index=True) if aug_frames else pd.DataFrame()
    df_aug_gt = pd.DataFrame(aug_uid_rows)
    
    # 6) Merge & Save
    df_all_patch = pd.concat([df_real_calib, df_aug], ignore_index=True)
    df_all_gt = pd.concat([df_gt, df_aug_gt], ignore_index=True).sort_values("uid_num")
    
    df_all_patch.to_csv(OUT_PATCH_ALIAS, index=False, encoding="utf-8-sig")
    df_all_patch.to_csv(OUT_PATCH_RANGE, index=False, encoding="utf-8-sig")
    
    df_all_gt.to_csv(OUT_UIDGT_ALIAS, index=False, encoding="utf-8-sig")
    df_all_gt.to_csv(OUT_UIDGT_RANGE, index=False, encoding="utf-8-sig")
    
    aprint(f"[17R] SAVE: {OUT_PATCH_ALIAS}")
    aprint(f"[17R] SAVE: {OUT_UIDGT_ALIAS}")
    aprint("===== 17R DONE =====")

if __name__ == "__main__":
    main()