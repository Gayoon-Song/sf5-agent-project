# ============================================================
# 10R_PATCH_LGBM_TRAIN_LOCAL_v3.py  (FULL REPLACEMENT • ASCII-safe)
#
# 기능
#   - 17R_DATA/S_PATCH_LABEL_UID150.csv 기반으로 LGBM 회귀 모델(시간/비용) 학습
#   - 학습/스키마 저장: 10R_MODELS/*
#   - 예측: --predict_csv 로 임의 CSV(예: 07S 템플릿/라벨 결과)를 받아 예측 컬럼 추가 저장
#
# 출력
#   - 10R_MODELS/patch_time_lgbm.pkl
#   - 10R_MODELS/patch_cost_lgbm.pkl
#   - 10R_MODELS/S_PATCH_WITH_TIME_COST_PRED_UID150.csv
#
# 사용 예
#   - 학습: python 10R_PATCH_LGBM_TRAIN_LOCAL_v3.py [--force]
#   - 예측: python 10R_PATCH_LGBM_TRAIN_LOCAL_v3.py --predict_csv "PATH_TO_CSV"
# ============================================================

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

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

# ---------- Lib checks ----------
try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(_s("[ERROR] lightgbm not installed. pip install lightgbm")) from e

try:
    import joblib
except ImportError as e:
    raise ImportError(_s("[ERROR] joblib not installed. pip install joblib")) from e

from sklearn.metrics import mean_absolute_error

# ---------- Helpers ----------
def extract_uid_series(df: pd.DataFrame) -> pd.Series:
    for col in ["uid", "UID", "UID_norm", "uid_num"]:
        if col in df.columns:
            s = df[col].astype(str)
            s_num = s.str.extract(r"(\d+)")[0].fillna(s)
            s_num = s_num.astype(str).str.replace(r"\D", "", regex=True).replace("", np.nan)
            s_num = s_num.fillna("0")
            return s_num.astype(int).astype(str).str.zfill(3)
    # If not found, return dummy
    return pd.Series(["000"] * len(df))

def pick_target(df, calib_col, auto_col, name="target"):
    if calib_col in df.columns:
        s = pd.to_numeric(df[calib_col], errors="coerce").fillna(0.0).abs()
        nonzero_ratio = (s > 0).mean()
        nonzero_sum = float(s.sum())
        if nonzero_sum > 1e-6 and nonzero_ratio >= 0.01:
            _print(f"[10R] {name}: use {calib_col} (valid)")
            return calib_col
    return auto_col

def build_feature_matrix(df, feature_cols, cat_cols):
    X = df.reindex(columns=feature_cols, fill_value=np.nan).copy()
    for c in feature_cols:
        if c in cat_cols:
            X[c] = X[c].astype("string").fillna("NA").astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X

def uid_split(df_uid3: pd.Series, valid_ratio=0.2, seed=42):
    uids = sorted(df_uid3.unique().tolist())
    rng = np.random.default_rng(seed)
    n_valid = max(1, int(len(uids) * valid_ratio))
    valid = set(rng.choice(uids, size=n_valid, replace=False).tolist())
    return valid

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--in_csv", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--predict_csv", default=None)
    # pipeline compat
    ap.add_argument("--uid", default=None) 
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--predict_only", action="store_true") # 호환용
    args, _unknown = ap.parse_known_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    BASE = ROOT_DIR
    IN_CSV = args.in_csv if args.in_csv else os.path.join(BASE, "17R_DATA", "S_PATCH_LABEL_UID150.csv")
    OUT = args.out_dir if args.out_dir else os.path.join(BASE, "10R_MODELS")
    os.makedirs(OUT, exist_ok=True)
    PRED_DIR = os.path.join(OUT, "predictions"); os.makedirs(PRED_DIR, exist_ok=True)

    time_pkl = os.path.join(OUT, "patch_time_lgbm.pkl")
    cost_pkl = os.path.join(OUT, "patch_cost_lgbm.pkl")
    meta_path = os.path.join(OUT, "model_meta.json")

    _print("====================================")
    _print("===== 10R PATCH LGBM TRAIN/PRED ====")
    _print("BASE_DIR : " + BASE)
    _print("OUT_DIR  : " + OUT)

    # -------- Prediction Mode (if requested or trained models exist + not force) --------
    have_models = os.path.isfile(time_pkl) and os.path.isfile(cost_pkl)
    
    # CASE 1: Prediction request from pipeline
    # s_pipeline calls: --predict_csv AUTO --uid ...
    predict_target = args.predict_csv
    
    if predict_target == "AUTO" and args.uid:
        # Resolve path for specific UID from 07S output
        u3 = extract_uid_series(pd.DataFrame({"uid": [args.uid]})).iloc[0]
        predict_target = os.path.join(BASE, "07S_PATCH_LABEL", f"UID_{u3}_PATCH_LABEL_TEMPLATE.csv")
        
    if predict_target:
        if not have_models:
            _print("[10R][ERROR] No models found for prediction. Run training first.")
            return

        paths = [p.strip() for p in re.split(r"[;,]", predict_target) if p.strip()]
        
        # Load models
        time_bundle = joblib.load(time_pkl)
        cost_bundle = joblib.load(cost_pkl)
        t_features, t_cats = time_bundle["features"], time_bundle["cat_cols"]
        c_features, c_cats = cost_bundle["features"], cost_bundle["cat_cols"]

        for pth in paths:
            if not os.path.isfile(pth):
                _print("[10R][WARN] predict file not found: " + pth); continue
            
            try:
                dfx = pd.read_csv(pth, encoding="utf-8-sig")
            except:
                dfx = pd.read_csv(pth, encoding="cp949")
            
            _print("[10R] predict on: " + pth)

            # Feature Matrix & Predict
            Xt = build_feature_matrix(dfx, t_features, t_cats)
            Xc = build_feature_matrix(dfx, c_features, c_cats)
            dfx["time_min_pred"] = time_bundle["model"].predict(Xt)
            dfx["cost_krw_pred"] = cost_bundle["model"].predict(Xc)

            # Save (Overwrite input or save as _PRED)
            # Pipeline expects results in place or specific location. 
            # 10R output convention: S_PATCH_WITH_TIME_COST_PRED_UIDxxx.csv
            
            # If input was 07S output, we might want to save as 10R output
            if "07S_PATCH_LABEL" in pth and "UID_" in pth:
                u_str = extract_uid_series(dfx).iloc[0]
                out_name = f"S_PATCH_WITH_TIME_COST_PRED_UID{u_str}.csv"
                out_path = os.path.join(OUT, out_name)
                dfx.to_csv(out_path, index=False, encoding="utf-8-sig")
                _print("[10R] saved prediction: " + out_path)
            else:
                # In-place or sidecar
                base, _ = os.path.splitext(pth)
                out_path = base + "_PRED.csv"
                dfx.to_csv(out_path, index=False, encoding="utf-8-sig")
                _print("[10R] saved prediction: " + out_path)

        _print("===== 10R PREDICT DONE =====")
        return

    # -------- Training Mode (if no predict_csv and (force or no models)) --------
    if not os.path.isfile(IN_CSV):
        _print("[10R][WARN] train CSV not found, cannot train: " + IN_CSV)
        return

    df = pd.read_csv(IN_CSV, encoding="utf-8-sig")
    df["uid_str"] = extract_uid_series(df)

    TIME_TGT = pick_target(df, "time_min_calib", "time_min_auto", "TIME_TGT")
    COST_TGT = pick_target(df, "cost_krw_calib", "cost_krw_auto", "COST_TGT")

    # Drop non-generalizable columns
    drop_cols = set([
        "uid","UID","uid_str","uid_num","UID_norm","patch_global_id",
        "pair_id","before_face_index","after_face_index"
    ])
    drop_cols.update([c for c in df.columns if c.endswith("_user") or c.endswith("_gt") or c.endswith("_pred")])
    drop_cols.update([c for c in df.columns if "measured" in c.lower() or c.startswith("why_") or c.startswith("text_")])
    
    # Feature Selection
    feature_cols = [c for c in df.columns if c not in drop_cols and c not in [TIME_TGT, COST_TGT]]
    cat_cols = [c for c in feature_cols if (df[c].dtype == "object" or str(df[c].dtype) == "category")]

    # Split
    valid_uids = uid_split(df["uid_str"], valid_ratio=0.2, seed=42)
    is_valid = df["uid_str"].isin(valid_uids)
    is_train = ~is_valid

    def build_xy(dfx):
        X = build_feature_matrix(dfx[feature_cols], feature_cols, cat_cols)
        y_time = pd.to_numeric(dfx[TIME_TGT], errors="coerce").fillna(0.0)
        y_cost = pd.to_numeric(dfx[COST_TGT], errors="coerce").fillna(0.0)
        return X, y_time, y_cost

    X_tr, y_time_tr, y_cost_tr = build_xy(df.loc[is_train])
    X_va, y_time_va, y_cost_va = build_xy(df.loc[is_valid])

    _print("[10R] training models ...")
    
    # Train Time Model
    dtr_t = lgb.Dataset(X_tr, label=y_time_tr, categorical_feature=cat_cols, free_raw_data=False)
    dva_t = lgb.Dataset(X_va, label=y_time_va, categorical_feature=cat_cols, free_raw_data=False)
    params = {
        "objective": "regression_l1", "metric": "mae", "learning_rate": 0.05,
        "num_leaves": 127, "min_data_in_leaf": 50, "feature_fraction": 0.9,
        "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42, "verbosity": -1,
    }
    model_time = lgb.train(
        params, dtr_t, num_boost_round=5000,
        valid_sets=[dtr_t, dva_t], valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)] # quiet
    )
    
    # Train Cost Model
    dtr_c = lgb.Dataset(X_tr, label=y_cost_tr, categorical_feature=cat_cols, free_raw_data=False)
    dva_c = lgb.Dataset(X_va, label=y_cost_va, categorical_feature=cat_cols, free_raw_data=False)
    model_cost = lgb.train(
        params, dtr_c, num_boost_round=5000,
        valid_sets=[dtr_c, dva_c], valid_names=["train","valid"],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)] # quiet
    )

    # Save
    joblib.dump({"model": model_time, "features": feature_cols, "cat_cols": cat_cols, "target": TIME_TGT}, time_pkl)
    joblib.dump({"model": model_cost, "features": feature_cols, "cat_cols": cat_cols, "target": COST_TGT}, cost_pkl)
    
    _print("===== 10R TRAIN DONE =====")

if __name__ == "__main__":
    main()