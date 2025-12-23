# ============================================
# 14R_AGENT_KB_EXPORT_LOCAL_UID150_v3.py  (FULL REPLACEMENT • ASCII-safe)
#
# - 유재성님 14R Colab FINAL FULL UID150 그대로 로컬용 변환
# - 13R UID summary + 12R/11R patch WHY/보정 결과
#   → RAG/Agent용 KB(CSV/JSONL) 생성
#
# 실행:
#   python 14R_AGENT_KB_EXPORT_LOCAL_UID150_v3.py
# ============================================

import os, json, re, argparse
import pandas as pd
import numpy as np
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

# ------------------------------
# 1) Path / Utils (원본 동일)
# ------------------------------
def find_file_under(root, filename):
    for dp, dn, fn in os.walk(root):
        if filename in fn:
            return os.path.join(dp, filename)
    return None

def load_first_existing(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

def pick_uid_col(df):
    for c in ["uid","UID","uid_str","uid_padded","uid_num"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "uid" in c.lower():
            return c
    raise KeyError("[14R] uid 컬럼을 찾을 수 없습니다.")

def norm_uid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    s = re.sub(r"[^0-9]", "", s)  # 숫자만 남김
    if s == "":
        return None
    return s.zfill(3)

def fmt_time_min(v):
    try: v = float(v)
    except: return "N/A"
    if np.isnan(v): return "N/A"
    h = int(v // 60)
    m = int(round(v - h*60))
    if h > 0 and m > 0: return f"{h}시간 {m}분"
    if h > 0: return f"{h}시간"
    return f"{m}분"

def fmt_cost(v):
    try: v = float(v)
    except: return "N/A"
    if np.isnan(v): return "N/A"
    return f"{int(round(v)):,}원"

def first_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--dir13", default=None)
    ap.add_argument("--dir12", default=None)
    ap.add_argument("--dir11", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    BASE = ROOT_DIR
    DIR_13R  = os.path.abspath(args.dir13) if args.dir13 else os.path.join(BASE, "13R_UID_REPORT")
    DIR_12R  = os.path.abspath(args.dir12) if args.dir12 else os.path.join(BASE, "12R_RULEBOOK")
    DIR_11R  = os.path.abspath(args.dir11) if args.dir11 else os.path.join(BASE, "11R_GROUND_TRUTH")
    OUT_DIR  = os.path.abspath(args.out) if args.out else os.path.join(BASE, "14R_AGENT_KB")
    os.makedirs(OUT_DIR, exist_ok=True)

    _print("====================================")
    _print("===== 14R RAG KB EXPORT (LOCAL v3) =====")
    _print("BASE_DIR : " + BASE)
    _print("OUT_DIR  : " + OUT_DIR)

    # ------------------------------
    # 2) UID Summary 입력 로딩
    # ------------------------------
    uid_path = load_first_existing([
        os.path.join(DIR_13R, "S_UID_LEVEL_SUMMARY.csv"),
        os.path.join(DIR_13R, "S_UID_REPORT_UID150.csv"),
        os.path.join(DIR_13R, "S_UID_REPORT.csv"),
    ])
    if uid_path is None:
        uid_path = (find_file_under(DIR_13R, "S_UID_LEVEL_SUMMARY.csv")
                    or find_file_under(DIR_13R, "S_UID_REPORT_UID150.csv")
                    or find_file_under(DIR_13R, "S_UID_REPORT.csv"))

    if uid_path is None:
        raise FileNotFoundError("[14R][ERROR] 13R UID summary 입력을 찾지 못했습니다.")

    df_uid = read_csv_robust(uid_path)
    uid_col_uid = pick_uid_col(df_uid)
    df_uid["uid"] = df_uid[uid_col_uid].apply(norm_uid)

    _print(f"[14R] UID summary 로딩: {uid_path}")
    _print(f"[14R] UID summary shape: {df_uid.shape}")

    n_patch_col   = first_col(df_uid, ["n_patch","patch_count","n_patches"])
    time_uid_col  = first_col(df_uid, ["time_min_calib_uid","time_min_pred_uid","time_min_measured"])
    cost_uid_col  = first_col(df_uid, ["cost_krw_calib_uid","cost_krw_pred_uid","cost_krw_measured"])

    # ------------------------------
    # 3) Patch Detail 입력 로딩
    # ------------------------------
    patch_agent_path = load_first_existing([
        os.path.join(DIR_13R, "S_UID_PATCH_DETAIL_FOR_AGENT.csv"),
    ])
    if patch_agent_path is None:
        patch_agent_path = find_file_under(DIR_13R, "S_UID_PATCH_DETAIL_FOR_AGENT.csv")

    if patch_agent_path is None:
        # fallback to 12R/11R results
        patch_agent_path = load_first_existing([
            os.path.join(DIR_12R, "S_PATCH_RULEBOOK_EXPLAIN_UID150.csv"),
            os.path.join(DIR_12R, "S_PATCH_RULEBOOK_EXPLAIN.csv"),
            os.path.join(DIR_11R, "S_PATCH_TIME_COST_CALIB_UID150.csv"),
            os.path.join(DIR_11R, "S_PATCH_TIME_COST_CALIB_UID.csv"),
        ])
        if patch_agent_path is None:
            patch_agent_path = (find_file_under(DIR_12R, "S_PATCH_RULEBOOK_EXPLAIN_UID150.csv")
                                or find_file_under(DIR_11R, "S_PATCH_TIME_COST_CALIB_UID150.csv"))

    if patch_agent_path is None:
        raise FileNotFoundError("[14R][ERROR] Patch detail 입력을 찾지 못했습니다.")

    df_patch = read_csv_robust(patch_agent_path)
    uid_col_patch = pick_uid_col(df_patch)
    df_patch["uid"] = df_patch[uid_col_patch].apply(norm_uid)

    if "why_total_ko" not in df_patch.columns:
        why_alt = None
        for c in df_patch.columns:
            if ("why" in c.lower()) and ("ko" in c.lower()):
                why_alt = c
                break
        if why_alt is not None:
            df_patch["why_total_ko"] = df_patch[why_alt]
        else:
            df_patch["why_total_ko"] = ""

    _print(f"[14R] Patch detail 로딩: {patch_agent_path}")
    _print(f"[14R] Patch detail shape: {df_patch.shape}")

    # ------------------------------
    # 4) KB 생성
    # ------------------------------
    kb_rows = []

    # 4-1) UID 레벨 KB
    for _, r in df_uid.iterrows():
        uid = r["uid"]
        if uid is None: continue

        n_patch  = r[n_patch_col] if n_patch_col else np.nan
        time_uid = r[time_uid_col] if time_uid_col else np.nan
        cost_uid = r[cost_uid_col] if cost_uid_col else np.nan

        top_time = r.get("top_patches_time_ko","")
        top_cost = r.get("top_patches_cost_ko","")

        lines = []
        lines.append(f"[UID {uid}] 설계변경 패치 요약입니다.")
        if pd.notna(n_patch): lines.append(f"- 총 패치 수: {int(n_patch)}개")
        else: lines.append(f"- 총 패치 수: N/A")

        if pd.notna(time_uid): lines.append(f"- 보정 총 시간: {fmt_time_min(time_uid)} (≈{float(time_uid):.1f}분)")
        else: lines.append(f"- 보정 총 시간: N/A")

        if pd.notna(cost_uid): lines.append(f"- 보정 총 비용: {fmt_cost(cost_uid)}")
        else: lines.append(f"- 보정 총 비용: N/A")

        if isinstance(top_time, str) and top_time.strip():
            lines.append("\n시간 기여 상위 패치 WHY:")
            lines.append(top_time.strip())

        if isinstance(top_cost, str) and top_cost.strip():
            lines.append("\n비용 기여 상위 패치 WHY:")
            lines.append(top_cost.strip())

        text_ko = "\n".join(lines)

        meta = {}
        for c in df_uid.columns:
            if c in ["top_patches_time_ko","top_patches_cost_ko"]: continue
            v = r.get(c, None)
            if isinstance(v, (float, np.floating)) and np.isnan(v): continue
            meta[c] = v

        kb_rows.append({
            "id": f"{uid}_UID",
            "uid": uid,
            "level": "UID",
            "text_ko": text_ko,
            "meta_json": json.dumps(meta, ensure_ascii=False)
        })

    # 4-2) PATCH 레벨 KB
    meta_keys = [
        "patch_global_id","patch_type","pair_id",
        "geom_class_auto","process_family_auto","process_code_auto",
        "machine_group_auto","setup_type_auto","difficulty_auto","nc_type_auto",
        "delta_max","delta_avg","delta_min","abs_delta_max","abs_delta_avg",
        "area_est","diag_mm","bbox_z_span","bbox_xy_area_ratio",
        "delta_dir","normal_main_axis","side_hint","is_visible_zplus",
        "time_min_pred","time_min_calib","cost_krw_pred","cost_krw_calib",
        "time_scale_uid","cost_scale_uid"
    ]

    for i, r in df_patch.iterrows():
        uid = r["uid"]
        if uid is None: continue
        gid = r.get("patch_global_id", i)
        text_ko = str(r.get("why_total_ko",""))

        meta = {"uid": uid}
        for k in meta_keys:
            if k in df_patch.columns:
                v = r.get(k, None)
                if isinstance(v, (float, np.floating)) and np.isnan(v): continue
                meta[k] = v

        try:
            pid_str = f"{int(gid):06d}"
        except:
            pid_str = f"{i:06d}"

        kb_rows.append({
            "id": f"{uid}_PATCH_{pid_str}",
            "uid": uid,
            "level": "PATCH",
            "text_ko": text_ko,
            "meta_json": json.dumps(meta, ensure_ascii=False)
        })

    df_kb = pd.DataFrame(kb_rows)

    # ------------------------------
    # 5) 저장 (CSV + JSONL)
    # ------------------------------
    OUT_CSV   = os.path.join(OUT_DIR, "S_AGENT_KNOWLEDGE_RAG.csv")
    OUT_JSONL = os.path.join(OUT_DIR, "S_AGENT_KNOWLEDGE_RAG.jsonl")

    df_kb.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in kb_rows:
            meta = json.loads(r["meta_json"]) if isinstance(r["meta_json"], str) else r["meta_json"]
            line = {
                "id": r["id"],
                "uid": r["uid"],
                "level": r["level"],
                "text_ko": r["text_ko"],
                "meta": meta
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    _print("====================================")
    _print("[14R] KB 저장 완료")
    _print(f" - CSV  : {OUT_CSV}")
    _print(f" - JSONL: {OUT_JSONL}")
    _print("------------------------------------")
    _print(str(df_kb["level"].value_counts()))
    _print("===== 14R RAG KB EXPORT 완료 =====")

if __name__ == "__main__":
    main()