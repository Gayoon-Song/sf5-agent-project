# ============================================================
# 15R_RAG_SEARCH_LOCAL_UID150_v7.py  (FULL REPLACEMENT • ASCII-safe)
#
# 역할
#   - 14R KB 로드 및 인덱싱 (FAISS/Cosine)
#   - 자연어 쿼리 검색 (UID 우선/벡터 검색)
#
# 실행 예
#   - 기본: python 15R_RAG_SEARCH_LOCAL_UID150_v7.py
#   - 쿼리: python 15R_RAG_SEARCH_LOCAL_UID150_v7.py --query "리브 두께 변경 이유"
#   - 재빌드: python 15R_RAG_SEARCH_LOCAL_UID150_v7.py --rebuild
# ============================================================

import os, re, sys, subprocess, importlib, json, time, argparse, hashlib
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

# -----------------------------
# 1) pip auto-install helpers
# -----------------------------
def _pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    subprocess.call(cmd)

def _ensure_pkg(pkg, import_name=None):
    try:
        return importlib.import_module(import_name or pkg)
    except Exception:
        _pip_install([pkg])
        return importlib.import_module(import_name or pkg)

_ensure_pkg("sentence-transformers", "sentence_transformers")
from sentence_transformers import SentenceTransformer

faiss = None
try:
    import faiss
except Exception:
    try:
        _pip_install(["faiss-cpu"])
        import faiss
    except Exception:
        faiss = None
        _print("[15R][WARN] faiss import 실패 → cosine fallback 사용")

# -----------------------------
# 2) CSV robust loader
# -----------------------------
def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    return pd.read_csv(path)

# -----------------------------
# 3) Path Settings
# -----------------------------
def build_paths(BASE_DIR):
    KB_DIR   = os.path.join(BASE_DIR, "14R_AGENT_KB")
    KB_CSV   = os.path.join(KB_DIR, "S_AGENT_KNOWLEDGE_RAG.csv")
    INDEX_DIR   = os.path.join(BASE_DIR, "15R_RAG_INDEX")
    FAISS_PATH  = os.path.join(INDEX_DIR, "faiss.index")
    EMB_PATH    = os.path.join(INDEX_DIR, "embeddings.npy")
    KB_CACHE_PQ = os.path.join(INDEX_DIR, "kb_compact.parquet")
    KB_CACHE_CSV= os.path.join(INDEX_DIR, "kb_compact.csv")
    META_PATH   = os.path.join(INDEX_DIR, "kb_meta.json")
    LOG_PATH    = os.path.join(INDEX_DIR, "search_log.csv")
    os.makedirs(INDEX_DIR, exist_ok=True)
    return KB_CSV, INDEX_DIR, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH, LOG_PATH

def find_file_under(root, filename):
    for r, _, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None

# -----------------------------
# 4) KB Load + doc_text 구성
# -----------------------------
def load_kb_from_csv(kb_csv):
    df = read_csv_robust(kb_csv)

    if "level" not in df.columns:
        cand = [c for c in df.columns if c.lower()=="level"]
        df["level"] = df[cand[0]] if cand else "PATCH"

    if "uid" not in df.columns and "UID" in df.columns:
        df["uid"] = df["UID"]
    df["uid"] = df["uid"].astype(str).str.extract(r"(\d+)")[0].fillna(df["uid"]).str.zfill(3)

    if "patch_global_id" not in df.columns:
        cand = [c for c in df.columns if c.lower() in ["patch_global_id","patch_id","global_id","patch_idx"]]
        if cand:
            df["patch_global_id"] = pd.to_numeric(df[cand[0]], errors="coerce")
        else:
            df["patch_global_id"] = np.where(
                df["level"].astype(str).str.upper()=="PATCH",
                np.arange(len(df), dtype=int), -1
            )

    ignore = {"uid","UID","level","patch_global_id","uid_num","uid_kind","pair_id","meta_json","score"}
    text_cols = [c for c in df.columns if c not in ignore and df[c].dtype==object]
    priority_keys = ["why","rule","summary","explain","ko","desc","text","근거","공정"]
    ordered = []
    for key in priority_keys:
        ordered += [c for c in text_cols if key in c.lower() and c not in ordered]
    ordered += [c for c in text_cols if c not in ordered]

    def build_doc_text(row):
        parts = []
        for c in ordered:
            v = row.get(c, "")
            if pd.notna(v):
                s = str(v).strip()
                if s and s.lower()!="nan": parts.append(s)
        if not parts:
            lvl = str(row["level"]).upper()
            if lvl == "UID": return f"UID {row['uid']} summary"
            return f"UID {row['uid']} PATCH {row.get('patch_global_id','?')}"
        uniq=[]; 
        for p in parts:
            if p not in uniq: uniq.append(p)
        return "\n".join(uniq[:10])

    df["doc_text"] = df.apply(build_doc_text, axis=1).fillna("").astype(str)
    return df

# -----------------------------
# 5) KB 메타(강화)
# -----------------------------
BUILD_VER = "15R_v7p_local"

def file_sha1(path, block=1<<20):
    h = hashlib.sha1()
    with open(path,"rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def current_kb_meta(kb_csv, df_head=None):
    stat = os.stat(kb_csv)
    meta = {
        "kb_csv": kb_csv, "mtime": stat.st_mtime, "size": stat.st_size,
        "sha1": file_sha1(kb_csv), "build_ver": BUILD_VER,
    }
    if df_head is not None:
        meta["cols"] = df_head.columns.tolist()
        meta["n_head"] = int(df_head.shape[0])
    else:
        try:
            meta["cols"] = read_csv_robust(kb_csv).head(0).columns.tolist()
            meta["n_head"] = 0
        except:
            meta["cols"] = []; meta["n_head"] = 0
    return meta

def load_saved_meta(META_PATH):
    if not os.path.exists(META_PATH): return None
    try: return json.load(open(META_PATH,"r",encoding="utf-8"))
    except: return None

def save_meta(META_PATH, meta):
    json.dump(meta, open(META_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def need_rebuild_kb_cache(saved, curr):
    if saved is None: return True
    keys = ["kb_csv","mtime","size","sha1","cols","build_ver"]
    for k in keys:
        if saved.get(k) != curr.get(k): return True
    return False

# -----------------------------
# 6. Parquet 캐시 fallback
# -----------------------------
def write_kb_cache(df, KB_CACHE_PQ, KB_CACHE_CSV):
    try:
        df.to_parquet(KB_CACHE_PQ, index=False)
        return "parquet"
    except Exception as e:
        df.to_csv(KB_CACHE_CSV, index=False, encoding="utf-8-sig")
        _print(f"[15R][WARN] parquet 저장 실패 -> CSV 캐시 대체: {e}")
        return "csv"

def read_kb_cache(KB_CACHE_PQ, KB_CACHE_CSV):
    if os.path.exists(KB_CACHE_PQ):
        try: return pd.read_parquet(KB_CACHE_PQ)
        except: pass
    if os.path.exists(KB_CACHE_CSV):
        return read_csv_robust(KB_CACHE_CSV)
    return None

# -----------------------------
# 7. Embedding + Index
# -----------------------------
def need_rebuild_emb(df, EMB_PATH):
    if not os.path.exists(EMB_PATH): return True
    try:
        emb_cached = np.load(EMB_PATH)
        return emb_cached.shape[0] != len(df)
    except: return True

# -----------------------------
# 8. UID 추출
# -----------------------------
uid_pat = re.compile(r"(?:uid\s*[:#]?\s*)?(\d{1,6})", re.IGNORECASE)

def extract_uid_from_query(q, df_kb=None):
    m = uid_pat.search(str(q))
    if not m: return None
    uid = str(int(m.group(1))).zfill(3)
    if df_kb is not None and "uid" in df_kb.columns:
        if uid in set(df_kb["uid"].astype(str).unique()): return uid
        return None
    return uid

# -----------------------------
# 9. 결과 포맷/로그
# -----------------------------
def pretty_print(df):
    show = [c for c in ["level","uid","patch_global_id","score","doc_text"] if c in df.columns]
    print(df[show].head(10).to_string(index=False))

def log_query(LOG_PATH, query, k, level, n_res):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    header = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", encoding="utf-8-sig") as f:
        if header: f.write("ts,query,k,level,n_results\n")
        q = str(query).replace("\n"," ").replace(","," ")
        f.write(f"{ts},{q},{k},{level or ''},{n_res}\n")

# -----------------------------
# 10. 메인
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--kb_csv", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--level", default=None)
    parser.add_argument("--no_uid_first", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    # pipeline compat
    parser.add_argument("--rebuild_if_needed", action="store_true")
    parser.add_argument("--clear_index", action="store_true")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_faiss", action="store_true")
    args = parser.parse_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    BASE = ROOT_DIR
    KB_CSV, INDEX_DIR, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH, LOG_PATH = build_paths(BASE)

    if args.kb_csv:
        KB_CSV = os.path.abspath(args.kb_csv)

    if not os.path.exists(KB_CSV):
        alt = find_file_under(BASE, "S_AGENT_KNOWLEDGE_RAG.csv")
        if alt is None:
            _print(f"[15R][ERROR] KB CSV not found. Expected: {KB_CSV}")
            return
        KB_CSV = alt

    _print("====================================")
    _print("===== 15R RAG SEARCH (LOCAL v7+) =====")
    _print(" BASE_DIR : " + BASE)
    _print(" KB_CSV   : " + KB_CSV)
    _print(" INDEX_DIR: " + INDEX_DIR)

    if args.clear_index:
        for fn in ["faiss.index","embeddings.npy","kb_compact.parquet","kb_compact.csv","kb_meta.json"]:
            fp = os.path.join(INDEX_DIR, fn)
            try:
                if os.path.exists(fp): os.remove(fp)
            except: pass
        _print("[15R] Index cleared.")

    global faiss
    st_model = SentenceTransformer(args.model)
    if args.no_faiss: faiss = None

    # Meta check
    df_head = read_csv_robust(KB_CSV).head(5)
    curr_meta  = current_kb_meta(KB_CSV, df_head)
    saved_meta = load_saved_meta(META_PATH)

    rebuild_flag = args.rebuild
    if args.rebuild_if_needed and need_rebuild_kb_cache(saved_meta, curr_meta):
        rebuild_flag = True

    if rebuild_flag or need_rebuild_kb_cache(saved_meta, curr_meta):
        _print("[15R] KB changed/rebuild -> Refresh Cache")
        df_kb = load_kb_from_csv(KB_CSV)
        write_kb_cache(df_kb, KB_CACHE_PQ, KB_CACHE_CSV)
        save_meta(META_PATH, curr_meta)
    else:
        df_kb = read_kb_cache(KB_CACHE_PQ, KB_CACHE_CSV)
        if df_kb is None:
            _print("[15R][WARN] Cache corrupted -> Reload")
            df_kb = load_kb_from_csv(KB_CSV)
            write_kb_cache(df_kb, KB_CACHE_PQ, KB_CACHE_CSV)
            save_meta(META_PATH, curr_meta)
        else:
            _print("[15R] Cache hit.")

    _print(f"[15R] KB shape: {df_kb.shape}")

    need_emb = rebuild_flag or need_rebuild_emb(df_kb, EMB_PATH)
    if need_emb:
        _print("[15R] Building Embeddings...")
        texts = df_kb["doc_text"].fillna("").astype(str).tolist()
        emb = st_model.encode(texts, batch_size=max(8, args.batch_size), show_progress_bar=True, normalize_embeddings=True)
        emb = np.asarray(emb, dtype="float32")
        np.save(EMB_PATH, emb)
        if faiss is not None:
            dim = emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb)
            faiss.write_index(index, FAISS_PATH)
        else:
            index = None
    else:
        emb = np.load(EMB_PATH)
        if faiss is not None and os.path.exists(FAISS_PATH):
            index = faiss.read_index(FAISS_PATH)
        else:
            index = None

    def search_once(query, k=5, level_filter=None, uid_first=True):
        q = str(query).strip()
        if not q: return pd.DataFrame()

        uid_q = extract_uid_from_query(q, df_kb=df_kb) if uid_first else None
        sub_df = df_kb
        if level_filter is not None and "level" in sub_df.columns:
            sub_df = sub_df[sub_df["level"].astype(str).str.upper()==level_filter.upper()]

        if uid_q is not None:
            sub_uid = sub_df[sub_df["uid"]==uid_q].copy()
            if len(sub_uid)>0:
                sub_uid = sub_uid.drop_duplicates(subset=["level","uid","patch_global_id"], keep="first").head(max(k,10))
                sub_uid["score"] = 1.0
                return sub_uid.head(k)

        q_emb = st_model.encode([q], normalize_embeddings=True).astype("float32")
        if index is not None:
            scores, idxs = index.search(q_emb, k*10)
            rows=[]; seen=set()
            for s,i in zip(scores[0], idxs[0]):
                r = df_kb.iloc[int(i)].to_dict()
                key = (r.get("level",""), r.get("uid",""), r.get("patch_global_id",-1))
                if key in seen: continue
                seen.add(key)
                r["score"]=float(s)
                rows.append(r)
                if len(rows) >= k*5: break
            out=pd.DataFrame(rows)
        else:
            sims=np.dot(emb, q_emb[0])
            top_idx=np.argsort(-sims)[:k*10]
            rows=[]; seen=set()
            for i in top_idx:
                r=df_kb.iloc[int(i)].to_dict()
                key = (r.get("level",""), r.get("uid",""), r.get("patch_global_id",-1))
                if key in seen: continue
                seen.add(key)
                r["score"]=float(sims[i])
                rows.append(r)
                if len(rows) >= k*5: break
            out=pd.DataFrame(rows)

        if "level" in out.columns and level_filter is not None:
            out = out[out["level"].astype(str).str.upper()==level_filter.upper()]
        
        if "score" in out.columns:
            out = out.sort_values("score", ascending=False)

        return out.head(k)

    if args.query:
        res = search_once(args.query, k=args.k, level_filter=args.level, uid_first=not args.no_uid_first)
        pretty_print(res)
        log_query(LOG_PATH, args.query, args.k, args.level, len(res))
        return

    _print("[15R] Ready. (Use --query to search)")

if __name__ == "__main__":
    main()