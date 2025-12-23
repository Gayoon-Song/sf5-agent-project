# ============================================================
# 16R_AGENT_WEB_LOCAL_UID150_v1.py  (FULL REPLACEMENT)
#
# 역할
#   - 15R 검색 엔진을 웹 UI(Gradio)로 제공 (검색 전용)
#   - "파일 업로드 및 추론" 기능은 app_gradio.py 사용 권장
#
# 실행
#   python 16R_AGENT_WEB_LOCAL_UID150_v1.py
# ============================================================

import os, re, sys, json, time, argparse, subprocess, importlib, hashlib
import numpy as np
import pandas as pd
from functools import lru_cache

# ---------- Root resolver ----------
def pick_base_dir(default_local=r"C:\sf5\sfsdh3"):
    env_root = os.environ.get("SF5_ROOT", "").strip()
    if env_root and os.path.isdir(env_root):
        return env_root
    return default_local

# ---------- pip helper ----------
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
_ensure_pkg("gradio", "gradio")
from sentence_transformers import SentenceTransformer
import gradio as gr

faiss = None
try:
    import faiss
except Exception:
    try:
        _pip_install(["faiss-cpu"])
        import faiss
    except Exception:
        faiss = None
        print("[16R][WARN] faiss import 실패 → cosine fallback 사용")

# ---------- CSV robust loader ----------
def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try: return pd.read_csv(path, encoding=enc)
        except: continue
    return pd.read_csv(path)

# ---------- Paths ----------
def build_paths(BASE_DIR):
    KB_DIR   = os.path.join(BASE_DIR, "14R_AGENT_KB")
    KB_CSV   = os.path.join(KB_DIR, "S_AGENT_KNOWLEDGE_RAG.csv")
    INDEX_DIR   = os.path.join(BASE_DIR, "15R_RAG_INDEX")
    FAISS_PATH  = os.path.join(INDEX_DIR, "faiss.index")
    EMB_PATH    = os.path.join(INDEX_DIR, "embeddings.npy")
    KB_CACHE_PQ = os.path.join(INDEX_DIR, "kb_compact.parquet")
    KB_CACHE_CSV= os.path.join(INDEX_DIR, "kb_compact.csv")
    META_PATH   = os.path.join(INDEX_DIR, "kb_meta.json")
    os.makedirs(INDEX_DIR, exist_ok=True)
    return KB_CSV, INDEX_DIR, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH

def find_file_under(root, filename):
    for r, _, files in os.walk(root):
        if filename in files: return os.path.join(r, filename)
    return None

# ---------- KB Load ----------
def load_kb_from_csv(kb_csv):
    df = read_csv_robust(kb_csv)
    if "level" not in df.columns:
        cand = [c for c in df.columns if c.lower()=="level"]
        df["level"] = df[cand[0]] if cand else "PATCH"
    df["level"] = df["level"].astype(str).str.upper()

    if "uid" not in df.columns and "UID" in df.columns: df["uid"] = df["UID"]
    df["uid"] = df["uid"].astype(str).str.extract(r"(\d+)")[0].fillna(df["uid"]).str.zfill(3)

    if "patch_global_id" not in df.columns:
        cand = [c for c in df.columns if c.lower() in ["patch_global_id","patch_id","global_id","patch_idx"]]
        if cand: df["patch_global_id"] = pd.to_numeric(df[cand[0]], errors="coerce")
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
        parts=[]
        for c in ordered:
            v=row.get(c,"")
            if pd.notna(v):
                s=str(v).strip()
                if s and s.lower()!="nan": parts.append(s)
        if not parts:
            lvl=str(row["level"]).upper()
            if lvl=="UID": return f"UID {row['uid']} 요약"
            return f"UID {row['uid']} PATCH {row.get('patch_global_id','?')}"
        uniq=[]; 
        for p in parts:
            if p not in uniq: uniq.append(p)
        return "\n".join(uniq[:10])

    df["doc_text"] = df.apply(build_doc_text, axis=1).fillna("").astype(str)
    return df

# ---------- KB Meta ----------
BUILD_VER = "16R_v1_local_reuse15R"

def file_sha1(path, block=1<<20):
    h=hashlib.sha1()
    with open(path,"rb") as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def current_kb_meta(kb_csv, df_head=None):
    stat = os.stat(kb_csv)
    meta = {
        "kb_csv": kb_csv, "mtime": stat.st_mtime, "size": stat.st_size,
        "sha1": file_sha1(kb_csv), "build_ver": BUILD_VER,
    }
    if df_head is not None: meta["cols"] = df_head.columns.tolist()
    else: meta["cols"] = read_csv_robust(kb_csv).head(0).columns.tolist()
    return meta

def load_saved_meta(META_PATH):
    if not os.path.exists(META_PATH): return None
    try: return json.load(open(META_PATH,"r",encoding="utf-8"))
    except: return None

def save_meta(META_PATH, meta):
    json.dump(meta, open(META_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def need_rebuild_kb_cache(saved, curr):
    if saved is None: return True
    for k in ["kb_csv","mtime","size","sha1","cols","build_ver"]:
        if saved.get(k)!=curr.get(k): return True
    return False

# ---------- Cache I/O ----------
def write_kb_cache(df, KB_CACHE_PQ, KB_CACHE_CSV):
    try:
        df.to_parquet(KB_CACHE_PQ, index=False)
        return "parquet"
    except Exception as e:
        df.to_csv(KB_CACHE_CSV, index=False, encoding="utf-8-sig")
        print(f"[16R][WARN] parquet 실패 -> CSV 캐시 사용: {e}")
        return "csv"

def read_kb_cache(KB_CACHE_PQ, KB_CACHE_CSV):
    if os.path.exists(KB_CACHE_PQ):
        try: return pd.read_parquet(KB_CACHE_PQ)
        except: pass
    if os.path.exists(KB_CACHE_CSV):
        return read_csv_robust(KB_CACHE_CSV)
    return None

# ---------- Embedding / Index ----------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
st_model = SentenceTransformer(MODEL_NAME)

def need_rebuild_emb(df, EMB_PATH):
    if not os.path.exists(EMB_PATH): return True
    try:
        emb_cached = np.load(EMB_PATH)
        return emb_cached.shape[0] != len(df)
    except: return True

# ---------- Global Load ----------
@lru_cache(maxsize=2)
def load_all(BASE_DIR, KB_CSV, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH, force_rebuild=False, nonce=0):
    if not os.path.exists(KB_CSV):
        alt = find_file_under(BASE_DIR, "S_AGENT_KNOWLEDGE_RAG.csv")
        if alt is None: raise FileNotFoundError(f"[16R][ERROR] KB CSV 없음: {KB_CSV}")
        KB_CSV = alt

    df_head = read_csv_robust(KB_CSV).head(5)
    curr_meta  = current_kb_meta(KB_CSV, df_head)
    saved_meta = load_saved_meta(META_PATH)

    if force_rebuild or need_rebuild_kb_cache(saved_meta, curr_meta):
        print("[16R] KB 변경/강제재생성 -> 캐시 최신화")
        df_kb = load_kb_from_csv(KB_CSV)
        mode = write_kb_cache(df_kb, KB_CACHE_PQ, KB_CACHE_CSV)
        save_meta(META_PATH, curr_meta)
    else:
        df_kb = read_kb_cache(KB_CACHE_PQ, KB_CACHE_CSV)
        if df_kb is None:
            print("[16R][WARN] 캐시 손상 -> KB 재로딩")
            df_kb = load_kb_from_csv(KB_CSV)
            mode = write_kb_cache(df_kb, KB_CACHE_PQ, KB_CACHE_CSV)
            save_meta(META_PATH, curr_meta)
        else:
            print("[16R] KB 캐시 재사용")

    if need_rebuild_emb(df_kb, EMB_PATH) or force_rebuild:
        print("[16R] 임베딩/인덱스 재생성")
        texts = df_kb["doc_text"].fillna("").astype(str).tolist()
        emb = st_model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
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

    print(f"[16R] KB shape: {df_kb.shape}")
    return df_kb, emb, index, KB_CSV

# ---------- Searcher ----------
uid_pat = re.compile(r"(?:uid\s*[:#]?\s*)?(\d{1,6})", re.IGNORECASE)

def extract_uid_from_query(q, df_kb=None):
    m = uid_pat.search(str(q))
    if not m: return None
    uid = str(int(m.group(1))).zfill(3)
    if df_kb is not None and "uid" in df_kb.columns:
        if uid in set(df_kb["uid"].astype(str).unique()): return uid
        return None
    return uid

def rag_search(df_kb, emb, index, query, k=5, level_filter=None, uid_filter=None, prefer_uid_first=True):
    q = str(query).strip()
    if not q: return []

    sub_df = df_kb
    if level_filter:
        sub_df = sub_df[sub_df["level"].astype(str).str.upper()==level_filter.upper()]
    if uid_filter:
        sub_df = sub_df[sub_df["uid"]==str(uid_filter).zfill(3)]

    uid_q = extract_uid_from_query(q, df_kb=df_kb) if prefer_uid_first else None
    if uid_q is not None:
        sub_uid = sub_df[sub_df["uid"]==uid_q].copy()
        if len(sub_uid)>0:
            sub_uid = sub_uid.drop_duplicates(subset=["level","uid","patch_global_id"], keep="first")
            sub_uid["score"]=1.0
            return sub_uid.head(max(k,10)).to_dict("records")

    q_emb = st_model.encode([q], normalize_embeddings=True).astype("float32")
    if index is not None:
        scores, idxs = index.search(q_emb, k*5)
        rows=[]; seen=set()
        for s,i in zip(scores[0], idxs[0]):
            r = df_kb.iloc[int(i)].to_dict()
            key=(r.get("level",""), r.get("uid",""), r.get("patch_global_id",-1))
            if key in seen: continue
            seen.add(key)
            r["score"]=float(s)
            rows.append(r)
            if len(rows)>=k*5: break
        out=pd.DataFrame(rows)
    else:
        sims=np.dot(emb, q_emb[0])
        top_idx=np.argsort(-sims)[:k*10]
        rows=[]; seen=set()
        for i in top_idx:
            r=df_kb.iloc[int(i)].to_dict()
            key=(r.get("level",""), r.get("uid",""), r.get("patch_global_id",-1))
            if key in seen: continue
            seen.add(key)
            r["score"]=float(sims[i])
            rows.append(r)
            if len(rows)>=k*5: break
        out=pd.DataFrame(rows)

    if level_filter:
        out = out[out["level"].astype(str).str.upper()==level_filter.upper()]
    if uid_filter:
        out = out[out["uid"]==str(uid_filter).zfill(3)]

    if "score" in out.columns:
        out = out.sort_values("score", ascending=False)
    return out.head(k).to_dict("records")

# ---------- Web UI ----------
def build_ui(load_fn, BASE_DIR, KB_CSV, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH, k_default=5):
    with gr.Blocks(title="16R Search Only") as demo:
        gr.Markdown("## 설계변경 RAG 검색 (16R)")
        with gr.Row():
            with gr.Column(scale=5):
                q = gr.Textbox(label="질문", lines=2, placeholder="검색어 입력")
                btn = gr.Button("검색")
            with gr.Column(scale=7):
                out = gr.Textbox(label="결과", lines=20)
        
        # event
        def on_search(query):
            df_kb, emb, index, _ = load_fn(BASE_DIR, KB_CSV, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH)
            res = rag_search(df_kb, emb, index, query, k=k_default)
            if not res: return "결과 없음"
            txt = ""
            for r in res:
                txt += f"[{r.get('score',0):.2f}] UID{r.get('uid')} {r.get('level')}\n{r.get('doc_text','')}\n\n"
            return txt

        btn.click(on_search, inputs=[q], outputs=[out])
        q.submit(on_search, inputs=[q], outputs=[out])
    return demo

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--kb_csv", default=None)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    BASE_DIR = os.path.abspath(args.root) if args.root else pick_base_dir()
    KB_CSV, INDEX_DIR, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH = build_paths(BASE_DIR)
    if args.kb_csv: KB_CSV = os.path.abspath(args.kb_csv)

    print("===== 16R SEARCH ONLY =====")
    print(" BASE :", BASE_DIR)

    demo = build_ui(load_all, BASE_DIR, KB_CSV, FAISS_PATH, EMB_PATH, KB_CACHE_PQ, KB_CACHE_CSV, META_PATH)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()