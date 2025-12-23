# ==============================================================
# 07R_FACE_PATCH_VIEW_LOCAL.py  (CRITICAL FIX • ONLINE SUPPORT)
#
# 역할
#   - 04S 결과를 바탕으로 변경 부위를 로컬 3D 창에 시각화
#   - 에이전트 파이프라인(s_pipeline)에서 호출될 때 에러 없이
#     해당 UID만 띄우거나, 실패 시 안전하게 종료(skip)하는 것이 목표
#
# 수정 사항
#   - --uid 인자 처리 확실하게 보장
#   - CSV 파일 경로 탐색 로직 강화 (UID_xxx vs LIVE_xxx)
#   - GUI 없는 환경 등에서 에러 발생 시 파이프라인 중단 방지 (try-except)
# ==============================================================

import os
import sys
import csv
import re
import argparse
import shutil
from typing import List, Dict, Set

# 0) ascii-safe print
def _s(x: str) -> str:
    return str(x).encode("ascii", "ignore").decode("ascii")

def _print(msg: str):
    print(_s(msg))

# 1) root dir
def resolve_root(root_cli=None):
    if root_cli: return os.path.abspath(root_cli)
    env_root = os.environ.get("SF5_ROOT", "").strip()
    if env_root: return os.path.abspath(env_root)
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))

ROOT_DIR = resolve_root()

# 2) params
DELTA_THRESH = 0.05
USE_PAIR_PATCH = True
USE_ADDED_FACE = True
USE_REMOVED_FACE = True

# 3) pythonocc import (GUI 의존성)
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Compound, topods_Face
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.BRep import BRep_Builder
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.Quantity import Quantity_NOC_RED, Quantity_NOC_GRAY75, Quantity_NOC_BLUE1
    HAS_GUI = True
except ImportError:
    _print("[WARN] pythonocc-core or display not available. 07R View skipped.")
    HAS_GUI = False
except Exception as e:
    _print(f"[WARN] pythonocc init failed: {e}. 07R View skipped.")
    HAS_GUI = False

# 4) utils
def _extract_uid_num(uid_raw: str):
    if uid_raw is None: return None
    s = str(uid_raw).strip()
    if s == "": return None
    nums = re.findall(r"\d+", s)
    if nums: return int(nums[-1])
    return None

def _uid3(uid_raw: str):
    n = _extract_uid_num(uid_raw)
    return f"{n:03d}" if n is not None else None

def load_step_shape(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"STEP file not found: {path}")
    rdr = STEPControl_Reader()
    st = rdr.ReadFile(path)
    if st != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {path}")
    rdr.TransferRoots()
    return rdr.OneShape()

def build_face_list(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(topods_Face(exp.Current()))
        exp.Next()
    return faces

def load_patch_features_csv(uid_raw: str):
    # 1. Try exact path match first (UID_195 -> 04S.../UID_195/UID_195_PATCH_FEATURES.csv)
    # 스크린샷 구조: 04S_FACE_PATCH / UID_195 / UID_195_PATCH_FEATURES
    
    # uid_raw가 "UID_195" 형태라고 가정
    tgt_dir = os.path.join(ROOT_DIR, "04S_FACE_PATCH", uid_raw)
    tgt_csv = os.path.join(tgt_dir, f"{uid_raw}_PATCH_FEATURES.csv")
    
    # 파일이 없으면 숫자만 추출해서 다시 시도 ("195" -> "UID_195")
    if not os.path.isfile(tgt_csv):
        u3 = _uid3(uid_raw)
        if u3:
            alt_tag = f"UID_{u3}"
            tgt_csv = os.path.join(ROOT_DIR, "04S_FACE_PATCH", alt_tag, f"{alt_tag}_PATCH_FEATURES.csv")

    # 그래도 없으면 확장자가 없는 파일인지 확인 (스크린샷에 확장자 없는 경우 대비)
    if not os.path.isfile(tgt_csv):
        no_ext = os.path.splitext(tgt_csv)[0]
        if os.path.isfile(no_ext):
            tgt_csv = no_ext

    if not os.path.isfile(tgt_csv):
        raise FileNotFoundError(f"PATCH_FEATURES CSV not found for {uid_raw}")

    rows = []
    # 인코딩 호환성 (utf-8-sig 우선)
    try:
        with open(tgt_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except:
        with open(tgt_csv, "r", encoding="cp949") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    return rows

def collect_changed_face_indices(patch_rows, delta_thresh):
    before_set, after_set = set(), set()
    for r in patch_rows:
        try:
            ptype = str(r.get("patch_type", "")).strip()
            dmax = float(r.get("delta_max", 0.0))
            if abs(dmax) < delta_thresh: continue
            
            b_idx = int(float(r.get("before_face_index", -1)))
            a_idx = int(float(r.get("after_face_index", -1)))

            if ptype == "pair_patch" and USE_PAIR_PATCH:
                if b_idx >= 0: before_set.add(b_idx)
                if a_idx >= 0: after_set.add(a_idx)
            elif ptype == "added_face" and USE_ADDED_FACE:
                if a_idx >= 0: after_set.add(a_idx)
            elif ptype == "removed_face" and USE_REMOVED_FACE:
                if b_idx >= 0: before_set.add(b_idx)
        except:
            continue
    return before_set, after_set

def make_compound(face_list, indices):
    builder = BRep_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for i in indices:
        if 0 <= i < len(face_list):
            builder.Add(comp, face_list[i])
    return comp

def view_uid(uid_raw, before_path=None, after_path=None):
    if not HAS_GUI:
        _print("No GUI environment. Skipping visualization.")
        return

    _print(f"[{uid_raw}] 07R Visualization Start")
    
    # 1. STEP 파일 경로 확정
    # s_pipeline에서 넘어온 before/after 경로가 있으면 그걸 최우선으로 사용
    if not (before_path and os.path.exists(before_path)):
        # 없으면 표준 경로에서 탐색
        u3 = _uid3(uid_raw)
        uid_tag = f"UID_{u3}"
        before_path = os.path.join(ROOT_DIR, "01_raw_L0", f"{uid_tag}_before.stp")
        after_path  = os.path.join(ROOT_DIR, "01_raw_L0", f"{uid_tag}_after.stp")
    
    if not os.path.exists(before_path) or not os.path.exists(after_path):
        _print(f"[ERR] STEP files missing for {uid_raw}. Skip.")
        return

    # 2. 로드
    shape_b = load_step_shape(before_path)
    shape_a = load_step_shape(after_path)
    faces_b = build_face_list(shape_b)
    faces_a = build_face_list(shape_a)

    # 3. 변경 부위 추출
    try:
        rows = load_patch_features_csv(uid_raw)
        b_set, a_set = collect_changed_face_indices(rows, DELTA_THRESH)
        _print(f"Changed faces: Before={len(b_set)}, After={len(a_set)}")
    except Exception as e:
        _print(f"[WARN] Failed to load patch CSV ({e}). Showing raw shapes only.")
        b_set, a_set = set(), set()

    comp_b = make_compound(faces_b, b_set)
    comp_a = make_compound(faces_a, a_set)

    # 4. Display
    display, start_display, _, _ = init_display()
    
    # Base (After) - Gray Transparent
    display.DisplayShape(shape_a, update=False, color=Quantity_NOC_GRAY75, transparency=0.8)
    
    # Changed (After) - Red
    if a_set:
        display.DisplayShape(comp_a, update=False, color=Quantity_NOC_RED, transparency=0.1)
        
    # Changed (Before) - Blue (Optional, for removed)
    if b_set:
        display.DisplayShape(comp_b, update=False, color=Quantity_NOC_BLUE1, transparency=0.6)

    display.FitAll()
    start_display()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", default=None)
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)
    ap.add_argument("--root", default=None)
    # 파이프라인 호환용 (무시)
    ap.add_argument("--force_if_missing", action="store_true")
    
    args, _ = ap.parse_known_args()

    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    # 온라인 모드 (파이프라인 호출)
    if args.uid:
        try:
            view_uid(args.uid, args.before, args.after)
        except Exception as e:
            # 07R은 '뷰어'이므로 실패해도 파이프라인 전체를 죽이지 않도록 함
            _print(f"[07R ERROR] Visualization failed but pipeline continues: {e}")
            sys.exit(0) # 정상 종료 처리하여 파이프라인 유지
    else:
        _print("[07R] No UID provided. Nothing to show.")

if __name__ == "__main__":
    main()