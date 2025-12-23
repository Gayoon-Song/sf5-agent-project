# ================================================================
# 08R_EXPORT_BASE_COARSE_STL.py  (FULL REPLACEMENT • Py38-safe)
#   - 기본: ONLINE (단일 파일 처리)
#   - 배치: --batch 플래그 명시 시에만 루프 실행
# ================================================================

import os
import re
import sys
import argparse
from typing import Optional

# ---------- ASCII-safe print ----------
def _s(x: str) -> str:
    return str(x).encode("ascii", "ignore").decode("ascii")
def _print(msg: str):
    print(_s(msg))

# ---------- Root resolver ----------
def pick_root_dir(default_local=r"C:\sf5\sfsdh3",
                  default_colab="/content/drive/MyDrive/sfsdh3"):
    env_sf5 = os.getenv("SF5_ROOT", "").strip()
    if env_sf5 and os.path.isdir(env_sf5):
        return env_sf5
    if os.path.isdir("/content/drive/MyDrive") and os.path.isdir(default_colab):
        return default_colab
    return default_local

ROOT_DIR = pick_root_dir()

def _env_float(k: str, default: float) -> float:
    try:
        return float(os.getenv(k, str(default)))
    except Exception:
        return default

def _env_int(k: str, default: int) -> int:
    try:
        return int(float(os.getenv(k, str(default))))
    except Exception:
        return default

MESH_DEFLECTION = _env_float("SF8_MESH_DEFLECT", 5.0)
UID_START = _env_int("SF8_UID_START", 1)
UID_END   = _env_int("SF8_UID_END", 14)

OUT_DIR_NAME = "08R_PATCH_STL"

def uid3(n: int) -> str:
    return f"{n:03d}"

def extract_uid_num(uid_raw: str) -> Optional[int]:
    if uid_raw is None: return None
    s = str(uid_raw).strip()
    if s == "": return None
    if s.isdigit(): return int(s)
    nums = re.findall(r"\d+", s)
    if nums: return int(nums[-1])
    try: return int(float(s))
    except Exception: return None

def norm_uid3(uid_raw: Optional[str], default_uid3: str = "001") -> str:
    n = extract_uid_num(uid_raw) if uid_raw is not None else None
    return (f"{n:03d}" if n is not None else default_uid3)

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopoDS import TopoDS_Shape
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.StlAPI import StlAPI_Writer
except ImportError as e:
    _print("pythonocc-core import failed: " + str(e))
    _print("activate env:  conda activate sdh_occ")
    _print("install     :  conda install -n sdh_occ -c conda-forge pythonocc-core")
    sys.exit(1)

def load_step_shape(path: str) -> "TopoDS_Shape":
    if not os.path.isfile(path):
        raise FileNotFoundError("STEP file not found: " + path)
    rdr = STEPControl_Reader()
    st = rdr.ReadFile(path)
    if st != IFSelect_RetDone:
        raise RuntimeError("STEP read failed: " + path)
    if rdr.TransferRoots() == 0:
        raise RuntimeError("STEP->Shape transfer failed: " + path)
    return rdr.OneShape()

def mesh_and_export_stl(shape: "TopoDS_Shape", out_path: str, deflection: float):
    _print(f"  meshing (deflection={deflection:.3f}) ...")
    BRepMesh_IncrementalMesh(shape, deflection)
    _print("  writing STL -> " + out_path)
    writer = StlAPI_Writer()
    ok = writer.Write(shape, out_path)
    if not ok:
        raise RuntimeError("STL write failed: " + out_path)

def process_uid(uid3_str: str):
    _print(f"[UID_{uid3_str}] start BASE_COARSE")
    step_before = os.path.join(ROOT_DIR, "01_raw_L0", f"UID_{uid3_str}_before.stp")
    if not os.path.isfile(step_before):
        raise FileNotFoundError("missing BEFORE STEP: " + step_before)

    out_dir = os.path.join(ROOT_DIR, OUT_DIR_NAME, f"UID_{uid3_str}")
    os.makedirs(out_dir, exist_ok=True)
    out_stl = os.path.join(out_dir, f"UID_{uid3_str}_BASE_COARSE.stl")

    _print("  load STEP (before): " + step_before)
    shape_before = load_step_shape(step_before)
    _print("  STEP loaded")

    mesh_and_export_stl(shape_before, out_stl, deflection=MESH_DEFLECTION)
    _print(f"[UID_{uid3_str}] done")

def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--uid", default=None)
    ap.add_argument("--before", default=None)
    ap.add_argument("--after", default=None)   # 호환용
    ap.add_argument("--root", default=None)
    ap.add_argument("--batch", action="store_true", help="명시 시에만 배치 루프 실행")
    args, _unknown = ap.parse_known_args()

    # 루트 경로 재설정
    global ROOT_DIR
    if args.root:
        ROOT_DIR = os.path.abspath(args.root)

    _print("===== 08R BASE COARSE START =====")
    _print("ROOT_DIR       : " + ROOT_DIR)
    _print(f"MESH_DEFLECT   : {MESH_DEFLECTION:.3f}")
    _print("OUT_DIR_NAME   : " + OUT_DIR_NAME)

    # 1) 온라인 모드: --before (필수)
    if args.before:
        u3 = norm_uid3(args.uid, default_uid3="001")
        _print("MODE           : ONLINE single")
        _print("UID            : " + u3)
        _print("BEFORE STEP    : " + args.before)

        shape_before = load_step_shape(args.before)
        out_dir = os.path.join(ROOT_DIR, OUT_DIR_NAME, f"UID_{u3}")
        os.makedirs(out_dir, exist_ok=True)
        out_stl = os.path.join(out_dir, f"UID_{u3}_BASE_COARSE.stl")
        mesh_and_export_stl(shape_before, out_stl, deflection=MESH_DEFLECTION)
        _print("[SAVE] " + out_stl)
        _print("===== 08R BASE COARSE DONE (ONLINE) =====")
        return

    # 2) 배치 모드: --batch 플래그가 있어야만 실행 (안전장치)
    if args.batch:
        _print("MODE           : BATCH (Explicit)")
        _print(f"UID RANGE      : {UID_START:03d} ~ {UID_END:03d}")

        any_fail = False
        for n in range(UID_START, UID_END + 1):
            u3 = uid3(n)
            try:
                process_uid(u3)
            except Exception as e:
                any_fail = True
                _print(f"[ERROR] UID_{u3}: " + str(e))
        if any_fail:
            _print("DONE with errors (see logs)")
            sys.exit(2)
        _print("===== 08R BASE COARSE DONE (BATCH) =====")
        return

    _print("MODE           : NONE (use --before for single or --batch for loop)")

if __name__ == "__main__":
    main()