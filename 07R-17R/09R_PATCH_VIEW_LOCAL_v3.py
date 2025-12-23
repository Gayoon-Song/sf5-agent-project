# ============================================================
# 09R_PATCH_VIEW_LOCAL_v3.py  (FULL REPLACEMENT • ASCII-safe)
#
# 역할
#   - 08R 산출물(UID_###_BASE_COARSE.stl, UID_###_PATCH*.stl)을 겹쳐서
#     HTML 뷰어(Plotly) 생성.
#
# 실행 예
#   - 배치:  python 09R_PATCH_VIEW_LOCAL_v3.py
#   - 단건:  python 09R_PATCH_VIEW_LOCAL_v3.py --uid 151
# ============================================================

import os
import re
import sys
import argparse
import numpy as np

# ---------- deps ----------
try:
    import trimesh
except ImportError as e:
    print("[ERROR] trimesh not installed. Activate env then: pip install trimesh")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError as e:
    print("[ERROR] plotly not installed. Activate env then: pip install plotly")
    sys.exit(1)

# ---------- utils ----------
def _s(x: str) -> str:
    return str(x).encode("ascii", "ignore").decode("ascii")

def _print(msg: str):
    print(_s(msg))

# ---------- Root resolver ----------
def pick_root_dir(cli_root=None,
                  default_local=r"C:\sf5\sfsdh3",
                  default_colab="/content/drive/MyDrive/sfsdh3"):
    if cli_root and os.path.isdir(cli_root):
        return os.path.abspath(cli_root)
    env_sf5 = os.getenv("SF5_ROOT", "").strip()
    if env_sf5 and os.path.isdir(env_sf5):
        return os.path.abspath(env_sf5)
    if os.path.isdir(default_colab):
        return default_colab
    return default_local

def _uid3_from_any(x):
    if x is None: return None
    s = str(x).strip()
    if s == "": return None
    if s.isdigit(): return f"{int(s):03d}"
    nums = re.findall(r"\d+", s)
    if nums: return f"{int(nums[-1]):03d}"
    try: return f"{int(float(s)):03d}"
    except Exception: return None

def _find_first(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

# ---------- STL loaders ----------
def _load_stl(path_candidates, label):
    path = _find_first(path_candidates)
    if path is None:
        _print(f"[WARN] {label} STL not found: {path_candidates[0]}")
        return None
    try:
        m = trimesh.load(path, force="mesh")
        if m.is_empty:
            _print(f"[WARN] {label} STL empty: {path}")
            return None
        _print(f"[OK]   {label} STL: {os.path.basename(path)}")
        return m
    except Exception as e:
        _print(f"[ERR]  {label} load failed: {e}")
        return None

def _wireframe_trace(mesh, name="Base", color="lightgray", opacity=0.30):
    edges = mesh.edges_unique
    verts = mesh.vertices
    seg = verts[edges.reshape(-1)]
    seg = seg.reshape(-1, 2, 3)
    nan = np.full((seg.shape[0], 1, 3), np.nan)
    pts = np.concatenate([seg[:, 0:1, :], seg[:, 1:2, :], nan], axis=1).reshape(-1, 3)
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="lines",
        line=dict(width=1, color=color),
        opacity=opacity, name=name,
    )

def _mesh_trace(mesh, name="Patch", color="red", opacity=0.98):
    v, f = mesh.vertices, mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity, flatshading=True, name=name,
    )

# ---------- core ----------
def make_overlay_html_for_uid(uid3: str, stl_root: str, out_root: str):
    uid_tag = f"UID_{uid3}"
    uid_dir = os.path.join(stl_root, uid_tag)

    base_candidates = [
        os.path.join(uid_dir, f"{uid_tag}_BASE_COARSE.stl"),
        os.path.join(uid_dir, f"{uid_tag}_BASE.stl"),
    ]
    patch_candidates = [
        os.path.join(uid_dir, f"{uid_tag}_PATCH.stl"),
        os.path.join(uid_dir, f"{uid_tag}_PATCH_ADDED.stl"),
    ]

    _print(f"\n===== {uid_tag} start =====")
    base_mesh  = _load_stl(base_candidates,  "BASE")
    patch_mesh = _load_stl(patch_candidates, "PATCH")

    if base_mesh is None and patch_mesh is None:
        _print(f"[WARN] {uid_tag}: no STL -> skip")
        return None

    traces = []
    if base_mesh is not None:
        traces.append(_wireframe_trace(base_mesh, name="Base (wireframe)", color="lightgray", opacity=0.30))
    if patch_mesh is not None:
        traces.append(_mesh_trace(patch_mesh, name="Patch (changed)", color="red", opacity=0.98))

    # Camera auto-fit logic
    bounds = []
    if base_mesh is not None:  bounds.append(base_mesh.bounds)
    if patch_mesh is not None: bounds.append(patch_mesh.bounds)

    if bounds:
        b = np.array(bounds)
        xyz_min = b[:, 0, :].min(axis=0)
        xyz_max = b[:, 1, :].max(axis=0)
        center = (xyz_min + xyz_max) / 2
        dx, dy, dz = xyz_max - xyz_min
        radius = float(max(dx, dy, dz) * 1.8)
    else:
        radius = 1000.0
        center = np.zeros(3)

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),  # default angle
                center=dict(x=0, y=0, z=0)
            ),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{uid_tag} Analysis Result",
    )

    os.makedirs(out_root, exist_ok=True)
    out_html = os.path.join(out_root, f"{uid_tag}_overlay.html")
    fig.write_html(out_html, include_plotlyjs="cdn")
    _print(f"[SAVE] {out_html}")
    return out_html

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--uid_start", type=int, default=1)
    ap.add_argument("--uid_end",   type=int, default=14)
    ap.add_argument("--uid",       default=None)
    ap.add_argument("--latest",    action="store_true")
    # pipeline compat
    ap.add_argument("--before", default=None)
    ap.add_argument("--after",  default=None)
    args, _ = ap.parse_known_args()

    ROOT = pick_root_dir(args.root)
    STL_ROOT = os.path.join(ROOT, "08R_PATCH_STL")
    OUT_ROOT = os.path.join(ROOT, "09R_PATCH_HTML")

    # UID 목록 결정
    uid_list = []
    if args.uid:
        u = _uid3_from_any(args.uid)
        if u: uid_list = [u]
    elif args.latest:
        # 실제 존재하는 UID_* 디렉토리의 숫자 최대값 1개
        if os.path.isdir(STL_ROOT):
            nums = []
            for d in os.listdir(STL_ROOT):
                if d.startswith("UID_") and os.path.isdir(os.path.join(STL_ROOT, d)):
                    m = re.search(r"UID_(\d+)$", d)
                    if m:
                        try: nums.append(int(m.group(1)))
                        except: pass
            if nums:
                uid_list = [f"{max(nums):03d}"]
    else:
        # Batch range
        uid_list = [f"{i:03d}" for i in range(args.uid_start, args.uid_end + 1)]

    _print("===================================")
    _print("===== 09R_PATCH_VIEW_LOCAL_v3 =====")
    _print("ROOT     : " + ROOT)
    _print("STL_ROOT : " + STL_ROOT)
    _print("OUT_ROOT : " + OUT_ROOT)
    
    if len(uid_list) == 1:
        _print("MODE     : SINGLE (UID_" + uid_list[0] + ")")
    else:
        _print(f"MODE     : BATCH ({len(uid_list)} UIDs)")
    _print("===================================")

    for u3 in uid_list:
        make_overlay_html_for_uid(u3, STL_ROOT, OUT_ROOT)

    _print("===== 09R DONE =====")

if __name__ == "__main__":
    main()