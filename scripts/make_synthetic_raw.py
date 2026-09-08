"""Generate a small synthetic 'raw_datasets_synth' folder in the exact layout that
database/graph_creation.py:create_mesh_dataset expects (D-Hydro map netcdf + DEM xyz
+ hydrograph txt + polygon pol), so train/val/test code paths can be smoke-tested
without the Zenodo download.

Run from repo root inside the mswegnn env:
    python plan/make_synthetic_raw.py --n_sim 6 --out database/raw_datasets_synth
The water-depth field is an analytic radial flood wave, NOT a hydraulic simulation.
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.getcwd())
from database.graph_creation import create_mesh_dhydro  # noqa: E402

HOURS = 96          # simulation length [h] (matches overview.csv)
DT_MIN = 60         # netcdf output interval [min] (utils.dataset assumes 60)


def rot(xy, theta, shift):
    """Rotate (N,2) points by theta and translate. Needed because graph_creation's
    boundary-node test compares raw coordinates and breaks on axis-aligned domains."""
    c, s = np.cos(theta), np.sin(theta)
    return np.asarray(xy) @ np.array([[c, -s], [s, c]]).T + shift


def write_polygon(path, L, W, ds, theta, shift):
    nx, ny = max(2, int(round(L / ds))), max(2, int(round(W / ds)))
    xs = np.linspace(0, L, nx + 1)
    ys = np.linspace(0, W, ny + 1)
    pts = ([(x, 0.0) for x in xs] + [(L, y) for y in ys[1:]]
           + [(x, W) for x in xs[::-1][1:]] + [(0.0, y) for y in ys[::-1][1:]])
    pts = rot(pts, theta, shift)
    with open(path, "w") as f:
        f.write("synthetic_polygon\n")
        f.write(f"{len(pts)} 2\n")
        for x, y in pts:
            f.write(f"{x:.3f},{y:.3f}\n")


def make_sim(i, out, rng):
    L = float(rng.uniform(900, 1100))
    W = float(rng.uniform(700, 900))
    ds = 200.0
    theta = float(rng.uniform(np.deg2rad(8), np.deg2rad(80)))
    shift = np.array([2000.0, 2000.0])
    pol = os.path.join(out, "Geometry", f"Polygon_{i}.pol")
    write_polygon(pol, L, W, ds, theta, shift)
    inflow_target = rot([(0.0, W / 2)], theta, shift)[0]

    # finest mesh, same call chain the original D-Hydro setup used
    m2d = create_mesh_dhydro(pol, number_of_multiscales=4, for_simulation=True)
    node_x, node_y = m2d.node_x.astype(float), m2d.node_y.astype(float)
    face_x, face_y = m2d.face_x.astype(float), m2d.face_y.astype(float)
    edge_nodes = m2d.edge_nodes.reshape(-1, 2).astype(int)
    edge_faces = m2d.edge_faces.reshape(-1, 2).astype(int)
    nodes_per_face = m2d.nodes_per_face.astype(int)
    face_nodes_flat = m2d.face_nodes.astype(int)
    nF, nE, nN = len(face_x), len(edge_nodes), len(node_x)

    # boundary edges = one missing face; pick ONE BC edge nearest to (0, W/2)
    is_bnd = (edge_faces < 0).any(1)
    mid = np.stack([node_x[edge_nodes].mean(1), node_y[edge_nodes].mean(1)], 1)
    cand = np.where(is_bnd)[0]
    bc_edge = cand[np.argmin(np.linalg.norm(mid[cand] - inflow_target, axis=1))]

    edge_type = np.ones(nE, dtype=np.int32)
    edge_type[is_bnd] = 3
    edge_type[bc_edge] = 2

    # D-Hydro convention used by Mesh._import_from_map_netcdf:
    #   BC edge -> [-1, face] ; other boundary edge -> [face, -1]
    ef = edge_faces.copy()
    for e in cand:
        f = ef[e][ef[e] >= 0][0]
        ef[e] = [-1, f] if e == bc_edge else [f, -1]

    # padded face_nodes (nF, maxN), 1-based, fill = -999 -> _FillValue
    maxN = nodes_per_face.max()
    fn = np.full((nF, maxN), -999, dtype=np.int32)
    p = 0
    for k, n in enumerate(nodes_per_face):
        fn[k, :n] = face_nodes_flat[p:p + n] + 1
        p += n

    # DEM: plane sloping away from the inflow + smooth bumps; on a regular grid
    x0, x1 = node_x.min() - 50, node_x.max() + 50
    y0, y1 = node_y.min() - 50, node_y.max() + 50
    gx, gy = np.meshgrid(np.linspace(x0, x1, 60), np.linspace(y0, y1, 50))
    bc_xy = mid[bc_edge]
    def dem_fn(x, y):
        d = np.hypot(x - bc_xy[0], y - bc_xy[1])
        return 0.0015 * d + 0.3 * np.sin(x / 150.0) * np.cos(y / 120.0) + 5.0
    np.savetxt(os.path.join(out, "DEM", f"DEM_{i}.xyz"),
               np.stack([gx.ravel(), gy.ravel(), dem_fn(gx, gy).ravel()], 1), fmt="%.3f")

    # hydrograph: [time(s), Q(m3/s)], one row per netcdf frame (hourly)
    t_h = np.arange(0, HOURS + 1)
    peak = float(rng.uniform(60, 140))
    Q = np.where(t_h <= 12, peak * t_h / 12.0, np.clip(peak * (1 - (t_h - 12) / 36.0), 0, None))
    np.savetxt(os.path.join(out, "Hydrograph", f"Hydrograph_{i}.txt"),
               np.stack([t_h * 3600.0, Q], 1), fmt="%.3f")

    # analytic flood wave on face centres: depth = level(t) - k*dist - DEM offset
    dem_f = dem_fn(face_x, face_y)
    dist = np.hypot(face_x - bc_xy[0], face_y - bc_xy[1])
    vol = np.cumsum(Q) * 3600.0                      # m3
    level = 0.0015 * np.sqrt(vol)                     # rough pond level rise [m], ~2-5 m peak
    WD = np.clip(level[:, None] - 0.004 * dist[None, :] - (dem_f - dem_f.min())[None, :], 0, None)
    WD = WD.astype(np.float32)                        # (T, nF)
    ux = (face_x - bc_xy[0]) / (dist + 1e-6)
    uy = (face_y - bc_xy[1]) / (dist + 1e-6)
    speed = 0.8 * WD / (1.0 + WD) * (Q[:, None] / max(peak, 1.0))
    VX = (speed * ux[None, :]).astype(np.float32)
    VY = (speed * uy[None, :]).astype(np.float32)

    ds_nc = xr.Dataset(
        {
            "mesh2d_node_x": ("nmesh2d_node", node_x),
            "mesh2d_node_y": ("nmesh2d_node", node_y),
            "mesh2d_face_x": ("nmesh2d_face", face_x),
            "mesh2d_face_y": ("nmesh2d_face", face_y),
            "mesh2d_edge_nodes": (("nmesh2d_edge", "Two"), (edge_nodes + 1).astype(np.int32)),
            "mesh2d_edge_type": ("nmesh2d_edge", edge_type),
            "mesh2d_edge_faces": (("nmesh2d_edge", "Two"), (ef + 1).astype(np.int32)),
            "mesh2d_face_nodes": (("nmesh2d_face", "max_nmesh2d_face_nodes"), fn),
            "mesh2d_waterdepth": (("time", "nmesh2d_face"), WD),
            "mesh2d_ucx": (("time", "nmesh2d_face"), VX),
            "mesh2d_ucy": (("time", "nmesh2d_face"), VY),
        },
        coords={"time": t_h * 3600.0},
    )
    enc = {"mesh2d_face_nodes": {"_FillValue": -999}}
    ds_nc.to_netcdf(os.path.join(out, "Simulations", f"output_{i}_map.nc"), encoding=enc)
    return nF, nE, nN, WD.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sim", type=int, default=6)
    ap.add_argument("--out", default="database/raw_datasets_synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for sub in ["Simulations", "DEM", "Hydrograph", "Geometry"]:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = ["seed,mesh_num_faces,simulation_time[h],computation_time[s]"]
    for i in range(1, a.n_sim + 1):
        nF, nE, nN, wdmax = make_sim(i, a.out, rng)
        rows.append(f"{i},{nF},{HOURS:.1f},{60.0:.4f}")
        print(f"sim {i}: faces={nF} edges={nE} nodes={nN} maxWD={wdmax:.2f} m")
    with open(os.path.join(a.out, "overview.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
