"""Generate a synthetic 'raw dataset' in the D-Hydro layout expected by
database/graph_creation.py::create_mesh_dataset.

The real simulations live on Zenodo (10.5281/zenodo.13326595), which is
unreachable from this network. This script builds a small, self-consistent
stand-in so the train/validate/test code paths can be exercised end to end.

The physics is FAKE. Only the file formats and mesh topology are faithful.

Layout produced under <out>/:
    Geometry/Polygon_{i}.pol
    DEM/DEM_{i}.xyz
    Hydrograph/Hydrograph_{i}.txt
    Simulations/output_{i}_map.nc
"""
import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset as NCDataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from database.graph_creation import create_mesh_dhydro  # noqa: E402

HOURS = 49            # hourly outputs -> 48 h of simulation
DOMAIN = 400.0        # domain side length [m]
VERTEX_SPACING = 50.0 # polygon vertex spacing -> controls coarsest mesh size


def write_polygon(path: str, rng: np.random.Generator) -> None:
    """Write a D-Hydro-style .pol file (2 header lines, then 'x,y' rows)."""
    n_side = int(DOMAIN // VERTEX_SPACING)
    s = np.linspace(0.0, DOMAIN, n_side + 1)
    bottom = np.stack([s, np.zeros_like(s)], -1)
    right = np.stack([np.full_like(s, DOMAIN), s], -1)
    top = np.stack([s[::-1], np.full_like(s, DOMAIN)], -1)
    left = np.stack([np.zeros_like(s), s[::-1]], -1)
    ring = np.concatenate([bottom[:-1], right[:-1], top[:-1], left[:-1]], 0)

    # small jitter on the interior of each side keeps the meshes non-identical
    jitter = rng.uniform(-4.0, 4.0, size=ring.shape)
    on_corner = ((ring == 0.0) | (ring == DOMAIN)).all(1)
    jitter[on_corner] = 0.0
    ring = ring + jitter
    ring = np.concatenate([ring, ring[:1]], 0)  # close the ring

    with open(path, "w") as f:
        f.write("BL01\n")
        f.write(f"    {ring.shape[0]}    2\n")
        for x, y in ring:
            f.write(f"{x:.6f},{y:.6f}\n")


def write_dem(path: str, rng: np.random.Generator) -> np.ndarray:
    """Write an x,y,z DEM covering the domain. Returns the raw array."""
    g = np.linspace(-10.0, DOMAIN + 10.0, 60)
    xx, yy = np.meshgrid(g, g)
    # gentle tilt + one smooth bump, so slopes are non-degenerate
    zz = (0.01 * xx + 0.004 * yy
          + 1.5 * np.exp(-(((xx - 250) ** 2 + (yy - 150) ** 2) / (2 * 70.0 ** 2)))
          + rng.uniform(-0.05, 0.05, xx.shape))
    dem = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], -1)
    np.savetxt(path, dem, fmt="%.6f")
    return dem


def write_hydrograph(path: str, rng: np.random.Generator) -> None:
    """Write 'time[s] discharge' rows, one per hourly output step."""
    t_s = np.arange(HOURS) * 3600.0
    peak = rng.uniform(8.0, 25.0)
    t_peak = rng.uniform(6.0, 14.0)
    hours = t_s / 3600.0
    q = peak * (hours / t_peak) ** 2 * np.exp(2 * (1 - hours / t_peak))
    np.savetxt(path, np.stack([t_s, q], -1), fmt="%.6f")


def pick_single_bc_edge(mesh) -> int:
    """Pick exactly one boundary edge as the BC edge.

    interpolate_BC_location_multiscale() asserts exactly one BC edge per mesh,
    so the finest mesh must declare exactly one.
    """
    boundary = np.where(mesh.edge_type > 1)[0]
    mid = mesh.node_xy[mesh.edge_index[:, boundary].T].mean(1)
    # choose the boundary edge whose midpoint is closest to the middle of the
    # x=0 side -- a stable, deterministic choice away from the corners
    target = np.array([0.0, DOMAIN / 2])
    return int(boundary[np.linalg.norm(mid - target, axis=1).argmin()])


def write_netcdf(path: str, mesh, bc_edge: int, dem: np.ndarray,
                 rng: np.random.Generator) -> None:
    """Write a D-Hydro-like map file for `mesh` (a graph_creation.Mesh)."""
    num_nodes = mesh.node_x.shape[0]
    num_faces = mesh.face_x.shape[0]
    num_edges = mesh.edge_index.shape[1]

    # --- edge_type: 1 normal, 2 the single BC edge, 3 other boundary edges ---
    edge_type = mesh.edge_type.astype(np.int32).copy()
    edge_type[edge_type > 1] = 3
    edge_type[bc_edge] = 2

    # --- edge_faces (1-based, 0 == missing) --------------------------------
    # graph_creation reads face_BC from slot 0 == -1 and face_bnd from slot 1 == -1,
    # so orientation of the missing entry encodes which boundary edge carries the BC.
    edge_faces = np.zeros((num_edges, 2), dtype=np.int32)
    face_of_edge = [[] for _ in range(num_edges)]
    node_pos = 0
    edge_lookup = {}
    for e in range(num_edges):
        a, b = mesh.edge_index[0, e], mesh.edge_index[1, e]
        edge_lookup[(min(a, b), max(a, b))] = e
    for fi, npf in enumerate(mesh.nodes_per_face):
        fn = mesh.face_nodes[node_pos:node_pos + npf]
        node_pos += npf
        for j in range(npf):
            a, b = fn[j], fn[(j + 1) % npf]
            e = edge_lookup.get((min(a, b), max(a, b)))
            if e is not None:
                face_of_edge[e].append(fi)
    for e, faces in enumerate(face_of_edge):
        if len(faces) >= 2:
            edge_faces[e] = [faces[0] + 1, faces[1] + 1]
        elif len(faces) == 1:
            if e == bc_edge:
                edge_faces[e] = [0, faces[0] + 1]      # BC boundary edge
            else:
                edge_faces[e] = [faces[0] + 1, 0]      # plain boundary edge
        else:
            raise RuntimeError(f"edge {e} borders no face")

    # --- face_nodes as a padded (num_faces, max_npf) float array -----------
    max_npf = int(mesh.nodes_per_face.max())
    face_nodes = np.full((num_faces, max_npf), np.nan)
    node_pos = 0
    for fi, npf in enumerate(mesh.nodes_per_face):
        face_nodes[fi, :npf] = mesh.face_nodes[node_pos:node_pos + npf] + 1  # 1-based
        node_pos += npf

    # --- fake water depth / velocity fields --------------------------------
    fx, fy = mesh.face_x, mesh.face_y
    dist = np.linalg.norm(np.stack([fx - 0.0, fy - DOMAIN / 2], -1), axis=1)
    t = np.arange(HOURS)[:, None]
    front = 12.0 * t                                   # flood front speed [m/h]
    wd = np.clip(1.2 * (1 - dist[None, :] / np.maximum(front, 1.0)), 0.0, None)
    wd = wd * (1 + 0.05 * rng.standard_normal(wd.shape))
    wd = np.clip(wd, 0.0, None)
    wet = wd > 0
    ucx = np.where(wet, 0.4 * wd, 0.0)
    ucy = np.where(wet, 0.1 * wd * np.sin(fy[None, :] / 50.0), 0.0)

    with NCDataset(path, "w", format="NETCDF4") as nc:
        nc.createDimension("nmesh2d_node", num_nodes)
        nc.createDimension("nmesh2d_face", num_faces)
        nc.createDimension("nmesh2d_edge", num_edges)
        nc.createDimension("Two", 2)
        nc.createDimension("max_nmesh2d_face_nodes", max_npf)
        nc.createDimension("time", HOURS)

        def var(name, dtype, dims, data, fill=None):
            kw = {"fill_value": fill} if fill is not None else {}
            v = nc.createVariable(name, dtype, dims, **kw)
            v[:] = data
            return v

        var("mesh2d_node_x", "f8", ("nmesh2d_node",), mesh.node_x)
        var("mesh2d_node_y", "f8", ("nmesh2d_node",), mesh.node_y)
        var("mesh2d_face_x", "f8", ("nmesh2d_face",), mesh.face_x)
        var("mesh2d_face_y", "f8", ("nmesh2d_face",), mesh.face_y)
        var("mesh2d_edge_nodes", "i4", ("nmesh2d_edge", "Two"),
            mesh.edge_index.T + 1)
        var("mesh2d_edge_type", "i4", ("nmesh2d_edge",), edge_type)
        var("mesh2d_edge_faces", "i4", ("nmesh2d_edge", "Two"), edge_faces)
        # NaN padding -> xarray yields a masked array -> mixed-mesh branch
        var("mesh2d_face_nodes", "f8",
            ("nmesh2d_face", "max_nmesh2d_face_nodes"), face_nodes)
        var("time", "f8", ("time",), np.arange(HOURS) * 3600.0)
        var("mesh2d_waterdepth", "f8", ("time", "nmesh2d_face"), wd)
        var("mesh2d_ucx", "f8", ("time", "nmesh2d_face"), ucx)
        var("mesh2d_ucy", "f8", ("time", "nmesh2d_face"), ucy)


def build_one(out_dir: str, sim_id: int, n_scales: int) -> None:
    rng = np.random.default_rng(1000 + sim_id)
    pol = os.path.join(out_dir, "Geometry", f"Polygon_{sim_id}.pol")
    dem_f = os.path.join(out_dir, "DEM", f"DEM_{sim_id}.xyz")
    hyd = os.path.join(out_dir, "Hydrograph", f"Hydrograph_{sim_id}.txt")
    nc_f = os.path.join(out_dir, "Simulations", f"output_{sim_id}_map.nc")

    write_polygon(pol, rng)
    dem = write_dem(dem_f, rng)
    write_hydrograph(hyd, rng)

    # The finest mesh must be the one create_mesh_dhydro produces for the
    # simulation, i.e. the last mesh of a `number_of_multiscales=n_scales` run.
    meshes = create_mesh_dhydro(pol, number_of_multiscales=n_scales,
                                for_simulation=False)
    fine = meshes[-1]
    bc_edge = pick_single_bc_edge(fine)
    write_netcdf(nc_f, fine, bc_edge, dem, rng)
    print(f"  sim {sim_id}: {fine.face_x.shape[0]} faces, "
          f"{fine.edge_index.shape[1]} edges -> {os.path.basename(nc_f)}",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(REPO, "database", "raw_datasets_synth"))
    p.add_argument("--ids", default="1-6,81-82",
                   help="comma-separated ids/ranges, e.g. '1-6,81-82'")
    p.add_argument("--n-scales", type=int, default=4)
    args = p.parse_args()

    ids = []
    for chunk in args.ids.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(chunk))

    for sub in ("Geometry", "DEM", "Hydrograph", "Simulations"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    print(f"writing synthetic raw dataset to {args.out}")
    for sim_id in ids:
        build_one(args.out, sim_id, args.n_scales)
    print("done")


if __name__ == "__main__":
    main()
