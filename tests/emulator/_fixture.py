# tests/emulator/_fixture.py
import h5py, numpy as np

PARAM_NAMES = ["ns","Ap","herei","heref","alphaq","hub","omegamh2","hireionz","bhfeedback"]

def write_synthetic_cache(path, n_sims=3, snaps_per_sim=2, n_alpha=4, n_k=8, seed=0):
    rng = np.random.default_rng(seed)
    G = n_sims * snaps_per_sim
    R = G * n_alpha
    P_tier_c_filt = np.empty((R, 15, n_k))
    P_tier_c_unfilt = np.empty((R, 15, n_k))
    counts = np.empty((R, 15), dtype=np.int64)
    P_tier_p = np.empty((R, n_k))
    kf = np.tile(np.linspace(1e-3, 0.1, n_k), (R, 1))
    params = np.empty((R, 9)); alpha_slope = np.empty(R); alpha_idx = np.empty(R, np.int32)
    target_F = np.empty(R); scale = np.empty(R); z_meta = np.empty(R); z_grid = np.empty(R)
    dv_kms = np.full(R, 7.0); nbins_native = np.empty(R, np.int32); group_idx = np.empty(R, np.int32)
    sim_name = np.empty(R, object)
    snap_f_nhi = np.empty((G, 30)); snap_n_abs = np.empty((G, 30), np.int64)
    snap_path = np.empty(G); snap_sim = np.empty(G, object); snap_snap = np.empty(G, np.int32)
    snap_dndx = {c: np.empty(G) for c in ("LLS","subDLA","DLA")}
    alpha_grid = np.linspace(0.66, 1.33, n_alpha)
    r = 0
    for s in range(n_sims):
        p = rng.uniform(0.5, 1.5, 9)
        for j in range(snaps_per_sim):
            g = s * snaps_per_sim + j
            z = 2.0 + 0.4 * j
            cnt15 = np.array([700, 60,40,30,20,15,10,8, 12,9,7,5,4, 3,2], np.int64)
            snap_f_nhi[g] = rng.uniform(1e-22, 1e-20, 30)
            snap_n_abs[g] = rng.integers(0, 50, 30)
            snap_path[g] = 1000.0; snap_sim[g] = f"sim{s}"; snap_snap[g] = j
            for c in snap_dndx: snap_dndx[c][g] = rng.uniform(0.01, 0.4)
            nyq = n_k - (g % 2)
            for a in range(n_alpha):
                base = rng.uniform(0.5, 2.0, (15, n_k))
                P_tier_c_filt[r] = base
                P_tier_c_unfilt[r] = base + rng.uniform(-0.05, 0.05, (15, n_k))
                counts[r] = cnt15
                w = cnt15 / cnt15.sum()
                Pp = np.einsum("c,ck->k", w, base)
                if nyq < n_k:
                    Pp[nyq:] = np.nan
                    P_tier_c_filt[r, :, nyq:] = np.nan
                    P_tier_c_unfilt[r, :, nyq:] = np.nan
                P_tier_p[r] = Pp
                params[r] = p; alpha_slope[r] = alpha_grid[a]; alpha_idx[r] = a
                z_meta[r] = z; z_grid[r] = round(z/0.2)*0.2; nbins_native[r] = 100 + g
                target_F[r] = np.exp(-alpha_grid[a] * 2.3e-3 * (1+z)**3.65)
                scale[r] = 1.0; group_idx[r] = g; sim_name[r] = f"sim{s}"
                r += 1
    with h5py.File(path, "w") as h:
        h.attrs.update(cache_version="3.3", n_k=n_k, n_rows=R, n_snaps=G,
                       tau_thresh=1e6, tier_c_recipe="uniform")
        h["P_tier_c_filtered"] = P_tier_c_filt; h["P_tier_c"] = P_tier_c_unfilt
        h["tier_c_counts"] = counts; h["P_tier_p"] = P_tier_p; h["kfkms"] = kf
        h["params"] = params; h["alpha_slope"] = alpha_slope; h["alpha_idx"] = alpha_idx
        h["target_F"] = target_F; h["scale"] = scale; h["z_meta"] = z_meta; h["z_grid"] = z_grid
        h["dv_kms"] = dv_kms; h["nbins_native"] = nbins_native; h["snap_group_idx"] = group_idx
        h.create_dataset("sim_name", data=np.array(sim_name, dtype=object), dtype=h5py.string_dtype())
        h.create_dataset("param_names", data=np.array(PARAM_NAMES, dtype=object), dtype=h5py.string_dtype())
        h["snap_f_nhi"] = snap_f_nhi; h["snap_n_absorbers"] = snap_n_abs
        h["snap_total_path_dX"] = snap_path; h["snap_snap"] = snap_snap
        h.create_dataset("snap_sim_name", data=np.array(snap_sim, dtype=object), dtype=h5py.string_dtype())
        for c in snap_dndx: h[f"snap_dNdX_{c}"] = snap_dndx[c]
