"""Universal transit score, generation 5: the effect of the transit service of an H3 res-10 cell on choosing transit over the car.

    $TORCH_PYTHON fit/torch_fit4.py <experiment> [variant ...]      (FLEX_VARIANT=buf)

Definition agreed with the user (2026-11-13):
  outcome     commuters by transit / (transit + car): walking, cycling, work-from-home and other modes are ignored. Four tasks share the curves:
              all transit, bus, rail (subway + commuter rail), tram (light rail), each against the same transit + car denominator.
  service     cell value A = max over the stops reachable from the cell of  M(mode) * H(headway) * S(speed) * W_mode(walk) (walking channel,
              <= 2 km) and, with the driving channel, A_far = max over the stations reachable by road (2-20 km) of M * H * S * F(road km).
              Smooth curves with few parameters (no free tables, so no irregular curves):
                H(h)    = 1 / (1 + (h / h50)^kh)            headway, h50 = headway at half value
                S(v)    = 1 / (1 + (v50 / v)^kv)            speed
                W_m(d)  = 1 / (1 + (d / d50_m)^kw)          walking, one half-value distance per mode, one shared steepness
                F(d)    = 1 / (1 + (d / f50)^kf)            driving to a station
                M       = (bus, tram, rail = 1)
              Block group: population-weighted mean of its cells.
  link        logit p = alpha_k + log(1 + A / eps_k) [+ thf_k log(1 + A_far / epsf_k)] [+ controls]   (slope 1: odds proportional to A / eps)
  score       direct = log(1 + A / eps_total) [+ thf log(1 + A_far / epsf)]   -- the cell's service term, 0 with no service
              total  = direct + gamma * beta_c * log(1 + A / eps_c)           -- plus the part through car ownership (SEM variants)
              score  = 100 * value / P95 (population-weighted over US res-10 cells), capped at 100. Nothing regional or socio-economic enters it.
  levels      raw       service only (+ intercepts, dispersion)
              controls  + density, 1 km density, jobs, distance to the CBD, metro size, income, poverty, education, age, fuel price, the workplace end
                        (transit access where the residents work, LODES flows) and metro intercepts (ridge)
              sem       + car ownership: car-free household share = f(service, controls) (beta-binomial), entering the transit equations as a mediator
                        (gamma * logit of the observed car-free share), + a shared block-group random term (latent transit propensity, ridge)
  checks      10 random starts (stability), metro-grouped 5-fold held-out skill (variability), the same fit on each Census region and on each
              third of the 1 km density (universality: the curves must agree over the service each region / setting actually has).
"""
from __future__ import annotations

import json
import os
import re
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch_fit as tf  # noqa: E402
import torch_fit2 as t2  # noqa: E402
import torch_fit3 as t3  # noqa: E402
from torch_fit import DEV, bb_loglik  # noqa: E402

RUNS = tf.RUNS
TASKS = ["total", "bus", "rail", "tram"]
WALK_MAX_M = 2000.0
import os as _os0
if _os0.environ.get("FIT4_WALK_MAX"):   # 2026-11-15 test 2: where the walking channel ends and the far (2-20 km) channel begins
    WALK_MAX_M = float(_os0.environ["FIT4_WALK_MAX"]); t3.WALK_MAX_M = WALK_MAX_M
# 2026-11-15 test 1 (user: "How do you know if it's car or walk?" -- the data have no access mode): one continuous distance curve over every
# item up to 20 km instead of a walking channel (<= 2 km) plus a separate far channel
ONECURVE = _os0.environ.get("FIT4_ONECURVE") == "1"
# 2026-11-15 (user): the headway curve is 1 at the minimum headway considered and flat below it, like the walk curve at its floor
H_FLOOR = float(_os0.environ.get("FIT4_H_FLOOR", "0") or 0)          # minutes; 0 = off (the runs queued before this option)
CONTROLS_PLUS = _os0.environ.get("FIT4_CONTROLS_PLUS") == "1"         # extra block-group controls (fit/build_extra_controls.py), car speed included
NO_METRO_RE = _os0.environ.get("FIT4_NO_METRO_RE") == "1"             # no metro intercepts (held-out metros get none either)
SPEED_CAR = _os0.environ.get("FIT4_SPEED_CAR") == "1"                 # speed curve on transit speed / the tract's car commute speed (x 20)
# 2026-11-17 (user: find a use for the stop speed and separate mode from speed: "the rail factor should not take the speed into account").
# FIT4_SPEED_VAR puts a log-linear speed term log S = b log(v / 20 km/h) next to the mode factor, so the mode factor is the value at equal speed
# (20 km/h); b is identified from the speed differences within each mode. Forms ('' keeps FIT4_SHAPE_S / FIT4_NO_SPEED):
#   power    one shared b
#   permode  one b per mode (bus, tram, rail): fast buses (express, BRT) against local ones, fast rail against slow
#   commute  b = b0 + b1 z(log car commute time of the tract): speed worth more where commutes are long
#   far      b only on items beyond 2 km (park-and-ride / feeder access: the long-trip riders)
#   nodecay  no speed in the walk-decay slope (c_v = 0): tests the one place speed is used today
SPEED_VAR = _os0.environ.get("FIT4_SPEED_VAR", "")
# FIT4_SPEED_CLIP=1: item speeds capped per mode (bus 5-70, tram 5-60, rail 10-130 km/h): end-to-end trip speeds reach 319 km/h for rail
# and 63+ km/h for 5 % of bus items (express / intercity patterns, bad shapes or times)
SPEED_CLIP = _os0.environ.get("FIT4_SPEED_CLIP") == "1"
# FIT4_FREEZE=name,name: parameters kept at their warm-start value (FIT4_INIT), also in the held-out folds. Used for the sequential split of
# speed and mode (user 2026-11-17: "make speed capture the most effect possible and mode avoid capturing any speed effect"): the speed terms
# fitted without a mode factor are frozen, then the mode factor only takes what speed leaves unexplained
FREEZE = [x for x in _os0.environ.get("FIT4_FREEZE", "").split(",") if x]
EXTRA2 = HERE / "data_buf" / "bg_extra2.parquet"
CONTROLS = ["log_pop_density", "log_job_density", "log_jobs_5km", "log_dist_cbd", "log_metro_pop", "log_income", "poverty_share", "bachelor_share",
            "under18_share", "over65_share"]
EXTRA = HERE / "data_buf" / "bg_extra.parquet"
# control study (2026-11-16): comma-separated control names to drop / add, and FIT4_NO_WORK=1 = no workplace-end transit access term
CTRL_DROP = {c for c in _os0.environ.get("FIT4_CTRL_DROP", "").split(",") if c}
CTRL_ADD = [c for c in _os0.environ.get("FIT4_CTRL_ADD", "").split(",") if c]
TF_FEATURES = {"zero_car_share", "cars_per_household", "commute_km", "commute_time", "renter_share"}
NO_WORK = _os0.environ.get("FIT4_NO_WORK") == "1"
CV_COLD = _os0.environ.get("FIT4_CV_COLD") == "1"         # held-out folds from the default start instead of the full-data fit (leak check)
CTRL_STUDY = _os0.environ.get("FIT4_CTRL_STUDY") == "1"   # clip the controls' extremes (the study's runs only, so the earlier runs stay reproducible)
EXTRA3 = HERE / "data_buf" / "bg_extra3.parquet"
# generalisation study (study 10, 2026-11-20): what makes the score carry over to metros the fit has not seen
WLAM = float(_os0.environ.get("FIT4_WLAM", "0.5"))          # fit weight = population x metro population ** -WLAM (0.5 = every run before; eval stays 0.5)
METRO_CTX = _os0.environ.get("FIT4_METRO_CTX") == "1"       # + b x (metro mean of the block groups' transit access term): how much of the metro the network reaches
CTRL_METRO_MEANS = _os0.environ.get("FIT4_CTRL_METRO_MEANS") == "1"   # + the metro (or county) population-weighted mean of every control: context effects
SAVE_HO = _os0.environ.get("FIT4_SAVE_HO", "1") == "1"      # held-out per-block-group predictions (<run>_ho.npz) for within-metro diagnostics


def _z(v: np.ndarray) -> np.ndarray:
    v = np.where(np.isfinite(v), v, np.nanmedian(v))
    return (v - v.mean()) / max(v.std(), 1e-9)


class Data4(t3.Data3):
    """Data3 (cells, items, flows) with the transit-vs-car outcomes, raw item speeds, car ownership and the extra block-group columns."""

    def __init__(self):
        super().__init__()
        t = self._tab
        tr = t["transit_commuters_b08301"].to_numpy(float); car = t["car_truck_van_commuters"].to_numpy(float)
        n = np.where(np.isfinite(tr) & np.isfinite(car), tr + car, np.nan)
        ok = np.isfinite(n) & (n > 0)
        T = lambda a: torch.tensor(np.where(ok, a, 0.0), dtype=torch.float32, device=DEV)  # noqa: E731
        self.y = T(np.minimum(np.maximum(tr, 0), np.where(ok, n, 0))); self.n = T(n); self.ok_total = torch.tensor(ok.astype(np.float32), device=DEV)
        ym = {"bus": t["bus_commuters"].to_numpy(float), "rail": (t["subway_commuters"].fillna(0) + t["commuter_rail_commuters"].fillna(0)).to_numpy(float),
              "tram": t["light_rail_commuters"].to_numpy(float)}
        self.n_mode = self.n
        self.y_mode = {m: T(np.minimum(np.maximum(np.nan_to_num(v), 0), np.where(ok, n, 0))) for m, v in ym.items()}
        self.ok_mode = {m: torch.tensor((ok & np.isfinite(v)).astype(np.float32), device=DEV) for m, v in ym.items()}
        # car ownership: households with no vehicle / all households
        hh_cols = ["acs5_vehiclesHouseholds0", "acs5_vehiclesHouseholds1", "acs5_vehiclesHouseholds2", "acs5_vehiclesHouseholds3Plus"]
        hh = t[hh_cols].sum(axis=1, min_count=4).to_numpy(float); h0 = t["acs5_vehiclesHouseholds0"].to_numpy(float)
        okc = np.isfinite(hh) & (hh > 0) & np.isfinite(h0)
        self.c_n = torch.tensor(np.where(okc, hh, 0), dtype=torch.float32, device=DEV)
        self.c_y = torch.tensor(np.where(okc, np.clip(h0, 0, np.where(okc, hh, 0)), 0), dtype=torch.float32, device=DEV)
        self.c_ok = torch.tensor(okc.astype(np.float32), device=DEV)
        share = np.where(okc, (h0 + 0.5) / (hh + 1.0), np.nan)   # mediator: logit of the observed car-free share (smoothed), centred
        lg = np.log(share / (1 - share)); self.carfree_logit = torch.tensor(np.nan_to_num(lg - np.nanmean(lg)), dtype=torch.float32, device=DEV)
        # raw item speeds (Data3 keeps raw headway it_h and distance it_d), in Data3's item order (walking items first)
        it = np.load(tf.TD / "items.npz"); far = it["d"].astype(np.float32) > t3.WALK_MAX_M
        order = np.argsort(far, kind="stable")
        v_raw = np.maximum(it["v"][order].astype(np.float32), 0.5)
        if SPEED_CLIP:
            md = it["mode"][order]
            lo, hi = np.array([5.0, 5.0, 10.0], np.float32)[md], np.array([70.0, 60.0, 130.0], np.float32)[md]
            v_raw = np.clip(v_raw, lo, hi)
        self.it_v = torch.tensor(v_raw, device=DEV)
        # extra block-group columns
        ex = pd.read_parquet(EXTRA).set_index("GEOID").reindex(pd.Index(self.geoid, name="GEOID"))
        self.region = ex["region"].fillna("").to_numpy(); self.division = ex["division"].fillna("").to_numpy()
        self.d1km = ex["pop_density_1km"].to_numpy(float)
        self.x_extra = np.stack([_z(np.log1p(ex["pop_density_1km"].to_numpy(float))), _z(np.log(ex["fuel_usd_gal"].to_numpy(float)))], 1)
        self.x_extra_names = ["log_pop_density_1km", "log_fuel_price"]
        if CONTROLS_PLUS or SPEED_CAR or SPEED_VAR == "commute":
            e2 = pd.read_parquet(EXTRA2).set_index("GEOID").reindex(pd.Index(self.geoid, name="GEOID"))
            if CONTROLS_PLUS:
                f = lambda c, log=False: _z(np.log1p(e2[c].to_numpy(float)) if log else e2[c].to_numpy(float))  # noqa: E731
                named = [("log_pop_density_500m", f("pop_density_500m", True)), ("log_pop_density_3km", f("pop_density_3km", True)),
                         ("black_share", f("black_share")), ("hispanic_share", f("hispanic_share")), ("asian_share", f("asian_share")),
                         *([("foreign_born_share", f("foreign_born_share"))] if np.nanmax(e2["foreign_born_share"].to_numpy(float)) > 0 else []),   # 0 before the 2026-11-15 census refresh
                         ("household_size", f("household_size")), ("children_hh_share", f("children_hh_share")), ("living_alone_share", f("living_alone_share")),
                         ("multifamily10_share", f("multifamily10_share")), ("detached_share", f("detached_share")), ("built_pre1950_share", f("built_pre1950_share")),
                         ("log_car_speed", _z(np.log(np.clip(e2["car_speed_kmh"].to_numpy(float), 5, 120))))]
                if "car_time_min" in CTRL_ADD:   # the control study (2026-11-16): car commute time of the tract
                    named.append(("log_car_time", _z(np.log(np.clip(e2["car_time_min"].to_numpy(float), 1, 180)))))
                self.x_extra = np.concatenate([self.x_extra, np.stack([v for _, v in named], 1)], 1)
                self.x_extra_names += [n for n, _ in named]
            cs = np.clip(e2["car_speed_kmh"].to_numpy(float), 5, 120)
            cs = np.where(np.isfinite(cs), cs, np.nanmedian(cs))
            self.car_speed_bg = torch.tensor(np.append(cs, np.median(cs)).astype(np.float32), device=DEV)   # + the dummy bucket of cell_orig
            ct = np.log(np.clip(e2["car_time_min"].to_numpy(float), 1, 180)); ct = np.where(np.isfinite(ct), ct, np.nanmedian(ct))
            ct = (ct - ct.mean()) / ct.std()
            self.car_time_z_bg = torch.tensor(np.append(ct, 0.0).astype(np.float32), device=DEV)   # z(log car commute time), dummy bucket 0
        x3 = [c for c in CTRL_ADD if c not in TF_FEATURES and c != "car_time_min"]
        if x3:   # the control study: more block-group controls (fit/build_extra_controls3.py), already transformed; + a missing flag where needed
            e3 = pd.read_parquet(EXTRA3).set_index("GEOID").reindex(pd.Index(self.geoid, name="GEOID"))
            for c in x3:
                v = e3[c].to_numpy(float); miss = ~np.isfinite(v)
                self.x_extra = np.concatenate([self.x_extra, _z(v)[:, None]], 1); self.x_extra_names.append(c)
                if miss.mean() > 0.002:
                    self.x_extra = np.concatenate([self.x_extra, (miss.astype(float) - miss.mean())[:, None]], 1); self.x_extra_names.append(c + " (missing)")
        self.metro_idx = torch.tensor(pd.factorize(self.group_names[self.group])[0], device=DEV)
        self.n_metro = int(self.metro_idx.max().item()) + 1
        print(f"data4: {int(ok.sum()):,} block groups with car + transit commuters, {int(okc.sum()):,} with households; "
              f"{self.n_metro:,} metros / counties", flush=True)

    def control_matrix(self) -> torch.Tensor:
        """CONTROLS (+ FIT4_CTRL_ADD of torch_fit's features: zero_car_share, cars_per_household, commute_km, commute_time, renter_share)
        + the extra columns, minus FIT4_CTRL_DROP (names as in self.control_names)."""
        base = [c for c in CONTROLS if c not in CTRL_DROP] + [c for c in CTRL_ADD if c in TF_FEATURES]
        if base:
            X = self.controls(base); names = list(self.last_control_names)
        else:   # every base control dropped (the study's no-controls run)
            X = torch.zeros((self.nb, 0), dtype=torch.float32, device=DEV); names = []
        keep = [i for i, n in enumerate(self.x_extra_names) if n.replace(" (missing)", "") not in CTRL_DROP]
        self.control_names = names + [self.x_extra_names[i] for i in keep]
        print(f"controls ({len(self.control_names)}): {', '.join(self.control_names)}", flush=True)
        M = torch.cat([X, torch.tensor(self.x_extra[:, keep], dtype=torch.float32, device=DEV)], 1)
        if CTRL_STUDY:   # each control clipped to its 0.1-99.9 % range and standardised again (one household size of 326 sd, shares above 1)
            for j, n in enumerate(self.control_names):
                if not n.endswith("(missing)"):
                    lo, hi = torch.quantile(M[::7, j].float(), torch.tensor([0.001, 0.999], device=M.device))
                    c = M[:, j].clamp(lo, hi); M[:, j] = (c - c.mean()) / c.std().clamp(min=1e-9)
        if CTRL_METRO_MEANS:   # the metro / county mean of each control (missing flags left out), standardised: the context a block group sits in
            g = torch.as_tensor(np.asarray(self.group), device=M.device).long(); ng = len(self.group_names); w = self.pop[:, None]
            cols = [j for j, n in enumerate(self.control_names) if not n.endswith("(missing)")]
            S = torch.zeros((ng, len(cols)), device=M.device).index_add(0, g, w * M[:, cols]) / torch.zeros((ng, 1), device=M.device).index_add(0, g, w).clamp(min=1.0)
            Mm = S[g]; Mm = (Mm - Mm.mean(0)) / Mm.std(0).clamp(min=1e-9)
            M = torch.cat([M, Mm], 1); self.control_names = self.control_names + [self.control_names[j] + " (metro mean)" for j in cols]
            print(f"+ metro means of {len(cols)} controls", flush=True)
        return M


def _ll(x):   # log-logistic in log space: log(1 / (1 + exp(x)))
    return -Fn.softplus(x)


import os as _os
CONVERGE = _os.environ.get('FIT4_CONVERGE', '1') == '1'   # default on: 3000 Adam steps + early stop left the starts spread along a ridge (stability rounds 1-5)
NO_SPEED = _os.environ.get('FIT4_NO_SPEED') == '1'
SHARED_WALK = _os.environ.get('FIT4_SHARED_WALK') == '1'   # one walking curve for every mode
SPEED_REL = _os.environ.get('FIT4_SPEED_REL') == '1'       # speed relative to the mode's median item speed
FIX_V50 = float(_os.environ.get('FIT4_FIX_V50', '0') or 0)
RETENTION = _os.environ.get('FIT4_RETENTION') == '1'   # walk curve per mode = value at 400 m (the mode weight) x own decay slope; one shared half-distance
H3_EDGE_M = {9: 200.8, 10: 75.9, 11: 28.7, 12: 10.8}   # h3.average_hexagon_edge_length(res, "m")
CELL_RES = 10   # the fit's cells
WALK_FLAT_M = float(_os.environ.get('FIT4_WALK_FLAT') or 4 * H3_EDGE_M[CELL_RES])
# walk curves flat below 2 cell diameters (user): 4 x edge = 303.5 m at res 10 -- finer detail needs a finer H3 resolution
W_KNOTS = ([WALK_FLAT_M * (20000.0 / WALK_FLAT_M) ** (i / 9) for i in range(10)] if ONECURVE   # floor .. 20 km, log-spaced
           else [WALK_FLAT_M * (2000.0 / WALK_FLAT_M) ** (i / 6) for i in range(7)])   # floor .. 2 km, log-spaced
D_REF = 400.0   # RETENTION: every mode's walk curve is 1 at 400 m, so the mode weight is the value there (rail = 1)
# FIT4_DECAY: the walk curve is 1 at the floor (WALK_FLAT_M) and only decays beyond it, so mode / headway / speed set the value at the stop and the
# curve only how fast it falls off; its slope k = exp(mode term [mode, both] + c_h log(h / 10 min) + c_v log(v / 20 km/h) [service, both])
DECAY = _os.environ.get('FIT4_DECAY', '')   # '' (shared slope) | mode | service | both
NO_MODE = _os.environ.get('FIT4_NO_MODE') == '1'
# Shape tests (2026-11-14, user: "shouldn't you try multiple shapes?"): one component at a time, the rest unchanged
SHAPE_H = _os.environ.get('FIT4_SHAPE_H', 'loglogistic')   # loglogistic | exp: exp(-h / tau) | wait: 1 / (1 + wait / w0), wait = cap (1 - exp(-h / 2cap)) | mono
SHAPE_W = _os.environ.get('FIT4_SHAPE_W', 'loglogistic')   # loglogistic (with FIT4_DECAY) | exp: exp(-k (d - floor) / 1 km) | mono  (FIT4_DECAY modes only)
SHAPE_S = _os.environ.get('FIT4_SHAPE_S', 'loglogistic')   # loglogistic | power: (v / 20 km/h)^b | mono   (none: FIT4_NO_SPEED=1)
AGG = _os.environ.get('FIT4_AGG', 'max')                     # per cell and mode: max (best item) | sum | ptal (best + 0.5 x each other = (sum + max) / 2)
# "mono": a free monotone curve -- piecewise linear in log(x), non-negative log-slopes between fixed knots (value 1 at the first knot for
# headway / walk, at the top knot for speed), the last slope carried on beyond the knots
H_KNOTS = [2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0]
S_KNOTS = [5.0, 8.0, 12.0, 18.0, 27.0, 40.0, 60.0, 100.0]


def _mono_down(lx, t, sl):
    """Decreasing log curve: -sum_k sl_k * clamp(lx - t_k, 0, t_{k+1} - t_k) (last segment open); 0 at and below t_0. `sl` [K] or per item [K, n]."""
    out = torch.zeros_like(lx)
    for k in range(len(t) - 1):
        hi = (t[k + 1] - t[k]) if k < len(t) - 2 else float("inf")
        out = out - sl[k] * (lx - t[k]).clamp(min=0.0, max=hi)
    return out


def _mono_up(lx, t, sl):
    """Increasing log curve (speed): 0 at and above the top knot, falling with slopes sl_k below it (first segment open)."""
    out = torch.zeros_like(lx)
    n = len(t) - 1
    for k in range(n):
        lo_len = (t[k + 1] - t[k]) if k > 0 else float("inf")
        out = out - sl[k] * (t[k + 1] - lx).clamp(min=0.0, max=lo_len)
    return out   # no mode factor: every mode worth the same at the stop at equal headway and speed
REPARAM = _os.environ.get('FIT4_REPARAM') == '1'   # curves by their value at a reference point (+ slope) instead of half-point (+ slope)
FIT_STEPS = int(_os.environ.get('FIT4_STEPS', '12000'))   # 12000: every start reaches the same optimum (round 4)
REF = {"h": math.log(10.0), "v": math.log(20.0), "w": math.log(400.0), "f": math.log(5000.0)}   # 10 min, 20 km/h, 400 m, 5 km
LOAD_MAX = math.log(4.0)   # bus / rail / tram loadings of the block-group propensity term: at most 4x apart (unbounded ran to 17.8 in the smoke test)

class Model4(torch.nn.Module):
    def __init__(self, d: Data4, level="raw", drive=False, ncontrol=0):
        super().__init__()
        P = lambda *v: torch.nn.Parameter(torch.tensor(v, dtype=torch.float32))  # noqa: E731
        self.level, self.drive = level, drive
        self.lh50, self.lkh = P(math.log(25.0)), P(math.log(1.5))
        self.lv50, self.lkv = P(math.log(12.0)), P(math.log(2.0))
        self.ld50, self.lkw = P(math.log(400.0), math.log(400.0), math.log(700.0)), P(math.log(2.0))   # bus, tram, rail (MODES order)
        if DECAY in ("service", "both"):
            self.c_h, self.c_v = P(0.0), P(0.0)
            if SPEED_VAR == "nodecay":
                self.c_v.requires_grad_(False)
        if RETENTION or DECAY in ("mode", "both"):   # one decay slope per mode; the half-distance is shared (ld50[0]) -- it only shapes the first ~200 m, which the data hardly see
            self.lkw = P(math.log(1.0), math.log(1.0), math.log(1.0))
        sp1 = math.log(math.expm1(1.0))   # softplus^-1(1): log-slope 1 per segment to start
        if SHAPE_H == "exp":
            self.ltau = P(math.log(10.0))
        elif SHAPE_H == "wait":
            self.lcap, self.lw0 = P(math.log(15.0)), P(math.log(5.0))
        elif SHAPE_H == "mono":
            self.th_h = torch.nn.Parameter(torch.full((len(H_KNOTS) - 1,), sp1))
        if SPEED_VAR in ("power", "permode", "commute", "far"):
            self.b_v = P(*{"power": [0.0], "permode": [0.0, 0.0, 0.0], "commute": [0.0, 0.0], "far": [0.0]}[SPEED_VAR])
        elif SHAPE_S == "power":
            self.b_v = P(0.3)
        elif SHAPE_S == "mono":
            self.th_s = torch.nn.Parameter(torch.full((len(S_KNOTS) - 1,), math.log(math.expm1(0.3))))
        if SHAPE_W == "mono":
            self.th_w = torch.nn.Parameter(torch.full((3, len(W_KNOTS) - 1), sp1))
            self.lkw.data.zero_(); self.lkw.requires_grad_(False)   # the mono curve carries the per-mode decay; the service terms still scale it
        self.lmode = P(-0.3, -0.2)   # bus, tram (rail = 1)
        if NO_MODE:
            self.lmode.data.zero_(); self.lmode.requires_grad_(False)
        Tn = len(TASKS)
        self.alpha = torch.nn.Parameter(torch.full((Tn,), -3.0)); self.leps = torch.nn.Parameter(torch.full((Tn,), math.log(0.05)))
        self.lkappa = torch.nn.Parameter(torch.full((Tn,), math.log(5.0)))
        if METRO_CTX:
            self.b_ctx = torch.nn.Parameter(torch.zeros(Tn))
            d.grp_t = torch.as_tensor(np.asarray(d.group), device=DEV).long(); d.n_grp = len(d.group_names)
        if drive:
            self.lf50, self.lkf = P(math.log(6000.0)), P(math.log(2.0))
            self.lthf = torch.nn.Parameter(torch.full((Tn,), math.log(0.5))); self.lepsf = torch.nn.Parameter(torch.full((Tn,), math.log(0.05)))
        if level in ("controls", "sem"):
            self.delta = torch.nn.Parameter(torch.zeros(Tn, ncontrol))
            self.ltheta = torch.nn.Parameter(torch.full((Tn,), math.log(0.5))); self.leps_d = torch.nn.Parameter(torch.full((Tn,), math.log(0.05)))
            self.metro_u = torch.nn.Parameter(torch.zeros(d.n_metro))
            if NO_METRO_RE:
                self.metro_u.requires_grad_(False)
        if level == "sem":
            self.gamma = torch.nn.Parameter(torch.full((Tn,), 0.3))          # car-free share -> transit
            self.c_alpha = P(-2.0); self.c_beta = P(0.5); self.c_leps = P(math.log(0.05)); self.c_lkappa = P(math.log(5.0))
            self.c_delta = torch.nn.Parameter(torch.zeros(ncontrol)); self.c_metro = torch.nn.Parameter(torch.zeros(d.n_metro))
            self.bg_u = torch.nn.Parameter(torch.zeros(d.nb)); self.lam_u = torch.nn.Parameter(torch.zeros(Tn - 1))   # loadings of bus/rail/tram (total = 1)
        for name in FREEZE:
            getattr(self, name).requires_grad_(False)

    # ---- curves -------------------------------------------------------------------------------------------------------------------------
    def logH(self, h):
        if H_FLOOR:   # 1 at the floor, flat below it
            hf = h.clamp(min=H_FLOOR)
            return self._logH_raw(hf) - self._logH_raw(torch.full_like(hf[:1], H_FLOOR))
        return self._logH_raw(h)

    def _logH_raw(self, h):
        h = h.clamp(min=0.5)
        if SHAPE_H == "exp":
            return -h / torch.exp(self.ltau)
        if SHAPE_H == "wait":
            cap = torch.exp(self.lcap)
            return -torch.log1p(cap * (1 - torch.exp(-h / (2 * cap))) / torch.exp(self.lw0))
        if SHAPE_H == "mono":
            return _mono_down(torch.log(h), [math.log(x) for x in H_KNOTS], Fn.softplus(self.th_h))
        return _ll(torch.exp(self.lkh) * (torch.log(h) - self.lh50))
    def logS(self, v):
        if NO_SPEED:
            return torch.zeros_like(v)
        if SHAPE_S == "power":
            return self.b_v * torch.log(v.clamp(min=0.5) / 20.0)
        if SHAPE_S == "mono":
            return _mono_up(torch.log(v.clamp(min=0.5)), [math.log(x) for x in S_KNOTS], Fn.softplus(self.th_s))
        lv50 = torch.tensor(math.log(FIX_V50), device=v.device) if FIX_V50 else self.lv50
        return _ll(torch.exp(self.lkv) * (lv50 - torch.log(v.clamp(min=0.5))))

    def logS_items(self, d, ix):
        if SPEED_VAR in ("power", "permode", "commute", "far"):
            lv = torch.log(d.it_v[ix].clamp(min=0.5) / 20.0)
            if SPEED_VAR == "power":
                return self.b_v[0] * lv
            if SPEED_VAR == "permode":
                md = d.it_mode[ix]
                return torch.where(md == 0, self.b_v[0], torch.where(md == 1, self.b_v[1], self.b_v[2])) * lv
            if SPEED_VAR == "commute":
                return (self.b_v[0] + self.b_v[1] * d.car_time_z_bg[d.cell_orig[d.it_cell[ix].long()]]) * lv
            return self.b_v[0] * lv * (d.it_d[ix] > 2000.0).float()
        if SPEED_REL:   # speed over the mode's median item speed (1 = typical for the mode); the half-point is then a ratio
            if not hasattr(d, "mode_med_v"):
                d.mode_med_v = torch.tensor(np.asarray(d.v_median, dtype=np.float32), device=d.it_v.device)
            return self.logS(d.it_v[ix] / d.mode_med_v[d.it_mode[ix].long()])
        if SPEED_CAR:   # transit speed relative to the local car commute speed, on the km/h scale (ratio 1 = 20)
            return self.logS(20.0 * d.it_v[ix] / d.car_speed_bg[d.cell_orig[d.it_cell[ix].long()]])
        return self.logS(d.it_v[ix])
    def logW(self, dist, mode, h=None, v=None):
        # per-mode half-distance chosen with where(): indexing the 3-value parameter by ~1e8 items made the backward ~200x slower (atomic adds on 3 slots)
        l = self.ld50
        d50 = l[0] if SHARED_WALK else torch.where(mode == 0, l[0], torch.where(mode == 1, l[1], l[2]))
        if DECAY:
            lk = self.lkw if self.lkw.numel() == 1 else torch.where(mode == 0, self.lkw[0], torch.where(mode == 1, self.lkw[1], self.lkw[2]))
            if DECAY in ("service", "both"):
                lk = lk + self.c_h * torch.log(h.clamp(min=0.5) / 10.0) + self.c_v * torch.log(v.clamp(min=0.5) / 20.0)
            k = torch.exp(lk); d50 = l[0]
            if SHAPE_W == "exp":
                return -k * (dist.clamp(min=WALK_FLAT_M) - WALK_FLAT_M) / 1000.0
            if SHAPE_W == "mono":   # per-mode free monotone curve, scaled by the service terms (k = 1 at 10 min / 20 km/h)
                th = Fn.softplus(self.th_w)
                sl = [torch.where(mode == 0, th[0, j], torch.where(mode == 1, th[1, j], th[2, j])) for j in range(th.shape[1])]
                return k * _mono_down(torch.log(dist.clamp(min=WALK_FLAT_M)), [math.log(x) for x in W_KNOTS], sl)
            return _ll(k * (torch.log(dist.clamp(min=WALK_FLAT_M)) - d50)) - _ll(k * (math.log(WALK_FLAT_M) - d50))   # = 1 at the floor, decays beyond
        if RETENTION:
            lk = torch.where(mode == 0, self.lkw[0], torch.where(mode == 1, self.lkw[1], self.lkw[2])); k = torch.exp(lk); d50 = l[0]
            return _ll(k * (torch.log(dist.clamp(min=WALK_FLAT_M)) - d50)) - _ll(k * (math.log(D_REF) - d50))   # = 1 at D_REF, flat below WALK_FLAT_M
        return _ll(torch.exp(self.lkw) * (torch.log(dist.clamp(min=WALK_FLAT_M)) - d50))   # flat below WALK_FLAT_M
    def logF(self, dist): return _ll(torch.exp(self.lkf) * (torch.log(dist.clamp(min=1.0)) - self.lf50))
    def lm(self): return torch.cat([self.lmode, torch.zeros(1, device=self.lmode.device)])

    def _cell_max(self, la, d, sl):   # reference implementation (slow backward: scatter_reduce amax over ~10^8 items); kept for checks
        idx = d.it_cell[sl] * 3 + d.it_mode[sl]
        out = torch.zeros(d.ncell * 3, device=la.device).scatter_reduce(0, idx, torch.exp(la), reduce="amax", include_self=True)
        return out.view(d.ncell, 3)

    def _cell_best(self, fn, d, lo, hi):
        """Per cell and mode exp(max item value): the winning item is found without gradients, then only the winners are recomputed with
        gradients -- the same value and (sub)gradient as `_cell_max`, ~20x faster backward."""
        G = d.ncell * 3
        with torch.no_grad():
            ix = torch.arange(lo, hi, device=d.it_h.device)
            la = fn(ix)
            g = d.it_cell[lo:hi].long() * 3 + d.it_mode[lo:hi].long()
            mx = torch.full((G,), -float("inf"), device=la.device).scatter_reduce(0, g, la, reduce="amax", include_self=True)
            hit = la >= mx[g]
            win = torch.full((G,), -1, dtype=torch.long, device=la.device).scatter_reduce(0, g[hit], ix[hit], reduce="amax", include_self=True)
            okg = torch.nonzero(win >= 0).squeeze(1)
            del ix, la, g, mx, hit
        vals = torch.exp(fn(win[okg]))
        return torch.zeros(G, device=vals.device).scatter(0, okg, vals).view(d.ncell, 3)

    def access(self, d):
        """(walking [ncell, 3], driving [ncell, 3] or None): per cell and mode the best item value without the mode value."""
        nw = d.n_walk
        fw = lambda ix: self.logH(d.it_h[ix]) + self.logS_items(d, ix) + self.logW(d.it_d[ix], d.it_mode[ix], d.it_h[ix], d.it_v[ix])  # noqa: E731
        walk = self._cell_best(fw, d, 0, d.it_h.numel() if ONECURVE else nw)   # ONECURVE: every item, its distance through the one curve
        if AGG in ("sum", "ptal"):   # every item counts (PTAL: the best fully, the others half)
            ix = torch.arange(0, nw, device=d.it_h.device)
            g = d.it_cell[:nw].long() * 3 + d.it_mode[:nw].long()
            tot = torch.zeros(d.ncell * 3, device=d.it_h.device).index_add(0, g, torch.exp(fw(ix))).view(d.ncell, 3)
            walk = tot if AGG == "sum" else 0.5 * (tot + walk)
        farm = None
        if self.drive:
            farm = self._cell_best(lambda ix: self.logH(d.it_h[ix]) + self.logS_items(d, ix) + self.logF(d.it_d[ix]), d, nw, d.it_h.numel())
        return walk, farm

    def access_reference(self, d):
        nw = d.n_walk
        ws = slice(0, nw)
        walk = self._cell_max(self.logH(d.it_h[ws]) + self.logS(d.it_v[ws]) + self.logW(d.it_d[ws], d.it_mode[ws]), d, ws)
        farm = None
        if self.drive:
            fs = slice(nw, None)
            farm = self._cell_max(self.logH(d.it_h[fs]) + self.logS(d.it_v[fs]) + self.logF(d.it_d[fs]), d, fs)
        return walk, farm

    def _cols(self, c):
        lm = self.lm()
        return torch.stack([(c * torch.exp(lm)[None, :]).max(dim=1).values, c[:, 0], c[:, 2], c[:, 1]], 1)   # TASKS order (MODES = bus, tram, rail)

    def _area(self, C, d):
        return torch.zeros((d.nb + 1, C.shape[1]), device=C.device).index_add(0, d.cell_orig, d.cell_pop[:, None] * C)[: d.nb] / d.pop.clamp(min=1.0)[:, None]

    def etas(self, d, X=None, use_re=True):
        walk, farm = self.access(d)
        C = self._cols(walk); A = self._area(C, d)
        e = self.alpha[None, :] + torch.log1p(A / torch.exp(self.leps)[None, :])
        if METRO_CTX:   # metro population-weighted mean of the access term, centred: one coefficient per task
            a = torch.log1p(A / torch.exp(self.leps)[None, :]); g = d.grp_t; w = d.pop[:, None]
            S = torch.zeros((d.n_grp, a.shape[1]), device=a.device).index_add(0, g, w * a) / torch.zeros((d.n_grp, 1), device=a.device).index_add(0, g, w).clamp(min=1.0)
            c = S[g]; e = e + self.b_ctx[None, :] * (c - (w * c).sum(0) / w.sum())
        if self.drive:
            Af = self._area(self._cols(farm), d)
            e = e + torch.exp(self.lthf)[None, :] * torch.log1p(Af / torch.exp(self.lepsf)[None, :])
        out = {"A": A[:, 0]}
        if self.level in ("controls", "sem"):
            Aw = torch.zeros((d.nbg_all, 4), device=C.device).index_add(0, d.cell_bg_all, C * d.cell_jobs[:, None]) / d.bg_jobs.clamp(min=1.0)[:, None]
            D = torch.zeros((d.nb, 4), device=C.device).index_add(0, d.fl_o, d.fl_wt[:, None] * t3._rows(Aw, d.fl_w))
            if not NO_WORK:
                e = e + torch.exp(self.ltheta)[None, :] * torch.log1p(D / torch.exp(self.leps_d)[None, :]) * d.has_flows[:, None]
            e = e + X @ self.delta.T
            if use_re:
                e = e + self.metro_u[d.metro_idx][:, None]
        if self.level == "sem":
            e = e + self.gamma[None, :] * d.carfree_logit[:, None]
            if use_re:
                lam = torch.cat([torch.ones(1, device=e.device), torch.exp(LOAD_MAX * torch.tanh(self.lam_u))])   # loadings bounded to [1/4, 4]
                e = e + self.bg_u[:, None] * lam[None, :]
            ec = self.c_alpha + self.c_beta * torch.log1p(A[:, 0] / torch.exp(self.c_leps)) + X @ self.c_delta
            if use_re:
                ec = ec + self.c_metro[d.metro_idx]
            out["car"] = ec
        out.update({task: e[:, k] for k, task in enumerate(TASKS)})
        return out

    def penalty(self, d):
        p = torch.zeros((), device=self.alpha.device)
        if self.level in ("controls", "sem"):
            p = p + 1e-4 * (self.metro_u ** 2).mean() + 1e-6 * (self.delta ** 2).sum()
        if self.level == "sem":
            p = p + 1e-4 * (self.c_metro ** 2).mean() + 1e-3 * (self.bg_u ** 2).mean() + 1e-4 * (self.lam_u ** 2).sum()
        return p

    def params(self) -> dict:
        g = lambda t: [round(float(x), 5) for x in torch.as_tensor(t).detach().cpu().reshape(-1)]  # noqa: E731
        p = {"h50_min": math.exp(self.lh50.item()), "kh": math.exp(self.lkh.item()), "v50_kmh": math.exp(self.lv50.item()), "kv": math.exp(self.lkv.item()),
             "d50_m": dict(zip(["bus", "tram", "rail"], [math.exp(x) for x in g(self.ld50)])), "kw": (math.exp(self.lkw.item()) if self.lkw.numel() == 1 else dict(zip(["bus", "tram", "rail"], [math.exp(x) for x in g(self.lkw)]))),
             "mode": dict(zip(["bus", "tram", "rail"], [math.exp(x) for x in g(self.lm())])), "eps": [math.exp(x) for x in g(self.leps)], "alpha": g(self.alpha)}
        if SPEED_VAR in ("power", "permode", "commute", "far"):
            p["speed_b"] = g(self.b_v)   # log S = b log(v / 20 km/h) (permode: bus, tram, rail; commute: b0, b1 per sd of log car time)
        if hasattr(self, "c_h"):
            p.update({"decay_c_h": float(self.c_h.item()), "decay_c_v": float(self.c_v.item())})
        with torch.no_grad():   # the fitted curves at reference points (any shape): headway / speed / walk (at 10 min, 20 km/h)
            dv = self.alpha.device
            T = lambda xs: torch.tensor(xs, dtype=torch.float32, device=dv)  # noqa: E731
            hs, vs, ds = [2, 5, 10, 15, 20, 30, 60, 120], [5, 10, 15, 20, 30, 50, 80], [300, 400, 600, 800, 1200, 1600, 2000]
            p["curve_H"] = dict(zip(map(str, hs), torch.exp(self.logH(T(hs))).tolist()))
            p["curve_S"] = dict(zip(map(str, vs), torch.exp(self.logS(T(vs))).tolist()))
            p["curve_W"] = {mo: dict(zip(map(str, ds), torch.exp(self.logW(T(ds), torch.full((len(ds),), m, device=dv), T([10.0] * len(ds)), T([20.0] * len(ds)))).tolist()))
                            for m, mo in enumerate(["bus", "tram", "rail"])}
        if DECAY:   # share of the value at the floor kept at each distance, at headway 10 min and 20 km/h (DECAY service/both: k varies with them)
            d50 = max(math.exp(g(self.ld50)[0]), 1e-30); fl = WALK_FLAT_M
            kk = p["kw"] if isinstance(p["kw"], dict) else {mo: p["kw"] for mo in ("bus", "tram", "rail")}
            sp = lambda x: max(x, 0.0) + math.log1p(math.exp(-abs(x)))  # noqa: E731   (stable softplus: d50 drifts to ~0 m = a pure power law)
            Wf = lambda dd, k: math.exp(sp(k * math.log(fl / d50)) - sp(k * math.log(max(dd, fl) / d50)))  # noqa: E731
            p["retention_at_10min_20kmh"] = {mo: {f"{dd}m": Wf(dd, k) for dd in (400, 800, 1600)} for mo, k in kk.items()}
            p["value_at_10min_20kmh"] = {mo: {f"{dd}m": p["mode"][mo] * Wf(dd, k) for dd in (round(fl), 400, 800, 1600)} for mo, k in kk.items()}
        if RETENTION:   # value retention per mode at equal headway and speed: walk value relative to 400 m, and mode value (weight x walk) at each distance
            sp = lambda x: max(x, 0.0) + math.log1p(math.exp(-abs(x)))  # noqa: E731
            l50 = g(self.ld50)[0]
            W = lambda d, k: math.exp(sp(k * (math.log(D_REF) - l50)) - sp(k * (math.log(max(d, WALK_FLAT_M)) - l50)))  # noqa: E731
            p["retention"] = {mo: {f"{dd}m": W(dd, p["kw"][mo]) for dd in (200, 800, 1600)} | {"per_doubling": 2 ** -p["kw"][mo]} for mo in p["kw"]}
            p["value"] = {mo: {f"{dd}m": p["mode"][mo] * W(dd, p["kw"][mo]) for dd in (200, 400, 800, 1600)} for mo in p["kw"]}
        if self.drive:
            p.update({"f50_m": math.exp(self.lf50.item()), "kf": math.exp(self.lkf.item()), "thf": [math.exp(x) for x in g(self.lthf)], "epsf": [math.exp(x) for x in g(self.lepsf)]})
        if self.level in ("controls", "sem"):
            p.update({"theta_workplace": [math.exp(x) for x in g(self.ltheta)], "delta": g(self.delta)})
        if self.level == "sem":
            p.update({"gamma_carfree": g(self.gamma), "car_beta": float(self.c_beta.item()), "car_eps": math.exp(self.c_leps.item()), "car_delta": g(self.c_delta),
                      "bg_u_sd": float(self.bg_u.std().item()), "loadings": [1.0] + [math.exp(LOAD_MAX * math.tanh(x)) for x in g(self.lam_u)]})
        return p


def _load_curves(m, c):
    with torch.no_grad():
        for k, v in c.items():
            getattr(m, k).copy_(torch.as_tensor(v, dtype=torch.float32))
Model4.load_curves = _load_curves


class ModelR(Model4):
    """Same model; each curve stored as (r, log slope) with r = k * (log x_ref - log x50) (speed: k * (log v50 - log v_ref)), so the
    value at the reference point -- which the data fix well -- is one parameter instead of a ridge between the half-point and the slope."""
    _CURVES = (("lh50", "lkh", "h", 1.0), ("lv50", "lkv", "v", -1.0), ("ld50", "lkw", "w", 1.0), ("lf50", "lkf", "f", 1.0))

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        for l50, lk, ref, sg in self._CURVES:
            if l50 in self._parameters:
                r = sg * torch.exp(self._parameters[lk].detach()) * (REF[ref] - self._parameters[l50].detach())
                del self._parameters[l50]; self.register_parameter("r_" + ref, torch.nn.Parameter(r.clone()))

    def _l50(self, ref, lk, sg): return REF[ref] - sg * getattr(self, "r_" + ref) / torch.exp(getattr(self, lk))
    lh50 = property(lambda m: m._l50("h", "lkh", 1.0))
    lv50 = property(lambda m: m._l50("v", "lkv", -1.0))
    ld50 = property(lambda m: m._l50("w", "lkw", 1.0))
    lf50 = property(lambda m: m._l50("f", "lkf", 1.0))

    def load_curves(self, c):
        with torch.no_grad():
            for l50, lk, ref, sg in self._CURVES:
                if lk in c:
                    k = torch.as_tensor(c[lk], dtype=torch.float32); getattr(self, lk).copy_(k)
                    getattr(self, "r_" + ref).copy_(sg * torch.exp(k) * (REF[ref] - torch.as_tensor(c[l50], dtype=torch.float32)))


def task_ll(m, d, X, use_re=True):
    E = m.etas(d, X, use_re)
    out = {}
    for k, task in enumerate(TASKS):
        mu = torch.sigmoid(E[task].double()).clamp(1e-7, 1 - 1e-7)
        y, mask = (d.y, d.ok_total) if task == "total" else (d.y_mode[task], d.ok_mode[task])
        n = d.n.clamp(min=1.0)
        out[task] = (bb_loglik(y.double(), n.double(), mu.double(), torch.exp(m.lkappa[k]).double()) / n.double(), mask)   # float64: lgamma differences of ~1e4 terms
    if "car" in E:
        mu = torch.sigmoid(E["car"].double()).clamp(1e-7, 1 - 1e-7); n = d.c_n.clamp(min=1.0)
        out["car"] = (bb_loglik(d.c_y.double(), n.double(), mu.double(), torch.exp(m.c_lkappa).double()) / n.double(), d.c_ok)
    return out


def null_losses(d, w, with_car=False):
    """Per task the best constant-share loss (the normaliser of multi_loss): float64 and polished to convergence, so it is the same number in
    every process (float32 + 600 Adam steps left ~1e-4 differences between runs -- as large as the model differences being compared)."""
    out = {}
    names = TASKS + (["car"] if with_car else [])
    for task in names:
        a = torch.tensor([-3.0], device=DEV, dtype=torch.float64, requires_grad=True); lk = torch.tensor([math.log(5.0)], device=DEV, dtype=torch.float64, requires_grad=True)
        if task == "car":
            y, n, mask = d.c_y, d.c_n.clamp(min=1.0), d.c_ok
        else:
            y, n, mask = (d.y, d.n.clamp(min=1.0), d.ok_total) if task == "total" else (d.y_mode[task], d.n.clamp(min=1.0), d.ok_mode[task])
        y, n = y.double(), n.double()
        ww = (w * mask).double(); ww = ww / ww.sum()
        f = lambda: -(ww * bb_loglik(y, n, torch.sigmoid(a).clamp(1e-7, 1 - 1e-7).expand_as(y), torch.exp(lk)) / n).sum()  # noqa: E731
        opt = torch.optim.Adam([a, lk], lr=0.05)
        for _ in range(600):
            opt.zero_grad(); L = f(); L.backward(); opt.step()
        o2 = torch.optim.LBFGS([a, lk], lr=0.5, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
        def cl():
            o2.zero_grad(); t = f(); t.backward(); return t
        o2.step(cl)
        with torch.no_grad():
            L = f()
        out[task] = (L.item(), a.detach().float(), lk.detach().float())
    print("nulls " + " ".join(f"{k} {v[0]:.10f}" for k, v in out.items()), flush=True)
    return out


def multi_loss(m, d, w, nulls, X, use_re=True):
    tl = task_ll(m, d, X, use_re); tot = 0.0; parts = {}
    for task, (ll, mask) in tl.items():
        ww = w * mask; ww = ww / ww.sum()
        L = -(ww.double() * ll.double()).sum(); parts[task] = L   # float64 sum: the converge test (1e-7) is below float32 resolution of a ~3.5 loss
        tot = tot + L / nulls[task][0]
    return tot, parts


def fit(m, d, w, nulls, X, steps=3000, lr=0.03, quiet=True):
    m.to(DEV)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    prev, flat = None, 0
    for s in range(steps):
        if s == steps // 3:
            for g in opt.param_groups:
                g["lr"] = lr / 3
        opt.zero_grad(); L, _ = multi_loss(m, d, w, nulls, X); (L + m.penalty(d)).backward(); opt.step()
        if not quiet and s % 1000 == 0:
            print(f"  step {s} loss {L.item():.5f}", flush=True)
        if not CONVERGE and s % 100 == 99 and s >= steps // 3:   # early stop (after the learning-rate drop): < 0.002 % improvement per 100 steps, twice in a row
            cur = L.item()
            flat = flat + 1 if prev is not None and (prev - cur) < 2e-5 * abs(cur) else 0
            prev = cur
            if flat >= 2:
                break
    m.adam_steps = s + 1
    o2 = torch.optim.LBFGS(params, lr=0.5, max_iter=80 if not CONVERGE else 150, history_size=20 if not CONVERGE else 50,
                           line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)

    def cl():
        o2.zero_grad(); t = multi_loss(m, d, w, nulls, X)[0] + m.penalty(d); t.backward(); return t
    def now():   # loss of the current state (LBFGS.step returns the loss *before* its step)
        with torch.no_grad():
            v = float(multi_loss(m, d, w, nulls, X)[0] + m.penalty(d))
        return v if math.isfinite(v) and all(torch.isfinite(p).all() for p in m.parameters()) else float("nan")
    best, snap = now(), {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.lbfgs_rounds = 0; m.nan_restored = False; fails = 0
    while m.lbfgs_rounds < (25 if CONVERGE else 1):   # converge: L-BFGS rounds until the loss improves less than 1e-7 per round
        o2.step(cl); m.lbfgs_rounds += 1; cur = now()
        if not quiet:
            print(f"    lbfgs round {m.lbfgs_rounds}: {cur:.7f}", flush=True)
        if not (cur <= best):   # diverged (nan) or worse: back to the last good state, fresh L-BFGS memory
            m.load_state_dict(snap); m.nan_restored = m.nan_restored or not math.isfinite(cur)
            o2 = torch.optim.LBFGS(params, lr=0.1, max_iter=150, history_size=50, line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
            fails += 1
            if fails >= 2:
                break
            continue
        done = best - cur < 1e-7
        best, snap = cur, {k: v.detach().clone() for k, v in m.state_dict().items()}
        if done:
            break
    with torch.no_grad():
        L, parts = multi_loss(m, d, w, nulls, X)
    return L.item(), {k: v.item() for k, v in parts.items()}


def randomize(m, rng):
    """Very different starting points: every curve parameter drawn wide around its default."""
    c = {"lh50": math.log(rng.uniform(5, 120)), "lkh": math.log(rng.uniform(0.3, 4)), "lv50": math.log(rng.uniform(3, 50)), "lkv": math.log(rng.uniform(0.3, 5)),
         "ld50": np.log(rng.uniform(80, 2000, 3)), "lkw": math.log(rng.uniform(0.5, 6)) if m.lkw.numel() == 1 else np.log(rng.uniform(0.3, 3, m.lkw.numel()))}
    lm_ = torch.tensor(rng.normal(0, 0.7, 2), dtype=torch.float32)
    if not NO_MODE:
        m.lmode.data.copy_(lm_)
    if hasattr(m, "c_h"):
        m.c_h.data.fill_(float(rng.normal(0, 0.3))); m.c_v.data.fill_(float(rng.normal(0, 0.3)))
    with torch.no_grad():   # shape-test parameters (FIT4_SHAPE_*): jittered around their defaults
        for name, sd in (("ltau", 0.6), ("lcap", 0.6), ("lw0", 0.6), ("b_v", 0.2), ("th_h", 0.7), ("th_s", 0.7), ("th_w", 0.7)):
            if hasattr(m, name):
                t = getattr(m, name); t.add_(torch.tensor(rng.normal(0, sd, tuple(t.shape)), dtype=t.dtype))
    if m.drive:
        c.update({"lf50": math.log(rng.uniform(2500, 15000)), "lkf": math.log(rng.uniform(0.5, 5))})
    m.load_curves(c)


def scores(m, d) -> dict:
    """P95 references (population-weighted over res-10 cells) of the direct and total service values, and the score curve helpers."""
    with torch.no_grad():
        walk, farm = m.access(d)
        A = m._cols(walk)[:, 0]
        direct = torch.log1p(A / torch.exp(m.leps[0]))
        if m.drive:
            direct = direct + torch.exp(m.lthf[0]) * torch.log1p(m._cols(farm)[:, 0] / torch.exp(m.lepsf[0]))
        total = direct + (m.gamma[0] * m.c_beta * torch.log1p(A / torch.exp(m.c_leps)) if m.level == "sem" else 0.0)
        w = d.cell_pop
        def p95(x):
            o = torch.argsort(x); cw = torch.cumsum(w[o], 0); return float(x[o][torch.searchsorted(cw, 0.95 * cw[-1]).clamp(max=len(x) - 1)])
        out = {"p95_direct": p95(direct), "p95_total": p95(total) if m.level == "sem" else None}
        for k, v in (("direct", direct), ("total", total if m.level == "sem" else None)):
            if v is None:
                continue
            s = (100 * v / out[f"p95_{k}"]).clamp(max=100)
            out[f"score_{k}_popmean"] = float((s * w).sum() / w.sum())
            out[f"share_cells_zero_{k}"] = float(((v <= 0) * w).sum() / w.sum())
        return out


def cv_skill(d, mk, X, w_fit, base_sd, steps=int(_os0.environ.get("FIT4_CV_STEPS", "900"))):   # FIT4_CV_STEPS: smoke tests only
    """Metro-grouped 5-fold held-out skill per task (1 - held-out loss / no-transit loss), random terms of held-out metros / block groups at 0."""
    folds = tf.cv_folds(d)
    w_eval = t2.weights(d, 0.5)
    ho = {t: np.zeros(d.nb) for t in TASKS}; ho0 = {t: np.zeros(d.nb) for t in TASKS}
    mu_ho = {t: np.zeros(d.nb, np.float32) for t in TASKS}   # held-out predicted share per block group (SAVE_HO)
    fold_params = []   # the curves refitted without each fifth of the metros: stability across metro subsets
    for f in range(5):
        tr = w_fit * torch.tensor((folds != f).astype(np.float32), device=DEV); tr = tr / tr.sum()
        nl = null_losses(d, tr, with_car=mk().level == "sem")
        m = mk()
        if CV_COLD:   # FIT4_CV_COLD=1: each fold fitted from the default start, so the held-out metros cannot leak in through the full-data fit
            fit(m, d, tr, nl, X, steps=FIT_STEPS)
        else:
            m.load_state_dict(base_sd); fit(m, d, tr, nl, X, steps=steps, lr=0.01)
        fold_params.append(m.params())
        with torch.no_grad():
            tl = task_ll(m, d, X, use_re=False)
            if SAVE_HO:
                E = m.etas(d, X, use_re=False)
            for task in TASKS:
                ho[task][folds == f] = tl[task][0].cpu().numpy()[folds == f]
                if SAVE_HO:
                    mu_ho[task][folds == f] = torch.sigmoid(E[task]).cpu().numpy()[folds == f]
                _, a, lk = nl[task]
                y = d.y if task == "total" else d.y_mode[task]
                ho0[task][folds == f] = (bb_loglik(y, d.n.clamp(min=1.0), torch.sigmoid(a).expand_as(y), torch.exp(lk)) / d.n.clamp(min=1.0)).cpu().numpy()[folds == f]
    wn = w_eval.cpu().numpy(); out = {}
    for task in TASKS:
        mask = (d.ok_total if task == "total" else d.ok_mode[task]).cpu().numpy()
        out[task] = float(1 - (wn * mask * ho[task]).sum() / (wn * mask * ho0[task]).sum())
        # per metro / county (the fold units): held-out and no-transit losses, for paired bootstrap tests between runs (control study)
        g = np.asarray(d.group); ng = len(d.group_names)
        out[task + "_unit"] = {"num": np.bincount(g, weights=wn * mask * ho[task], minlength=ng).tolist(),
                               "den": np.bincount(g, weights=wn * mask * ho0[task], minlength=ng).tolist()}
        # (the sums are log-likelihoods: negative, so no max(., tiny) guard)
        out[task + "_folds"] = [float(1 - (wn * mask * ho[task] * (folds == f)).sum() / (wn * mask * ho0[task] * (folds == f)).sum()) for f in range(5)]
    uf = np.full(len(d.group_names), -1); uf[np.asarray(d.group)] = folds; out["unit_fold"] = uf.tolist()   # the fold of each metro / county
    out["fold_params"] = fold_params
    if SAVE_HO:
        out["_ho"] = {"group": np.asarray(d.group), "n": d.n.cpu().numpy(), "w": wn, "y_total": d.y.cpu().numpy(),
                      **{f"y_{t}": d.y_mode[t].cpu().numpy() for t in TASKS if t != "total"}, **{f"mu_{t}": v for t, v in mu_ho.items()},
                      **{f"ok_{t}": (d.ok_total if t == "total" else d.ok_mode[t]).cpu().numpy() for t in TASKS}}
    return out


def run(d, name, level="raw", drive=False, starts=10, steps=None, cv=True, keep=None):
    steps = steps or FIT_STEPS
    starts = int(_os.environ.get('FIT4_RUN_STARTS') or starts)   # held-out comparisons of model forms: fewer starts (each start converges, round 4)
    name = name + _os.environ.get('FIT4_SUFFIX', '')   # stage-2 model forms (FIT4_DECAY / FIT4_NO_MODE) saved side by side
    keep0 = d.keep.clone()
    if keep is not None:
        d.keep = keep0 * torch.tensor(keep.astype(np.float32), device=DEV)
    try:
        X = d.control_matrix() if level in ("controls", "sem") else None
        w = t3.weights3(d, True, WLAM)
        nulls = null_losses(d, w, with_car=level == "sem")
        mk = lambda: Model4(d, level, drive, X.shape[1] if X is not None else 0)  # noqa: E731
        runs = []
        init = _os.environ.get("FIT4_INIT")   # warm start (per-metro fits: from the national fit of the same form)
        for sd in range(starts):
            m = mk()
            if sd:
                randomize(m, np.random.default_rng(500 + sd))
            elif init:
                isd = torch.load(init, map_location="cpu")
                # a warm start must not overwrite what this form fixes (NO_MODE: mode factors = 1) nor carry a speed curve the init never fitted
                # (an init fitted with NO_SPEED has kv ~ 0, where the speed curve's gradient vanishes and it would stay flat)
                skip = {"lmode"} if NO_MODE else set()
                if not NO_SPEED and "lkv" in isd and math.exp(float(isd["lkv"])) < 0.05:
                    skip |= {"lv50", "lkv", "b_v", "th_s"}
                    if SPEED_REL:   # the half-point is then a ratio to the mode's median speed: start at the median
                        m.lv50.data.fill_(0.0)
                m.load_state_dict({k: v for k, v in isd.items() if k not in skip and k in m.state_dict() and m.state_dict()[k].shape == v.shape}, strict=False)
                if skip:
                    print(f"warm start: kept this form's own {sorted(skip & set(m.state_dict()))}", flush=True)
            t0 = time.time(); L, parts = fit(m, d, w, nulls, X, steps=steps)
            runs.append({"seed": sd, "loss": L, "parts": parts, "params": m.params(), "scores": scores(m.to(DEV), d), "seconds": round(time.time() - t0), "adam_steps": getattr(m, "adam_steps", steps),
                         "_sd": {k: v.detach().clone() for k, v in m.state_dict().items()}})
            p = runs[-1]["params"]
            print(f"  [{name}] start {sd}: loss {L:.5f} h50 {p['h50_min']:.1f} kh {p['kh']:.2f} v50 {p['v50_kmh']:.1f} d50 "
                  f"{(str(round(p['d50_m']['bus'])) + ' (one curve) c_h ' + format(p.get('decay_c_h', 0), '.2f') + ' c_v ' + format(p.get('decay_c_v', 0), '.2f')) if DECAY else '/'.join(f'{v:.0f}' for v in p['d50_m'].values())} kw {p['kw'] if not isinstance(p['kw'], float) else round(p['kw'], 2)} mode {'/'.join(f'{v:.2f}' for v in p['mode'].values())}"
                  + (f" f50 {p['f50_m']:.0f}" if drive else "") + (f" gamma {p['gamma_carfree'][0]:.2f} car_beta {p['car_beta']:.2f}" if level == "sem" else ""), flush=True)
        best = min(range(len(runs)), key=lambda i: runs[i]["loss"])
        flat = lambda p: {k: v for k, v in pd.json_normalize(p, sep=".").iloc[0].items() if isinstance(v, (int, float))}  # noqa: E731
        keys = flat(runs[best]["params"]).keys()
        spread = {k: {"min": min(flat(r["params"])[k] for r in runs), "max": max(flat(r["params"])[k] for r in runs)} for k in keys}
        out = {"name": name, "level": level, "drive": drive, "starts": starts, "cv_cold": CV_COLD, "control_names": getattr(d, "control_names", None), "no_work": NO_WORK, "wlam": WLAM, "metro_ctx": METRO_CTX, "ctrl_metro_means": CTRL_METRO_MEANS, "loss_range": [min(r["loss"] for r in runs), max(r["loss"] for r in runs)],
               "spread": spread, "best": best}
        if cv:
            out["cv"] = cv_skill(d, mk, X, w, runs[best]["_sd"])
            ho_arr = out["cv"].pop("_ho", None)
            if ho_arr is not None:
                np.savez_compressed(RUNS / f"{name}_ho.npz", **ho_arr)
        torch.save(runs[best]["_sd"], RUNS / f"{name}.pt")
        for r in runs:
            r.pop("_sd")
        out["runs"] = runs
        (RUNS / f"{name}.json").write_text(json.dumps(out, indent=1, default=float))
        print(f"[{name}] loss {out['loss_range'][0]:.5f}-{out['loss_range'][1]:.5f}" + (f" cv {json.dumps({k: round(v, 4) for k, v in out['cv'].items() if isinstance(v, float)})}" if cv else ""), flush=True)
        return out
    finally:
        d.keep = keep0


def main():
    e = sys.argv[1]
    only = sys.argv[2:]
    d = Data4()
    if e == "national":
        for level in ("raw", "controls", "sem"):
            for drive in (False, True):
                tag = f"g5_{level}_{'wd' if drive else 'w'}"
                if only and tag not in only:
                    continue
                run(d, tag, level, drive)
    elif e == "regions":   # universality: the same model fitted on each Census region alone (no held-out skill: compared on curves / scores)
        level = only[0] if only else "controls"; drive = "drive" in only
        for reg in ("Northeast", "Midwest", "South", "West"):
            run(d, f"g5_{level}_{'wd' if drive else 'w'}_region_{reg}", level, drive, starts=10, cv=False, keep=(d.region == reg))
    elif e == "metros":   # universality across metros (user 2026-11-15: "do the per metro area tests"): the form fitted on each large metro alone
        level = only[0] if only else "raw"; drive = "drive" in only
        nm = int(os.environ.get("FIT4_N_METROS", "25"))
        tr = np.bincount(d.group, weights=d.y.cpu().numpy() * d.keep.cpu().numpy(), minlength=len(d.group_names))
        top = [g for g in np.argsort(-tr)[:nm] if tr[g] > 0]
        print("metros (transit commuters): " + "; ".join(f"{d.group_names[g]} {tr[g]:,.0f}" for g in top), flush=True)
        for g in top:
            slug = re.sub(r"[^A-Za-z0-9]+", "_", str(d.group_names[g]))[:40].strip("_")
            run(d, f"g5_{level}_{'wd' if drive else 'w'}_metro_{slug}", level, drive, starts=int(os.environ.get("FIT4_RUN_STARTS", "1")), cv=False,
                keep=(d.group == g))
    elif e == "density":   # universality across settings: low / medium / high 1 km density thirds (population-weighted), each fitted alone
        level = only[0] if only else "controls"; drive = "drive" in only
        dd = np.nan_to_num(d.d1km); pw = d.pop.cpu().numpy(); o = np.argsort(dd); cw = np.cumsum(pw[o]) / pw.sum()
        cut = [dd[o][np.searchsorted(cw, q)] for q in (1 / 3, 2 / 3)]
        bands = {"low": dd < cut[0], "medium": (dd >= cut[0]) & (dd < cut[1]), "high": dd >= cut[1]}
        print(f"density bands: {cut[0]:,.0f} / {cut[1]:,.0f} residents per km2 within 1 km", flush=True)
        for b, mask in bands.items():
            run(d, f"g5_{level}_{'wd' if drive else 'w'}_density_{b}", level, drive, starts=10, cv=False, keep=mask)
    elif e == "stability":   # FIT4_CONVERGE / FIT4_NO_SPEED / FIT4_FIX_V50 select the variant; every start saved as it finishes
        tag = os.environ.get("FIT4_TAG", "stab")
        level = only[0] if only else "raw"; drive = "drive" in only; nst = int(os.environ.get("FIT4_STARTS", "4"))
        X = d.control_matrix() if level in ("controls", "sem") else None
        w = t3.weights3(d, True, 0.5); nulls = null_losses(d, w, with_car=level == "sem")
        res = []
        for sd in range(int(os.environ.get("FIT4_SEED0", "0")), nst):   # FIT4_SEED0: resume a run's remaining starts
            m = (ModelR if REPARAM else Model4)(d, level, drive, X.shape[1] if X is not None else 0)
            if sd:
                randomize(m, np.random.default_rng(500 + sd))
            t0 = time.time(); L, parts = fit(m, d, w, nulls, X, steps=FIT_STEPS, quiet=False)
            pr = m.params(); pr.pop("delta", None)
            res.append({"seed": sd, "loss": L, "params": pr, "seconds": round(time.time() - t0), "adam_steps": getattr(m, "adam_steps", 3000),
                        "lbfgs_rounds": getattr(m, "lbfgs_rounds", 1), "nan_restored": getattr(m, "nan_restored", False),
                        "parts": parts, "nulls": {k: v[0] for k, v in nulls.items()}})   # parts: per-task weighted log-likelihood, comparable across runs
            (RUNS / f"{tag}.json").write_text(json.dumps(res, indent=1, default=float))
            print(f"[{tag}] start {sd}: loss {L:.7f} h50 {pr['h50_min']:.2f} kh {pr['kh']:.3f} v50 {pr['v50_kmh']:.2f} kv {pr['kv']:.3f} "
                  f"d50 {'/'.join(f'{v:.0f}' for v in pr['d50_m'].values())} kw {pr['kw'] if isinstance(pr['kw'], float) else '/'.join(f'{v:.3f}' for v in pr['kw'].values())} mode {'/'.join(f'{v:.3f}' for v in pr['mode'].values())} "
                  + (f"c_h {pr['decay_c_h']:.3f} c_v {pr['decay_c_v']:.3f} " if 'decay_c_h' in pr else "") + f"({res[-1]['seconds']}s, {res[-1]['lbfgs_rounds']} lbfgs rounds)", flush=True)
    elif e == "smoke":   # quick end-to-end check
        run(d, "g5_smoke", only[0] if only else "sem", "drive" in only, starts=2, steps=200, cv=False)


if __name__ == "__main__":
    main()
