"""
shared machinery for the four step0 stages (thin CLIs: step0a-step0d).

  step0a_rl_runs      : collect RL behavior replays (upright + chosen alphas, all seeds)
  step0b_fit_flows    : train 1-D spline flows on replays
  step0c_gate         : run G1-G4 acceptance checks (blocking, content-addressed)
  step0d_stamp_strata : measure realized K1 for the CHOSEN alphas (labels, not
                        selection; strata are picked directly via
                        config rl_runs.alphas_chosen), warn on weak separation
                        (advisory; the pilot gate still guards the campaign),
                        and write the canonical alphas_chosen.yaml.

each stage is idempotent (load-or-compute, sha256-keyed caches); --force retrains.
config is read-only; alphas_chosen.yaml is written to config['data_dir'].
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from src.utils.io import _load_config
from src.utils.pendulum import PendulumCfg
from src.utils.pendulum_q import QGridCfg
from src.utils.rl_behavior_source import (
    SoftQTrainerCfg, load_or_collect_replay, replay_cache_path)
from src.models.flow.train_flow_policy import (
    FlowTrainCfg, load_or_train_flow, flow_ckpt_path, heldout_indices)
from ex.utils.flow_gate import run_gate, assert_gate
from ex.utils.realized_kl_table import measure_realized_k1, write_alphas_chosen


def build_env_and_q_cfg(config):
    """extract PendulumCfg and QGridCfg from config dict.

    args:
        config: loaded yaml config dict

    returns:
        tuple (env_cfg: PendulumCfg, q_cfg: QGridCfg)
    """
    pend = config["pendulum"]
    q_gr = config["q_grid"]

    # mu0_box: read nested theta + theta_dot bounds and coerce into frozen tuples
    mu0_dict = pend["mu0"]
    mu0_box = (
        tuple(float(x) for x in mu0_dict["theta_bounds"]),
        tuple(float(x) for x in mu0_dict["theta_dot_bounds"]),
    )

    env_cfg = PendulumCfg(
        g=float(pend["g"]),
        ell=float(pend["ell"]),
        m=float(pend["m"]),
        dt=float(pend["dt"]),
        action_clip=float(pend["action_clip"]),
        theta_dot_clip=float(pend["theta_dot_clip"]),
        mu0_box=mu0_box,
    )

    q_cfg = QGridCfg(
        N_theta=int(q_gr["N_theta"]),
        N_theta_dot=int(q_gr["N_theta_dot"]),
        N_action=int(q_gr["N_action"]),
        gamma=float(q_gr["gamma"]),
        fqi_max_iter=int(q_gr["fqi_max_iter"]),
        fqi_tol=float(q_gr["fqi_tol"]),
    )

    return env_cfg, q_cfg


def build_flow_cfg(config, seed=None):
    """construct FlowTrainCfg from config dict.

    args:
        config: loaded yaml config dict
        seed: optional seed override for unique per-replay cfgs

    returns:
        FlowTrainCfg instance with all required fields
    """
    flow_cfg = config["flow"]
    train_cfg = flow_cfg["train"]

    cfg_seed = seed if seed is not None else int(train_cfg.get("seed", 0))

    return FlowTrainCfg(
        steps=int(train_cfg["steps"]),
        batch_size=int(train_cfg["batch"]),  # config key is "batch"
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        ema_decay=float(train_cfg["ema_decay"]),
        heldout_frac=float(train_cfg["heldout_frac"]),
        warmup_frac=float(train_cfg["warmup_frac"]),
        seed=cfg_seed,
        hidden_dim=int(flow_cfg["hidden_dim"]),
        n_layers=int(flow_cfg["n_layers"]),
        n_bins=int(flow_cfg["n_bins"]),
    )



def policy_specs(config):
    """(name, r_name, alpha) triples: expert + the config-chosen alphas
    (+ optional rl_runs.extra_diag_alphas, trained only for diagnostics)."""
    rl_cfg = config["rl_runs"]
    specs = [("upright", "r_upright", 0)]
    for alpha in list(rl_cfg["alphas_chosen"]) + list(rl_cfg.get("extra_diag_alphas", [])):
        specs.append((f"occupancy_a_{alpha:.4e}", "r_O", alpha))
    return specs


def get_ckpt_path(config, env_cfg, q_cfg, trainer_cfg, r_name, alpha, seed,
                  flow_seed=None):
    """resolve the content-addressed flow ckpt path for one policy.

    delegates to replay_cache_path + flow_ckpt_path so step0 never replicates
    hash logic. `seed` is the RL replay seed (campaign uses 0 only);
    `flow_seed` is the FLOW TRAINING seed (defaults to `seed`); the gate's
    G4 refits the SAME replay with flow seeds 0..n-1 to test label stability
    under refitting. requires the replay file to exist (run step0a first).
    """
    replay_path = replay_cache_path(
        env_cfg, q_cfg, r_name, alpha, trainer_cfg, seed,
        config["rl_runs"]["cache_dir"])
    if not Path(replay_path).exists():
        raise FileNotFoundError(
            f"missing replay {replay_path} for r_name={r_name} alpha={alpha} "
            f"seed={seed}; run step0a first")
    fs = seed if flow_seed is None else flow_seed
    ckpt_path, _ = flow_ckpt_path(
        replay_path, build_flow_cfg(config, seed=fs),
        config["flow"]["ckpt_dir"])
    return ckpt_path


def mode_rl(config, args, env_cfg, q_cfg, trainer_cfg):
    """collect RL replays for upright + chosen policies (seed 0 only).

    the campaign uses a single behavior run per policy; the gate's G4 tests
    refit stability by retraining FLOWS with different seeds on this same
    replay, so no multi-seed RL collection is needed.
    """
    rl_cfg = config["rl_runs"]
    cache_dir = rl_cfg["cache_dir"]

    policies = policy_specs(config)

    print(f"RL replay collection: {len(policies)} policies, seed 0")

    for policy_name, r_name, alpha in policies:
        for seed in [0]:
            try:
                h5_path = load_or_collect_replay(
                    env_cfg=env_cfg,
                    q_cfg=q_cfg,
                    r_name=r_name,
                    alpha=alpha,
                    trainer_cfg=trainer_cfg,
                    seed=seed,
                    cache_dir=cache_dir,
                    force=args.force,
                )

                # extract n_pairs from h5 data
                with h5py.File(h5_path, "r") as f:
                    n_pairs = f["states"].shape[0]
                    n_phases = 3  # fixed: early, mid, late thirds

                print(
                    f"{policy_name:15s} alpha={alpha:6.2f} seed={seed} -> {n_pairs:7d} pairs in {n_phases} phases"
                )

            except Exception as e:
                print(f"RL run failed: {policy_name} alpha={alpha} seed={seed}. Check logs. Partial ladder detected.")
                raise

    print("RL replay collection complete: all policies, all seeds cached")
    print(f"✓ step0 rl done.")


def mode_flows(config, args, env_cfg, q_cfg, trainer_cfg):
    """train flows on replays (all seeds for gate stability)."""
    rl_cfg = config["rl_runs"]
    n_seeds_gate = rl_cfg["n_seeds_gate"]
    cache_dir = rl_cfg["cache_dir"]
    flow_ckpt_dir = config["flow"]["ckpt_dir"]

    policies = policy_specs(config)

    print(f"Flow training: {len(policies)} policies × {n_seeds_gate} flow seeds "
          "on the SAME seed-0 replay (G4 refit-stability design)")

    # device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    for policy_name, r_name, alpha in policies:
        for seed in range(n_seeds_gate):
            # prerequisite: the single seed-0 replay must exist
            try:
                replay_path = load_or_collect_replay(
                    env_cfg=env_cfg,
                    q_cfg=q_cfg,
                    r_name=r_name,
                    alpha=alpha,
                    trainer_cfg=trainer_cfg,
                    seed=0,
                    cache_dir=cache_dir,
                    force=False,  # just check existence
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Missing replay for policy={policy_name}. Run step0a first."
                )

            # refit the same replay with flow-training seed `seed`
            flow_cfg_obj = build_flow_cfg(config, seed=seed)
            ckpt_path = load_or_train_flow(
                replay_path=replay_path,
                cfg=flow_cfg_obj,
                cache_dir=flow_ckpt_dir,
                device=device,
                force=args.force,
            )

            # extract hash for logging
            ckpt_hash = Path(ckpt_path).stem.split("_")[1][:8]
            print(f"Flow trained: replay_hash={ckpt_hash}... policy={policy_name} alpha={alpha} seed={seed}")

    print("Flow training complete: all campaign flows ready")
    print(f"✓ step0 flows done.")


def mode_gate(config, args, env_cfg, q_cfg, trainer_cfg):
    """run oracle-free acceptance gate checks (G1-G4) across all seeds."""
    rl_cfg = config["rl_runs"]
    n_seeds_gate = rl_cfg["n_seeds_gate"]
    cache_dir = rl_cfg["cache_dir"]
    gate_out_dir = Path(config["gate"]["out_dir"])
    gate_out_dir.mkdir(parents=True, exist_ok=True)

    policies = policy_specs(config)

    # prerequisite: all refit flows (replay seed 0, flow seeds 0..n-1) exist
    for policy_name, r_name, alpha in policies:
        for fs in range(n_seeds_gate):
            ckpt_path = get_ckpt_path(
                config, env_cfg, q_cfg, trainer_cfg, r_name, alpha,
                seed=0, flow_seed=fs)
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(
                    f"Missing flow for {policy_name} flow_seed={fs}. "
                    "Run step0b first.")

    print(f"Gate checks: {len(policies)} policies × {n_seeds_gate} seeds")

    for policy_name, r_name, alpha in policies:
        # load primary replay for held-out evaluation
        primary_replay_path = load_or_collect_replay(
            env_cfg=env_cfg,
            q_cfg=q_cfg,
            r_name=r_name,
            alpha=alpha,
            trainer_cfg=trainer_cfg,
            seed=0,
            cache_dir=cache_dir,
            force=False,
        )

        with h5py.File(primary_replay_path, "r") as f:
            all_states = f["states"][:]  # [N, 2]
            all_actions = f["actions"][:]  # [N]

        # gate evaluates only the trainer's held-out split (never train data);
        # heldout_indices is the shared single source of truth for the split.
        hidx = heldout_indices(len(all_actions), build_flow_cfg(config, seed=0))
        heldout_states = all_states[hidx]
        heldout_actions = all_actions[hidx]

        # gather ckpt paths for all seeds
        ckpt_paths = [
            get_ckpt_path(config, env_cfg, q_cfg, trainer_cfg, r_name, alpha,
                          seed=0, flow_seed=fs)
            for fs in range(n_seeds_gate)]

        # construct cross_actions from other policies
        cross_actions = {}
        for other_policy_name, other_r_name, other_alpha in policies:
            if policy_name != other_policy_name:
                other_replay_path = load_or_collect_replay(
                    env_cfg=env_cfg,
                    q_cfg=q_cfg,
                    r_name=other_r_name,
                    alpha=other_alpha,
                    trainer_cfg=trainer_cfg,
                    seed=0,
                    cache_dir=cache_dir,
                    force=False,
                )
                with h5py.File(other_replay_path, "r") as f:
                    o_states = f["states"][:]
                    o_actions = f["actions"][:]
                # held-out slice of the other policy's replay (unseen data)
                o_hidx = heldout_indices(
                    len(o_actions), build_flow_cfg(config, seed=0))
                cross_actions[other_policy_name] = {
                    "states": o_states[o_hidx],
                    "actions": o_actions[o_hidx],
                }

        # measure realized K1 for seed stability (G4)
        # G4 (same-data refit stability): for each flow-training seed fs,
        # re-measure the stratum label K1(E-flow(fs) || O-flow(fs)) with both
        # flows refit on their unchanged seed-0 replays. label swings across
        # fs would mean the labels are optimizer artifacts, not properties of
        # the behavior data. the expert has no own K1 -> exempt (None); its
        # refit stability is exercised inside every stratum's joint measure.
        realized_kl_by_seed = None
        if r_name != "r_upright":
            realized_kl_by_seed = {}
            for fs in range(n_seeds_gate):
                ckpt_E_fs = get_ckpt_path(
                    config, env_cfg, q_cfg, trainer_cfg, "r_upright", 0,
                    seed=0, flow_seed=fs)
                measured = measure_realized_k1(
                    ladder=[{"alpha": alpha, "ckpt_O": ckpt_paths[fs]}],
                    ckpt_E=ckpt_E_fs,
                    env_cfg=env_cfg,
                    T=int(config["trajectory"]["T"]),
                    M=200,
                    seed=0,
                )
                if measured:
                    realized_kl_by_seed[f"flow_seed{fs}"] = measured[0]["kl_hat"]

        # run gate checks
        gate_report = run_gate(
            policy_label=policy_name,
            ckpt_paths=ckpt_paths,
            heldout={"states": heldout_states, "actions": heldout_actions},
            cross_actions=cross_actions,
            realized_kl_by_seed=realized_kl_by_seed,
            out_dir=str(gate_out_dir),
        )

        accepted = gate_report.get("accepted", False)
        verdict = "ACCEPTED" if accepted else "REJECTED"
        print(f"Policy {policy_name:15s} alpha={alpha:6.2f}: {verdict} (G1-G4)")

    print("All gate checks passed: ready for difficulty selection.")
    print(f"✓ step0 gate done.")


def mode_stamp_strata(config, args, env_cfg, q_cfg, trainer_cfg):
    """measure realized K1 for the CHOSEN alphas and stamp alphas_chosen.yaml.

    strata are selected DIRECTLY via config rl_runs.alphas_chosen (alpha is
    the knob); the K1 measurement here provides the stratum LABELS written
    into cell attrs, not a selection mechanism. separation is checked as an
    ADVISORY warning only; the 5-cell pilot remains the campaign guard.
    ladder-style sweeps over extra alphas live in diagnostics, not here.
    """
    rl_cfg = config["rl_runs"]
    alphas_chosen = [float(a) for a in rl_cfg["alphas_chosen"]]
    cache_dir = rl_cfg["cache_dir"]
    gate_out_dir = config["gate"]["out_dir"]
    alpha_sel_cfg = config["alpha_selection"]
    data_dir = config["data_dir"]

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # prerequisite: gate reports accepted for expert + every chosen alpha
    ckpt_E = get_ckpt_path(
        config, env_cfg, q_cfg, trainer_cfg, "r_upright", 0, seed=0)
    ckpts_O = {}
    for alpha in alphas_chosen:
        ckpts_O[alpha] = get_ckpt_path(
            config, env_cfg, q_cfg, trainer_cfg, "r_O", alpha, seed=0)
    for label, ckpt in [("upright", ckpt_E)] + [
            (f"alpha={a:g}", c) for a, c in ckpts_O.items()]:
        try:
            assert_gate(gate_out_dir, ckpt)
        except (FileNotFoundError, RuntimeError):
            raise RuntimeError(
                f"Gate rejection for {label} blocks stamping. "
                "Check gate reports and rerun step0a/step0b/step0c.")

    # measure realized K1 (+SE) for the chosen alphas under the flows
    measured = measure_realized_k1(
        ladder=[{"alpha": a, "ckpt_O": ckpts_O[a]} for a in alphas_chosen],
        ckpt_E=ckpt_E,
        env_cfg=env_cfg,
        T=int(config["trajectory"]["T"]),
        M=int(alpha_sel_cfg["M"]),
        seed=0,
    )

    # order strata by realized difficulty; labels 0..n-1 ascending
    measured = sorted(measured, key=lambda e: float(e["kl_hat"]))

    # advisory separation check (warn, never block)
    sep_mult = float(alpha_sel_cfg["separation_mult"])
    for lo, hi in zip(measured[:-1], measured[1:]):
        gap = float(hi["kl_hat"]) - float(lo["kl_hat"])
        need = sep_mult * (float(lo["kl_se"]) + float(hi["kl_se"]))
        if gap < need:
            print(
                f"WARNING: strata alpha={lo['alpha']:g} and alpha={hi['alpha']:g} "
                f"realized K1 gap {gap:.3f} < {sep_mult}x combined SE {need:.3f}; "
                "consider different alphas_chosen before the campaign.")

    # canonical doc via the shared writer (single source of truth for schema)
    from ex.utils.flow_gate import _compute_ckpt_hash

    strata = []
    for i, entry in enumerate(measured):
        a = float(entry["alpha"])
        replay_O = replay_cache_path(
            env_cfg, q_cfg, "r_O", a, trainer_cfg, 0, cache_dir)
        strata.append({
            "alpha": a,
            "ckpt_O": str(entry["ckpt_O"]),
            "flow_hash_O": _compute_ckpt_hash(str(entry["ckpt_O"])),
            "rl_hash_O": Path(replay_O).stem.removeprefix("replay_"),
            "K1_realized_flow": float(entry["kl_hat"]),
            "K1_se": float(entry["kl_se"]),
            "stratum_label": int(i),
        })
    ladder_doc = [{"alpha": float(e["alpha"]), "kl_hat": float(e["kl_hat"]),
                   "kl_se": float(e["kl_se"])} for e in measured]
    write_alphas_chosen(data_dir, ckpt_E, strata, ladder_doc)

    k1s = [s["K1_realized_flow"] for s in strata]
    print(f"Stamped {len(strata)} strata (alphas {[s['alpha'] for s in strata]}, "
          f"realized K1 {['%.2f' % k for k in k1s]}) to alphas_chosen.yaml.")
    print("v step0d stamp-strata done.")



def mode_scout(config, args, env_cfg, q_cfg, trainer_cfg):
    """gate-free realized-K1 sweep over ALL trained candidate alphas.

    the alpha-iteration tool: measures K1(p^pi_E || p^pi_O(alpha)) +- SE under
    the (possibly scout-scale) flows for every candidate in alphas_chosen +
    extra_diag_alphas whose seed-0 flow ckpt exists, prints a sorted table
    with adjacent-separation advisories, and writes {data_dir}/scout_k1.yaml.
    informational only: no gates required, nothing downstream consumes it.
    """
    from ex.utils.realized_kl_table import measure_realized_k1

    rl_cfg = config["rl_runs"]
    candidates = [float(a) for a in
                  list(rl_cfg["alphas_chosen"]) + list(rl_cfg.get("extra_diag_alphas", []))]
    if not candidates:
        raise ValueError("no candidate alphas in alphas_chosen/extra_diag_alphas")

    ckpt_E = get_ckpt_path(
        config, env_cfg, q_cfg, trainer_cfg, "r_upright", 0, seed=0)
    ladder = []
    for a in sorted(set(candidates)):
        try:
            ckpt_O = get_ckpt_path(
                config, env_cfg, q_cfg, trainer_cfg, "r_O", a, seed=0)
        except FileNotFoundError:
            print(f"skip alpha={a:g}: replay missing (run step0a first)")
            continue
        if not Path(ckpt_O).exists():
            print(f"skip alpha={a:g}: flow ckpt missing (run step0b first)")
            continue
        ladder.append({"alpha": a, "ckpt_O": ckpt_O})

    if not ladder:
        raise FileNotFoundError("no trained candidate flows found; run step0a/step0b first")

    measured = measure_realized_k1(
        ladder=ladder, ckpt_E=ckpt_E, env_cfg=env_cfg,
        T=int(config["trajectory"]["T"]),
        M=int(config["alpha_selection"]["M"]), seed=0)
    measured = sorted(measured, key=lambda e: float(e["kl_hat"]))

    sep_mult = float(config["alpha_selection"]["separation_mult"])
    print(f"\nscout K1 table (M={config['alpha_selection']['M']}, "
          f"flow steps={config['flow']['train']['steps']}):")
    print(f"{'alpha':>10} {'K1':>10} {'SE':>8}")
    for e in measured:
        print(f"{e['alpha']:>10.4g} {e['kl_hat']:>10.3f} {e['kl_se']:>8.3f}")
    for lo, hi in zip(measured[:-1], measured[1:]):
        gap = float(hi["kl_hat"]) - float(lo["kl_hat"])
        need = sep_mult * (float(lo["kl_se"]) + float(hi["kl_se"]))
        if gap < need:
            print(f"  advisory: alpha={lo['alpha']:g} vs alpha={hi['alpha']:g} "
                  f"gap {gap:.3f} < {sep_mult}x combined SE {need:.3f}")

    # filename carries the flow scale: scout tables at different flow
    # quality coexist (the flow is the GT, so K1 depends on fit quality)
    tag = config.get("scout_tag", "")
    tag = f"{tag}_" if tag else ""
    out = Path(config["data_dir"]) / f"scout_k1_{tag}fs{int(config['flow']['train']['steps'])}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump({"flow_steps": int(config["flow"]["train"]["steps"]),
                        "M": int(config["alpha_selection"]["M"]),
                        "table": [{"alpha": float(e["alpha"]),
                                   "kl_hat": float(e["kl_hat"]),
                                   "kl_se": float(e["kl_se"])} for e in measured]}, f,
                       sort_keys=False)
    print(f"written {out}")
    print("step0 scout done.")


def cli(mode_fn, description):
    """shared thin-CLI entry: parse --config/--force, build cfgs, run one mode."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str,
                        default="ex/semisynth/pendulum/config.yaml",
                        help="config yaml (e.g. config_scout.yaml for alpha scouting)")
    parser.add_argument("--force", action="store_true",
                        help="force recomputation (ignore caches)")
    args = parser.parse_args()
    config = _load_config(args.config)
    env_cfg, q_cfg = build_env_and_q_cfg(config)
    trainer_cfg = SoftQTrainerCfg(**config["rl_runs"]["trainer"])
    mode_fn(config, args, env_cfg, q_cfg, trainer_cfg)
