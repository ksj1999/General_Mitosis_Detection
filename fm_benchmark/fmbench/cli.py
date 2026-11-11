"""Command line interface: python -m fmbench <command> ...

    tasks         list the benchmark tasks and whether their data is ready
    models        list the model configs in configs/models/ (--check: download, load and test each)
    download      download (and prepare) open datasets; print steps for gated ones
    prepare       build tiles / indexes from raw downloads (MIDOG, DLBCL, ...)
    embed         cache frozen embeddings for a model on one or more tasks
    probe         train the Virchow2 linear probe on cached embeddings and record results
    run           embed + probe every available mode, for one model and a set of tasks
    leaderboard   write results/leaderboard.{md,csv} (ours + Virchow2 paper)
    compare       McNemar test between two recorded runs on the same task
    hest-download download HEST-Benchmark at the revision Virchow2 used
    hest          HEST gene-expression regression (Virchow2 Table 3)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import paths
from .tasks import TASK_SETS, get_task, list_tasks


def _task_names(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        out.extend(TASK_SETS.get(v, [v]))
    return list(dict.fromkeys(out))


def cmd_tasks(args):
    print(f"data root:  {paths.data_root()}\ncache root: {paths.cache_root()}\n")
    for name, cls in list_tasks().items():
        task = cls()
        st = task.status() if args.check else None
        state = "" if st is None else ("ready " + str(st.counts) if st.available else st.message)
        print(f"{name:16s} {cls.display_name:14s} access={cls.access:13s} {state}")
        if args.verbose and cls.description:
            print(f"{'':16s} {cls.description}")


def cmd_models(args):
    from .models.base import list_specs, load_spec

    specs = list_specs()
    if not args.check:
        for name, s in specs.items():
            print(f"{name:28s} adapter={s.adapter:12s} modes={','.join(s.modes):18s} {s.display_name}")
        return
    # --check: download each model (into the Hugging Face cache), load it and run a dummy batch.
    import gc

    import torch

    from .models.adapters import build_encoder

    failed = []
    for name in args.names or list(specs):
        try:
            spec = specs[name] if name in specs else load_spec(name)
            enc = build_encoder(spec, torch.device(args.device or "cpu"))
            feats = enc(torch.zeros(2, 3, spec.resize_to, spec.resize_to))
            dims = {k: int(v.shape[1]) for k, v in feats.items()}
            print(f"{'OK':9s}{name:28s} {dims}", flush=True)
            del enc, feats
        except Exception as e:  # noqa: BLE001 - report every model, including gated ones without access
            msg = " ".join(str(e).split())[:300]
            gated = any(w in msg.lower() for w in ("gated", "401", "403", "access to model", "restricted"))
            print(f"{'NOACCESS' if gated else 'FAILED':9s}{name:28s} {type(e).__name__}: {msg}", flush=True)
            failed.append(name)
        gc.collect()
    if failed:
        print(f"{len(failed)} model(s) not usable: {' '.join(failed)}")
        sys.exit(1)


def cmd_download(args):
    from .download import extract, fetch

    for name in _task_names(args.task):
        task = get_task(name)
        print(f"== {name}")
        specs = task.download_specs()
        if not specs:
            print(task.instructions() or "  nothing to download automatically")
        elif not args.force and task.status().available:
            print("  data already in place; skipping download (--force to download again)")
            specs = []
        for spec in specs:
            dest = task.raw_dir / spec.extract_to if spec.extract_to else task.raw_dir
            if spec.extract and not args.force and (dest / f".extracted-{spec.filename}").exists():
                print(f"  {spec.filename}: already extracted; skipping download")
                continue
            f = fetch(spec, task.download_dir)
            if spec.extract:
                extract(f, dest, spec.extract_members)
        if specs and task.instructions():
            print(task.instructions())
        if not args.no_prepare:
            task.prepare(force=args.force)
        print("  status:", task.status())


def cmd_prepare(args):
    for name in _task_names(args.task):
        task = get_task(name)
        print(f"== {name}")
        task.prepare(force=args.force)
        print("  status:", task.status())


def _load_model(args):
    from .embed import pick_device
    from .models.adapters import build_encoder
    from .models.base import load_spec

    spec = load_spec(args.model)
    device = pick_device(args.device)
    if args.batch_size:
        spec.batch_size = args.batch_size
    return spec, device, build_encoder


def cmd_embed(args):
    from .embed import embed_task

    spec, device, build_encoder = _load_model(args)
    encoder = build_encoder(spec, device)
    for name in _task_names(args.task):
        embed_task(spec, encoder, get_task(name), args.limit, args.workers, None, args.overwrite)


def _probe_cfg(args):
    from .probe import ProbeConfig

    cfg = ProbeConfig()
    if args.iterations:
        cfg.iterations = args.iterations
    return cfg


def cmd_probe(args):
    from .embed import available_modes
    from .models.base import load_spec
    from .runner import run_probe

    spec = load_spec(args.model)
    for name in _task_names(args.task):
        task = get_task(name)
        modes = args.mode or [m for m in spec.modes if m in available_modes(spec.name, name, args.limit)]
        if not modes:
            print(f"[probe] {spec.name} x {name}: no cached embeddings; run `fmbench embed` first", file=sys.stderr)
            continue
        for mode in modes:
            for seed in args.seeds:
                run_probe(spec, task, mode, seed, _probe_cfg(args), args.limit, args.probe_device, args.n_boot)


def cmd_run(args):
    from .embed import available_modes, embed_task
    from .runner import run_probe

    spec, device, build_encoder = _load_model(args)
    encoder = None
    failed = []
    for name in _task_names(args.task):
        # one failing task (bad file, cache mismatch, OOM) must not cost the other tasks of a long GPU job
        try:
            task = get_task(name)
            st = task.status()
            if not st.available:
                print(f"[run] skip {name}: {st.message}", file=sys.stderr)
                continue
            if encoder is None:
                encoder = build_encoder(spec, device)
            embed_task(spec, encoder, task, args.limit, args.workers, None, args.overwrite)
            for mode in [m for m in spec.modes if m in available_modes(spec.name, name, args.limit)]:
                for seed in args.seeds:
                    run_probe(spec, task, mode, seed, _probe_cfg(args), args.limit, args.probe_device, args.n_boot)
        except Exception:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[run] FAILED {spec.name} x {name}; continuing with the next task", file=sys.stderr)
            failed.append(name)
            if encoder is not None and str(device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
    from .leaderboard import build

    if args.limit:  # smoke runs: never overwrite results/leaderboard.md
        print("smoke leaderboard:", build(include_smoke=True, out_dir=paths.results_root() / "smoke"))
    else:
        print("leaderboard:", build())
    if failed:
        sys.exit(f"[run] {spec.name}: {len(failed)} task(s) failed: {', '.join(failed)}")


def cmd_leaderboard(args):
    from .leaderboard import build

    print(build(include_smoke=args.include_smoke))


def cmd_compare(args):
    from .metrics import mcnemar
    from .runner import predictions_path

    def load(ref):
        model, mode = ref.split(":")
        d = np.load(predictions_path(model, args.task, mode, args.seed, args.limit))
        return d["y_true"], d["pred"]

    ya, pa = load(args.a)
    yb, pb = load(args.b)
    if not np.array_equal(ya, yb):
        sys.exit("the two runs do not have the same test labels")
    from sklearn.metrics import f1_score

    print(f"{args.a}: weighted F1 {f1_score(ya, pa, average='weighted'):.4f}")
    print(f"{args.b}: weighted F1 {f1_score(yb, pb, average='weighted'):.4f}")
    print("McNemar:", mcnemar(ya, pa, pb))


def cmd_hest_download(args):
    from . import hest

    print(hest.download(args.hest_task, args.revision))


def cmd_hest(args):
    from . import hest
    from .models.base import MODES

    if args.stage == "regress":  # embeddings from an earlier `--stage embed` run: no encoder, no GPU
        if args.overwrite:
            raise SystemExit("--overwrite re-embeds, so it needs --stage embed or all")
        from .models.base import load_spec

        spec, encoder = load_spec(args.model), None
    else:
        spec, device, build_encoder = _load_model(args)
        encoder = build_encoder(spec, device)
    for task in args.hest_task:
        print(f"== HEST {task}")
        if encoder is not None:
            hest.embed(spec, encoder, task, args.revision, args.workers, args.overwrite)
        if args.stage == "embed":
            continue
        dims = hest.cached_dims(spec, task, args.revision)
        modes = args.mode or [m for m in spec.modes if set(MODES[m]) <= dims]
        for mode in modes:
            if args.skip_done and hest.is_done(spec, task, mode, args.method, args.revision, args.rf_max_features):
                print(f"  {mode}: already in runs.jsonl, skipped (--skip-done)")
                continue
            hest.evaluate(spec, task, mode, args.method, args.revision, max_features=args.rf_max_features)
    if args.stage != "embed":
        from .leaderboard import build

        print("leaderboard:", build())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fmbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tasks")
    p.add_argument("--check", action="store_true", help="load every task to report counts")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("models", help="list model configs; --check downloads, loads and runs each one")
    p.add_argument("names", nargs="*", help="models to check (default: all configs)")
    p.add_argument("--check", action="store_true")
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_models)

    for name, fn in (("download", cmd_download), ("prepare", cmd_prepare)):
        p = sub.add_parser(name)
        p.add_argument("task", nargs="+", help=f"task names or sets: {', '.join(TASK_SETS)}")
        p.add_argument("--force", action="store_true")
        if name == "download":
            p.add_argument("--no-prepare", action="store_true")
        p.set_defaults(fn=fn)

    def common(p, model=True):
        if model:
            p.add_argument("--model", required=True, help="model config name or path to a YAML file")
        p.add_argument("--task", nargs="+", default=["paper8"], help=f"task names or sets: {', '.join(TASK_SETS)}")
        p.add_argument("--limit", type=int, default=0, help="smoke test: at most N tiles per split")
        p.add_argument("--device", default=None)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=0)

    p = sub.add_parser("embed")
    common(p)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_embed)

    def probe_args(p):
        p.add_argument("--seeds", type=int, nargs="+", default=[0])
        p.add_argument("--iterations", type=int, default=0, help="override 12,500 (smoke tests only)")
        p.add_argument("--probe-device", default=None)
        p.add_argument("--n-boot", type=int, default=1000)

    p = sub.add_parser("probe")
    common(p)
    p.add_argument("--mode", nargs="+", default=None, help="cls, cls+mean, mean, pooled (default: all cached)")
    probe_args(p)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--overwrite", action="store_true")
    probe_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("leaderboard")
    p.add_argument("--include-smoke", action="store_true")
    p.set_defaults(fn=cmd_leaderboard)

    p = sub.add_parser("compare")
    p.add_argument("--task", required=True)
    p.add_argument("--a", required=True, help="model:mode")
    p.add_argument("--b", required=True, help="model:mode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_compare)

    from .hest import RF_MAX_FEATURES as hest_rf_max_features, TASKS as HEST_TASKS, VIRCHOW2_REVISION

    p = sub.add_parser("hest-download", help="download HEST-Benchmark (about 41 GB for all 10 tasks)")
    p.add_argument("--hest-task", nargs="+", default=HEST_TASKS, choices=HEST_TASKS)
    p.add_argument("--revision", default=VIRCHOW2_REVISION, help="hest-bench revision (default: the one Virchow2 used)")
    p.set_defaults(fn=cmd_hest_download)

    p = sub.add_parser("hest", help="embed HEST-Benchmark patches and run the gene-expression regression")
    p.add_argument("--model", required=True)
    p.add_argument("--hest-task", nargs="+", default=HEST_TASKS, choices=HEST_TASKS)
    p.add_argument("--revision", default=VIRCHOW2_REVISION)
    p.add_argument("--method", default="rf", choices=["rf", "ridge"],
                   help="rf = Virchow2 Table 3 (HEST v1.0.2); ridge = current HEST leaderboard")
    p.add_argument("--mode", nargs="+", default=None)
    p.add_argument("--stage", default="all", choices=["all", "embed", "regress"],
                   help="embed only (GPU), regress only from cached embeddings (CPU is enough), or both")
    p.add_argument("--rf-max-features", default=hest_rf_max_features, type=lambda v: float(v) if v == "1.0" else v,
                   choices=[hest_rf_max_features, 1.0],
                   help="features tried per split: 'sqrt' (cuML 24.06+, default) or 1.0 (cuML 24.04 and earlier; ablation)")
    p.add_argument("--skip-done", action="store_true",
                   help="skip regressions already in runs.jsonl (same model, task, mode, method, max_features, revision)")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_hest)

    args = ap.parse_args(argv)
    from .provenance import code_sha

    code_sha()  # hash the sources now, before a later update of the checkout could change them (cached)
    args.fn(args)


if __name__ == "__main__":
    main()
