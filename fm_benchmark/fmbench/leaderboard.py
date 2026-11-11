"""Merge our probe runs with the published Virchow2 numbers into comparison tables.

Writes <results>/leaderboard.md and <results>/leaderboard.csv. Paper rows come from
reference/virchow2_table2_ood.csv (Table 2 of arXiv 2408.00738v3). Our rows are the mean over
seeds of the latest run per (model, task, mode, seed); smoke runs (--limit) are excluded.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from . import paths
from .models.base import TABLE_GROUP
from .runner import read_runs

OOD_COLUMNS = ["PCam", "CRC", "CRC No-Norm", "WILDS", "TILS", "MHIST", "DLBCL", "MIDOG"]
GROUPS = ["CLS+Mean", "CLS-Only"]
PAPER_MODE_GROUP = {"cls+mean": "CLS+Mean", "mean": "CLS+Mean", "cls": "CLS-Only"}
# Columns whose numbers cannot be compared with the paper. DLBCL: the paper does not define its label; with our
# 5-year survival label about 79% of test tiles are survivors, so always predicting "alive" scores about 70, above
# every DLBCL number in Table 2 (53.9-62.9). See NOTES.md.
NOT_COMPARABLE = {"DLBCL"}
CLASS_BALANCE_TASKS = {"dlbcl", "dlbcl_he"}  # always shown in the class-balance table (plus any task with a cluster CI)


def paper_table2() -> list[dict]:
    rows = []
    with (paths.REFERENCE_DIR / "virchow2_table2_ood.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({"source": "paper", "group": PAPER_MODE_GROUP[r["mode"]], "model": r["model"], "mode": r["mode"],
                         "values": {c: float(r[c]) for c in OOD_COLUMNS}, "paper_average": float(r["Average"]),
                         "leakage": {}})
    return rows


def our_rows(include_smoke: bool = False) -> list[dict]:
    latest: dict = {}
    for r in read_runs():
        if r.get("smoke") and not include_smoke:
            continue
        if r.get("kind", "tile") != "tile" or not r.get("paper_column"):
            continue
        key = (r["model"], r["task"], r["mode"], r["seed"], r.get("limit", 0))
        if key not in latest or r["time_utc"] >= latest[key]["time_utc"]:
            latest[key] = r
    agg: dict = defaultdict(lambda: {"vals": defaultdict(list), "leak": {}, "display": "", "paper_name": None})
    for (model, task, mode, seed, limit), r in latest.items():
        a = agg[(model, mode, limit)]  # smoke runs never share a cell with full runs
        a["vals"][r["paper_column"]].append(100 * r["metrics"]["weighted_f1"])
        a["display"] = r.get("model_display") or model
        a["paper_name"] = r.get("paper_name")
        if r.get("leakage"):
            a["leak"][r["paper_column"]] = r["leakage"]
    rows = []
    for (model, mode, limit), a in sorted(agg.items()):
        vals = {c: sum(v) / len(v) for c, v in a["vals"].items()}
        seeds = {c: len(v) for c, v in a["vals"].items()}
        rows.append({"source": "ours", "group": TABLE_GROUP.get(mode, mode), "model": a["display"], "model_key": model,
                     "mode": _mode_label(mode, limit), "limit": limit, "values": vals, "seeds": seeds,
                     "leakage": a["leak"], "paper_name": a["paper_name"]})
    return rows


def extra_rows(include_smoke: bool = False) -> list[dict]:
    """Our tile runs on tasks without a paper column (mean over seeds of the latest run per seed)."""
    latest: dict = {}
    for r in read_runs():
        if r.get("kind", "tile") != "tile" or r.get("paper_column") or (r.get("smoke") and not include_smoke):
            continue
        key = (r["model"], r["task"], r["mode"], r["seed"], r.get("limit", 0))
        if key not in latest or r["time_utc"] >= latest[key]["time_utc"]:
            latest[key] = r
    agg: dict = {}
    for (model, task, mode, _seed, limit), r in latest.items():
        a = agg.setdefault((model, mode, limit), {"model": r.get("model_display") or model,
                                                  "mode": _mode_label(mode, limit), "vals": defaultdict(list),
                                                  "leakage": {}})
        a["vals"][task].append(100 * r["metrics"]["weighted_f1"])
        if r.get("leakage"):
            a["leakage"][task] = r["leakage"]
    return [{"model": a["model"], "mode": a["mode"], "leakage": a["leakage"],
             "values": {t: sum(v) / len(v) for t, v in a["vals"].items()}} for _, a in sorted(agg.items())]


def _class_balance_table(include_smoke: bool = False) -> list[str]:
    """Weighted F1 next to the majority-class baseline, balanced accuracy, AUROC and the group-level CI.

    Shown for DLBCL (whose weighted F1 mostly reflects the class prior) and for every task whose runs
    carry a patient- or slide-level bootstrap CI. Values are the mean over seeds; CIs are from the lowest seed.
    """
    latest: dict = {}
    for r in read_runs():  # the same choice of runs as the main tables: the latest per (task, model, mode, seed)
        if r.get("kind", "tile") != "tile" or (r.get("smoke") and not include_smoke):
            continue
        key = (r["task"], r["model"], r["mode"], r["seed"], r.get("limit", 0))
        if key not in latest or r["time_utc"] >= latest[key]["time_utc"]:
            latest[key] = r
    groups: dict = defaultdict(list)
    for (task, model, mode, seed, limit), r in latest.items():
        groups[(task, model, mode, limit)].append(r)
    groups = {k: rs for k, rs in groups.items()
              if k[0] in CLASS_BALANCE_TASKS or any(r.get("weighted_f1_ci95_cluster") for r in rs)}
    if not groups:
        return []
    lines = ["## Class balance and group-level uncertainty", "",
             "Weighted F1 rewards predicting the majority class when classes are unbalanced. `Majority` is the weighted "
             "F1 of always predicting the most common training class; a probe can score below it. The by-group CI "
             "resamples whole patients or slides, which is the more realistic interval when the test split has few of "
             "them. All values x 100. Weighted F1, Majority, balanced accuracy and AUROC are means over seeds; both CIs "
             "are from the lowest seed. Runs recorded before these columns existed leave them empty; re-running "
             "`fmbench run` for the model re-uses the cached embeddings and fills them in.", "",
             "| Task | Model | Mode | Weighted F1 | Majority | Balanced acc. | AUROC | 95% CI (tiles) | 95% CI (by group) |",
             "|---|---|---|---|---|---|---|---|---|"]

    def mean(rs, key):
        vals = [r["metrics"].get(key) for r in rs if r["metrics"].get(key) is not None]
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        return _fmt(100 * sum(vals) / len(vals)) if vals else ""

    def ci(r, key):
        v = r.get(key)
        return f"{_fmt(100 * v[0])}-{_fmt(100 * v[1])}" if v and not any(math.isnan(x) for x in v) else ""

    for (task, model, mode, limit), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        first = rs[0]
        unit = first.get("cluster_unit")
        by_group = (f"{ci(first, 'weighted_f1_ci95_cluster')} ({first.get('n_test_clusters')} {unit}s)"
                    if first.get("weighted_f1_ci95_cluster") else "")
        lines.append(f"| {task} | {first.get('model_display') or model} | {_mode_label(mode, limit)} | "
                     f"{mean(rs, 'weighted_f1')} | {mean(rs, 'majority_weighted_f1')} | {mean(rs, 'balanced_accuracy')} | "
                     f"{mean(rs, 'auroc')} | {ci(first, 'weighted_f1_ci95')} | {by_group} |")
    return lines + [""]


def _mode_label(mode: str, limit: int) -> str:
    return f"{mode} [limit {limit}]" if limit else mode


def _avg(values: dict) -> tuple[float, int]:
    have = [values[c] for c in OOD_COLUMNS if c in values]
    return (sum(have) / len(have) if have else math.nan), len(have)


def _fmt(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}"


def build(include_smoke: bool = False, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.results_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    paper, ours = paper_table2(), our_rows(include_smoke)
    lines = ["# Virchow2 tile benchmark: leaderboard", "",
             "Weighted F1 x 100 on the test split of each task (Virchow2 linear-probe protocol). "
             "Paper rows are copied from Table 2 of Virchow2 (arXiv 2408.00738v3). Our rows are the mean over seeds. "
             "The average is shown only when all 8 tasks are present; otherwise the cell shows the partial mean and k/8.", ""]
    csv_rows = []
    for group in GROUPS:
        lines += [f"## {group}", "", "| Source | Model | Mode | " + " | ".join(OOD_COLUMNS) + " | Average |",
                  "|---|---|---|" + "---|" * (len(OOD_COLUMNS) + 1)]
        for r in [x for x in paper if x["group"] == group] + [x for x in ours if x["group"] == group]:
            cells = []
            for c in OOD_COLUMNS:
                v = r["values"].get(c)
                cell = _fmt(v)
                if c in r.get("leakage", {}):
                    cell += " ‡"
                cells.append(cell)
            if r["source"] == "paper":
                avg_cell = _fmt(r["paper_average"])
            else:
                avg, k = _avg(r["values"])
                avg_cell = _fmt(avg) if k == len(OOD_COLUMNS) else (f"({_fmt(avg)}, {k}/8)" if k else "")
            lines.append(f"| {r['source']} | {r['model']} | {r['mode']} | " + " | ".join(cells) + f" | {avg_cell} |")
            csv_rows.append({"group": group, "source": r["source"], "model": r["model"], "mode": r["mode"],
                             **{c: r["values"].get(c, "") for c in OOD_COLUMNS}, "average": avg_cell,
                             "leakage": ";".join(c for c in OOD_COLUMNS if c in r.get("leakage", {}))})
        lines.append("")

    # reproduction check: our run of a published model against its paper row
    repro = []
    paper_idx = {(p["model"], p["group"]): p for p in paper}
    for r in ours:
        p = paper_idx.get((r.get("paper_name"), r["group"]))
        if p and not r.get("limit"):
            repro.append((r, p))
    if repro:
        lines += ["## Reproduction check (ours minus paper)", "",
                  "| Model | Mode | " + " | ".join(OOD_COLUMNS) + " | Mean abs. difference (signed) † |",
                  "|---|---|" + "---|" * (len(OOD_COLUMNS) + 1)]
        for r, p in repro:
            d, comparable = [], []
            for c in OOD_COLUMNS:
                if c not in r["values"]:
                    d.append("")
                    continue
                diff = r["values"][c] - p["values"][c]
                if c in NOT_COMPARABLE:
                    d.append(f"({_fmt(diff)}) †")
                else:
                    d.append(_fmt(diff))
                    comparable.append(diff)
            k = len([c for c in OOD_COLUMNS if c not in NOT_COMPARABLE])
            mean = ""
            if comparable:  # the absolute mean, so that errors of opposite sign do not cancel; the signed mean shows bias
                n = len(comparable)
                signed = round(sum(comparable) / n, 1) + 0.0  # + 0.0 turns -0.0 into 0.0
                mean = (f"{_fmt(sum(abs(x) for x in comparable) / n)} ({signed:+.1f})"
                        + ("" if n == k else f", {n}/{k}"))
            lines.append(f"| {r['model']} | {r['mode']} | " + " | ".join(d) + f" | {mean} |")
        lines += ["", "† Not comparable with the paper: for " + ", ".join(sorted(NOT_COMPARABLE)) + " the paper gives only "
                  "\"binary 5-year survival\" with a patient-level split, not its rule for censored patients or the split "
                  "itself, and with ours the majority-class baseline (69.5) is above every Table 2 DLBCL number (see the "
                  "class-balance table and NOTES.md). It is shown in brackets and left out of the mean differences.", ""]

    extra = extra_rows(include_smoke)
    if extra:
        tasks = sorted({t for r in extra for t in r["values"]})
        lines += ["## Tasks outside Virchow2 Table 2", "",
                  "Weighted F1 x 100. `midog_nobreast` drops the human breast cancer images of MIDOG++; "
                  "`dlbcl_he` keeps only the H&E tiles of DLBCL. No paper numbers exist for these.", "",
                  "| Model | Mode | " + " | ".join(tasks) + " |", "|---|---|" + "---|" * len(tasks)]
        for r in extra:
            cells = [_fmt(r["values"].get(t)) + (" ‡" if t in r["leakage"] else "") for t in tasks]
            lines.append(f"| {r['model']} | {r['mode']} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += _class_balance_table(include_smoke)

    from .hest import markdown as hest_markdown

    lines += hest_markdown()

    if any(r.get("leakage") for r in ours) or any(r.get("leakage") for r in extra):
        lines += ["‡ The model was trained on data from the same source as this task (see `leakage` in runs.jsonl): "
                  "models fine-tuned on the MIDOG++ breast images, for MIDOG, or models pretrained on TCGA slides "
                  "(Phikon, Phikon-v2, CTransPath), for TILS. Where Table 2 has a row for the same model (Phikon, CTransPath), "
                  "that row has the same exposure but is not marked. Use `midog_nobreast` for a cleaner MIDOG comparison; "
                  "TILS has no clean alternative.", ""]
    md = out_dir / "leaderboard.md"
    md.write_text("\n".join(lines))
    with (out_dir / "leaderboard.csv").open("w", newline="") as f:
        # leakage: the columns marked ‡ in leaderboard.md, separated by ";"
        w = csv.DictWriter(f, fieldnames=["group", "source", "model", "mode", *OOD_COLUMNS, "average", "leakage"])
        w.writeheader()
        w.writerows(csv_rows)
    return md
