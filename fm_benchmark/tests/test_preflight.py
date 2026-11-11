"""Guards added after the pre-flight audit of the first real-data runs on TGI: a task is not ready while
its archive is still downloading or extracting, one failing task does not end a `run`, a torn line in
runs.jsonl is skipped, and midog_nobreast keeps midog's split for the images they share."""
import json

import pytest
import torch

from fmbench import paths
from fmbench.tasks.base import DownloadSpec
from test_pipeline import ColourTask


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for var, sub in (("FMBENCH_DATA", "data"), ("FMBENCH_CACHE", "cache"), ("FMBENCH_RESULTS", "results")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    return tmp_path


class ArchiveTask(ColourTask):
    def download_specs(self):
        return [DownloadSpec(url="http://example.invalid/x.zip", filename="x.zip", extract=True, extract_to="x")]


def test_status_waits_for_extraction_marker(env):
    task = ArchiveTask()
    assert task.status().available  # data placed by hand: no archive, no marker

    task.download_dir.mkdir(parents=True)
    part = task.download_dir / "x.zip.part"
    part.write_bytes(b"partial")  # still downloading
    st = task.status()
    assert not st.available and "x.zip" in st.message

    part.rename(task.download_dir / "x.zip")  # downloaded, extraction running (or it failed)
    assert not task.status().available

    (task.raw_dir / "x").mkdir(parents=True)
    (task.raw_dir / "x" / ".extracted-x.zip").write_text("ok\n")  # extract() writes this last
    st = task.status()
    assert st.available and st.counts["test"] == 30


class BrokenTask(ColourTask):
    name = "wilds"

    def official_splits(self):
        raise OSError("truncated file")  # not FileNotFoundError: status() would report it, so fail later

    def status(self):
        from fmbench.tasks.base import TaskStatus

        return TaskStatus(True, "ready")


def test_run_continues_after_a_failing_task(env, monkeypatch):
    import fmbench.cli as cli
    from fmbench.models.adapters import build_encoder
    from fmbench.models.base import ModelSpec
    from fmbench.runner import read_runs

    spec = ModelSpec(name="tiny_run", adapter="torchvision", params={"arch": "resnet18", "weights": "none"},
                     modes=["pooled"], batch_size=16, precision="fp32", resize_to=32)
    monkeypatch.setattr(cli, "_load_model", lambda args: (spec, torch.device("cpu"), build_encoder))
    monkeypatch.setattr(cli, "get_task", lambda name: BrokenTask() if name == "wilds" else ColourTask())
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--model", "tiny_run", "--task", "wilds", "pcam", "--workers", "0",
                  "--iterations", "40", "--n-boot", "10", "--probe-device", "cpu"])
    assert "wilds" in str(exc.value.code)
    runs = read_runs()
    assert [(r["model"], r["task"]) for r in runs] == [("tiny_run", "pcam")]  # pcam still ran after wilds failed


def test_read_runs_skips_a_torn_line(env, capsys):
    from fmbench.runner import append_run, read_runs, runs_path

    append_run({"run_id": "a", "time_utc": "2026-01-01T00:00:00"})
    with runs_path().open("a") as f:
        f.write('{"run_id": "b", "time_')  # a write cut short by another job on another node
        f.write("\n")
    append_run({"run_id": "c", "time_utc": "2026-01-01T00:00:01"})
    assert [r["run_id"] for r in read_runs()] == ["a", "c"]
    assert "not valid JSON" in capsys.readouterr().err


def test_midog_nobreast_keeps_midog_split():
    from fmbench.tasks.midog import build_index

    db = paths.REPO_DIR / "midogpp" / "databases" / "MIDOG++.json"
    if not db.exists():
        pytest.skip("MIDOG++.json not in this checkout")
    coco = json.loads(db.read_text())
    full = {r["ann_id"]: r for r in build_index(coco, ())}
    sub = build_index(coco, ("human breast cancer",))
    assert sub and all(r["tumor_type"] != "human breast cancer" for r in sub)
    assert all(full[r["ann_id"]] == r for r in sub)  # same tile, label and split as in midog
