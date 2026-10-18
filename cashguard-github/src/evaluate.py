"""Video-level evaluation of the rule engine on saved timelines.

Duplicated videos (byte-identical files, e.g. X-4 == X-47) are merged into one video whose labels
are the union of timecodes: the same footage can contain several violations.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.rules import detect_events, verdict


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def merge_duplicates(labels):
    """labels: video_id, violation, t_sec, path [, type] -> one row per unique file, t_list = all timecodes."""
    labels = labels.copy()
    labels["md5"] = labels.path.map(file_md5)
    rows = []
    for _, g in labels.groupby("md5", sort=False):
        ts = sorted(float(t) for t in g.t_sec.dropna())
        rows.append({"video_id": "+".join(g.video_id), "key": g.video_id.iloc[0], "violation": bool(g.violation.any()),
                     "t_list": ts, "types": [t for t in g.get("type", pd.Series(dtype=str)).dropna()]})
    return pd.DataFrame(rows)


def run_rules(videos, timelines_dir, frame_h=848):
    """videos: output of merge_duplicates. Returns {key: [Event]}."""
    out = {}
    for key in videos.key:
        P = pd.read_parquet(Path(timelines_dir) / f"{key}_persons.parquet")
        O = pd.read_parquet(Path(timelines_dir) / f"{key}_objects.parquet")
        pose_f = Path(timelines_dir) / f"{key}_pose.parquet"
        out[key] = detect_events(P, O, frame_h, pose=pd.read_parquet(pose_f) if pose_f.exists() else None)
    return out


def score(videos, events, threshold, tol=2.0):
    rows = []
    for r in videos.itertuples():
        v, kept = verdict(events[r.key], threshold)
        pred = v == "violation"
        err, type_ok = None, None
        if r.violation and pred and r.t_list:
            # each labelled timecode -> nearest predicted event
            errs = [min(abs(e.t - t) for e in kept) for t in r.t_list]
            err = float(min(errs))
            if r.types:
                nearest = min(kept, key=lambda e: min(abs(e.t - t) for t in r.t_list))
                type_ok = nearest.type in r.types
        rows.append({"video": r.video_id, "true": r.violation, "pred": pred, "t_err": err, "type_ok": type_ok,
                     "pred_events": "; ".join(f"{e.type}@{e.t:.1f}({e.score:.2f})" for e in kept)})
    df = pd.DataFrame(rows)
    pos, neg = df[df.true], df[~df.true]
    tp, fp = int(pos.pred.sum()), int(neg.pred.sum())
    errs = pos.t_err.dropna()
    summary = {"threshold": threshold, "recall": f"{tp}/{len(pos)}", "false_alarms": f"{fp}/{len(neg)}",
               "precision": round(tp / max(tp + fp, 1), 3),
               "t_err_median": round(float(errs.median()), 2) if len(errs) else None,
               "t_err_max": round(float(errs.max()), 2) if len(errs) else None,
               f"t_within_{tol:g}s": f"{int((errs <= tol).sum())}/{len(pos)}",
               "type_ok": f"{int(pos.type_ok.fillna(False).sum())}/{len(pos)}"}
    return summary, df
