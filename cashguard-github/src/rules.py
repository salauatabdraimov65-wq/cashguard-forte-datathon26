"""Rule engine: per-video timelines (persons / objects) -> candidate violation events.

Input tables (one row per detection, coordinates in pixels of the original frame):
  persons: t, track, x0, y0, x1, y1, conf
  objects: t, cls, x0, y0, x1, y1, conf
Output: list of events {type, t, score, rule, description}. The video verdict is "violation"
when any event score >= threshold.

Each rule encodes one observable pattern from the training videos (see video_notes.csv).
Rules that rely on classes the draft detector does not see yet (safe_open, reliable cash)
start firing once the detector is retrained on reviewed labels.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

# people seen through the glass wall are small and low-confidence -> keep only in-room people
PERSON_CONF = 0.6
PERSON_MIN_H = 0.11          # min box height as a fraction of frame height  (fallback)
PERSON_REL_H = 0.50         # min box height relative to the 75th percentile of people heights in the video
PERSON_MIN_H_FLOOR = 0.05   # never accept tiny boxes (< 5% of the frame height)
OBJ_CONF = 0.4
SMOOTH = 5                   # frames (~1.25 s at 4 fps) for median smoothing
ARRIVAL_STABLE_S = 2.0       # a new person must stay this long (filters detector flicker)
SAFE_OPEN_MIN_S = 8.0
COUNT_ALONE_MIN_S = 8.0      # R8: seconds of hands-with-cash while alone
COUNT_MACHINE_MAX = 0.05     # R8: max share of hand-cash time with cash at the counting machine        # clean videos: safe opened for work <= ~7 s; violation: left open longer

VIOLATION_TYPES = {
    "prev_client_cash_unsecured": "Ценности предыдущего клиента оставлены открыто",
    "leave_without_securing": "Кассир покинул рабочее место, не убрав ценности",
    "storage_or_keys_unattended": "Сейф/хранилище оставлено открытым без присмотра",
    "personal_items_in_cash_zone": "Посторонние/личные предметы в кассовой зоне",
    "counting_without_controller": "Пересчёт ценностей в одиночку, без второго лица",
    "outsider_in_cash_zone": "Посторонний в зоне кассира",
    "collection_bag_handover": "Передача инкассаторской сумки",
}


@dataclass
class Event:
    type: str
    t: float
    score: float
    rule: str
    description: str

    def as_dict(self):
        return {"type": self.type, "t": round(float(self.t), 2), "score": round(float(self.score), 3),
                "rule": self.rule, "description": self.description}


# ---------------------------------------------------------------- helpers
def min_person_height(persons, frame_h):
    """People behind the glass wall are much smaller than people in the room. The cut-off is relative to the
    people of this video (75th percentile of confident boxes), so it survives another camera distance/angle;
    the fixed fraction of the frame is only a fallback for very short videos."""
    if persons is None or not len(persons):
        return PERSON_MIN_H * frame_h
    h = (persons.y1 - persons.y0)[persons.conf >= PERSON_CONF]
    if len(h) < 8:
        return PERSON_MIN_H * frame_h
    return max(PERSON_MIN_H_FLOOR * frame_h, PERSON_REL_H * float(np.percentile(h, 75)))


def in_room(persons, frame_h):
    if persons is None or not len(persons):
        return pd.DataFrame(columns=["t", "track", "x0", "y0", "x1", "y1", "conf"])
    p = persons.copy()
    h = p.y1 - p.y0
    return p[(p.conf >= PERSON_CONF) & (h >= min_person_height(persons, frame_h))]


def people_count(persons, ts):
    n = persons.groupby("t").size().reindex(ts, fill_value=0)
    return n.rolling(SMOOTH, center=True, min_periods=1).median().round()


def presence(objects, cls, ts, conf=OBJ_CONF):
    """Smoothed 0/1 presence of a class per timestamp."""
    hit = objects[(objects.cls == cls) & (objects.conf >= conf)].t.unique() if len(objects) else []
    s = pd.Series(np.isin(ts, hit).astype(float), index=ts)
    return s.rolling(SMOOTH, center=True, min_periods=1).mean()


def window_frac(series, t0, t1):
    w = series[(series.index >= t0) & (series.index <= t1)]
    return float(w.mean()) if len(w) else 0.0


def changes(n):
    """(t, before, after) for every change of the smoothed people count."""
    v, ts = n.values, n.index.values
    return [(ts[i], v[i - 1], v[i]) for i in range(1, len(v)) if v[i] != v[i - 1]]


def first_run(mask, min_len):
    """Start time of the first run of True of at least `min_len` samples."""
    run, start = 0, None
    for t, m in mask.items():
        if m:
            start = t if run == 0 else start
            run += 1
            if run >= min_len:
                return start
        else:
            run = 0
    return None


def overlap_frac(a, b):
    """Fraction of box b covered by box a."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / max((b[2] - b[0]) * (b[3] - b[1]), 1.0)


def hands_near_cash(pose, objects, ts, frame_h, reach_frac=0.18):
    """Smoothed 0..1 per timestamp: a wrist of an in-room person is within reach of a cash box."""
    if pose is None or not len(pose):
        return pd.Series(0.0, index=ts)
    K = pose[(pose.y1 - pose.y0) >= min_person_height(pose.assign(conf=1.0), frame_h)]
    C = objects[(objects.cls == "cash") & (objects.conf >= OBJ_CONF)] if len(objects) else objects
    hit = {}
    for t, g in C.groupby("t"):
        found = False
        for kp in K[(K.t - t).abs() < 0.13].itertuples():
            reach = reach_frac * (kp.y1 - kp.y0)
            for wx, wy, wc in ((kp.l_wrist_x, kp.l_wrist_y, kp.l_wrist_c), (kp.r_wrist_x, kp.r_wrist_y, kp.r_wrist_c)):
                if wc < 0.3:
                    continue
                for c in g.itertuples():
                    dx, dy = max(c.x0 - wx, 0, wx - c.x1), max(c.y0 - wy, 0, wy - c.y1)
                    found |= (dx * dx + dy * dy) ** 0.5 <= reach
        hit[t] = found
    s = pd.Series([float(hit.get(t, False)) for t in ts], index=ts)
    return s.rolling(SMOOTH, center=True, min_periods=1).mean()


def cash_at_machine_times(objects):
    """Timestamps where a cash box touches a counting machine box (money goes through the machine)."""
    if not len(objects):
        return set()
    cash = objects[(objects.cls == "cash") & (objects.conf >= OBJ_CONF)]
    mach = objects[(objects.cls == "counting_machine") & (objects.conf >= 0.5)]
    out = set()
    for t, g in cash.groupby("t"):
        for m in mach[(mach.t - t).abs() < 0.13].itertuples():
            pad = 0.15 * (m.x1 - m.x0)
            if any(not (c.x1 < m.x0 - pad or c.x0 > m.x1 + pad or c.y1 < m.y0 - pad or c.y0 > m.y1 + pad) for c in g.itertuples()):
                out.add(t)
    return out


# ---------------------------------------------------------------- rules
def detect_events(persons, objects, frame_h, fps_sample=4.0, pose=None):
    persons = in_room(persons, frame_h)
    if objects is None:
        objects = pd.DataFrame(columns=["t", "cls", "x0", "y0", "x1", "y1", "conf"])
    ts = np.sort(np.unique(np.r_[objects.t.unique() if len(objects) else [], persons.t.unique()]))
    if not len(ts):
        return []
    sec = lambda s: max(1, int(round(s * fps_sample)))  # noqa: E731
    n = people_count(persons, ts)
    cash = presence(objects, "cash", ts)
    bag = presence(objects, "collection_bag", ts)
    safe_open = presence(objects, "safe_open", ts)
    events = []

    # R1: a new person arrives (and stays) while cash is already lying out (previous client's cash)
    for t, a, b in changes(n):
        stable = window_frac(n, t, t + ARRIVAL_STABLE_S) >= b - 0.25 and window_frac(n, t - ARRIVAL_STABLE_S, t - 0.1) <= a + 0.25
        if b > a and a >= 1 and stable:
            before, around = window_frac(cash, t - 3, t), window_frac(cash, t - 1, t + 2)
            if before >= 0.3 and around >= 0.3:
                events.append(Event("prev_client_cash_unsecured", t, 0.5 + 0.5 * min(before, around), "R1",
                                    "В момент прихода следующего клиента на столе открыто лежат купюры."))

    # R2: everybody leaves the room while valuables stay on the desk / the safe stays open
    for t, a, b in changes(n):
        if a >= 1 and b == 0 and window_frac(n, t, t + 2) == 0:
            left = {"купюры": window_frac(cash, t, t + 3), "инкассаторская сумка": window_frac(bag, t, t + 3)}
            items = [k for k, v in left.items() if v >= 0.5]
            opened = window_frac(safe_open, t - 1, t + 3) >= 0.5
            if items or opened:
                what = ", ".join(items + (["открытый сейф"] if opened else []))
                events.append(Event("leave_without_securing", max(t - 2, 0), 0.8, "R2",
                                    f"Кассир покидает помещение, на рабочем месте остаются: {what}."))

    # R3: safe open with nobody next to it
    safe_boxes = objects[objects.cls == "safe_open"]
    if len(safe_boxes):
        sb = safe_boxes[["x0", "y0", "x1", "y1"]].median().values
        near = persons.groupby("t").apply(
            lambda g: any(overlap_frac(r, sb) > 0.1 for r in g[["x0", "y0", "x1", "y1"]].values)
        ).reindex(ts, fill_value=False)
        long_open = first_run(safe_open >= 0.5, sec(SAFE_OPEN_MIN_S))
        start = first_run((safe_open >= 0.5) & ~near.astype(bool), sec(2)) if long_open is not None else None
        if long_open is not None:
            start = long_open if start is None else min(start, long_open + 2.0)
            events.append(Event("storage_or_keys_unattended", start, 0.8, "R3",
                                "Дверца сейфа открыта, рядом с сейфом никого нет."))

    # R4: personal items (handbag, cup) on the desk
    for cls, what, min_s, conf in (("handbag", "личная сумка", 2, 0.5), ("cup", "стакан/напиток", 1, OBJ_CONF)):
        start = first_run(presence(objects, cls, ts, conf=conf) >= 0.5, sec(min_s))
        if start is not None:
            events.append(Event("personal_items_in_cash_zone", start, 0.7, "R4",
                                f"В кассовой зоне рядом с ценностями находится {what}."))

    # R5: one person alone handles cash for a while (no controller / client)
    alone_cash = (n == 1) & (cash >= 0.5)
    start = first_run(alone_cash, sec(3))
    if start is not None:
        events.append(Event("counting_without_controller", start, 0.3, "R5",
                            "Кассир один в помещении и работает с наличными без второго лица."))

    # R6: three or more people in the room for a while (someone beyond cashier + client)
    start = first_run(n >= 3, sec(2))
    if start is not None:
        events.append(Event("outsider_in_cash_zone", start, 0.4, "R6",
                            "В помещении одновременно находится третий человек помимо кассира и клиента."))

    # R7: collection bag handed over (weak: clean videos also contain handovers)
    start = first_run((bag >= 0.5) & (n >= 2), sec(1))
    if start is not None:
        events.append(Event("collection_bag_handover", start, 0.35, "R7",
                            "Инкассаторская сумка передаётся кассиру; проверка сумки не наблюдается."))

    # R8: cashier alone counts cash by hand for a long time and the money never goes through the machine
    # (clean solo videos: >= 16% of hand-cash time at the machine; violation X-44: 0%)
    if pose is not None and len(pose):
        hc = hands_near_cash(pose, objects, ts, frame_h)
        hands = hc >= 0.5
        alone_hands = hands & (n == 1)
        machine_ratio = len(cash_at_machine_times(objects)) / max(hands.sum(), 1)
        if alone_hands.sum() / fps_sample >= COUNT_ALONE_MIN_S and machine_ratio <= COUNT_MACHINE_MAX:
            start = first_run(alone_hands, sec(2))
            if start is None:
                start = float(alone_hands.index[alone_hands.values.argmax()])
            events.append(Event("counting_without_controller", start, 0.7, "R8",
                                "Кассир один пересчитывает наличные вручную, без счётной машинки и без второго лица."))
    return events


def verdict(events, threshold=0.5, min_gap=3.0, same_type_gap=10.0, max_events=3):
    """Keep events above threshold. One video may hold several violations (e.g. X-4 == X-47:
    57 s and 68 s), so the same type is kept again if it is more than `same_type_gap` s apart."""
    kept = sorted((e for e in events if e.score >= threshold), key=lambda e: -e.score)
    out = []
    for e in kept:
        if any(abs(o.t - e.t) < (same_type_gap if o.type == e.type else min_gap) for o in out):
            continue
        out.append(e)
        if len(out) >= max_events:
            break
    return ("violation" if out else "no_violation"), sorted(out, key=lambda e: e.t)
