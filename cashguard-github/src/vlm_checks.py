"""Targeted yes/no questions to a local VLM (Qwen3.5) on candidate moments found in the timelines.

The VLM never judges "is there a violation"; it answers one narrow visual question per moment.
Candidates come from the timelines, so the number of VLM calls stays small:
  Q_COUNT    - hands near cash for a while  -> is the cashier counting banknotes by hand?
  Q_OUTSIDER - two or more people in the room -> is a second person on the cashier's side of the desk?
  Q_SECURE   - the room becomes empty (cashier leaves) -> did the cashier put things into the safe / close it?
"""
import numpy as np
import pandas as pd

from src.rules import Event, changes, hands_near_cash, in_room, people_count, presence

QUESTIONS = {
    "count": ("You see consecutive frames from an overhead camera in a bank cash room. "
              "Is the cashier counting banknotes by hand (fanning or flipping through a stack of bills with the fingers), "
              "rather than using the money counting machine? Answer yes or no."),
    "outsider": ("You see consecutive frames from an overhead camera in a bank cash room. The cashier works at a desk; "
                 "clients sit on the opposite side of the desk. Is there a second person standing on the cashier's side "
                 "of the desk (next to the cashier, behind the desk), not on the client side? Answer yes or no."),
    "secure": ("You see consecutive frames from an overhead camera in a bank cash room, just before the cashier leaves. "
               "Does the cashier put money or items into the grey metal safe or close the safe door before leaving? "
               "Answer yes or no."),
}


def runs(mask, min_len):
    """[(start_t, end_t)] of True runs at least `min_len` samples long."""
    out, start, n, prev = [], None, 0, None
    for t, m in mask.items():
        if m:
            start = t if n == 0 else start
            n += 1
        else:
            if n >= min_len:
                out.append((start, prev))
            n = 0
        prev = t
    if n >= min_len:
        out.append((start, prev))
    return out


def candidates(persons, objects, pose, frame_h, fps_sample=4.0, step_s=3.0):
    """Candidate moments: list of dicts {q, t, frames: [t1, t2, t3]}."""
    P = in_room(persons, frame_h)
    ts = np.sort(np.unique(np.r_[objects.t.unique() if len(objects) else [], P.t.unique()]))
    if not len(ts):
        return []
    n = people_count(P, ts)
    out = []
    # count: hands near cash >= 1.5 s; one moment per 3 s inside each run
    hc = hands_near_cash(pose, objects, ts, frame_h)
    for a, b in runs(hc >= 0.5, int(1.5 * fps_sample)):
        for t in np.arange(a, b + 1e-6, step_s):
            out.append({"q": "count", "t": float(t), "frames": [t, t + 0.75, t + 1.5]})
    # outsider: >= 2 people in the room; one moment per 3 s
    for a, b in runs(n >= 2, int(1.5 * fps_sample)):
        for t in np.arange(a, b + 1e-6, step_s):
            out.append({"q": "outsider", "t": float(t), "frames": [t, t + 0.75, t + 1.5]})
    # secure: every time the room becomes empty; frames from the 10 s before
    for t, a, b in changes(n):
        if a >= 1 and b == 0:
            out.append({"q": "secure", "t": float(t), "frames": [max(t - 9, 0), max(t - 6, 0), max(t - 3, 0), t]})
    return out


# --- state questions: after the cashier leaves / to confirm rule events (states are easier than actions)
QUESTIONS2 = {
    "leftover": ("You see frames from an overhead camera in a bank cash room right after the cashier has left the room. "
                 "Is anything left on the cashier's desk - banknotes, a stack of money, a cash bag, documents or other items "
                 "(not counting the money counting machine, laptop, phone handset and monitor) - or is the grey metal safe left open?"),
    "verify_cash": ("You see frames from an overhead camera in a bank cash room. Are banknotes or a stack of money lying openly "
                    "on the desk surface (not held in someone's hands and not inside the counting machine)?"),
    "verify_handbag": ("You see frames from an overhead camera in a bank cash room. Is there a personal handbag, purse or backpack "
                       "placed on the cashier's desk?"),
    "verify_cup": ("You see frames from an overhead camera in a bank cash room. Is there a cup, mug, glass or drink on the "
                   "cashier's desk or in the cashier's hand?"),
}


def candidates2(persons, objects, frame_h, events):
    """Moments for QUESTIONS2: after every exit (room empty) + every rule event that needs confirmation."""
    P = in_room(persons, frame_h)
    ts = np.sort(np.unique(np.r_[objects.t.unique() if len(objects) else [], P.t.unique()]))
    out = []
    if len(ts):
        n = people_count(P, ts)
        for t, a, b in changes(n):
            if a >= 1 and b == 0:
                out.append({"q": "leftover", "t": float(t), "frames": [t + 1.0, t + 2.0, t + 3.0]})
    for e in events:
        q = {"R1": "verify_cash"}.get(e.rule)
        if e.rule == "R4":
            q = "verify_cup" if "стакан" in e.description else "verify_handbag"
        if q:
            out.append({"q": q, "t": float(e.t), "rule": e.rule, "frames": [e.t - 0.5, e.t + 0.5, e.t + 1.5]})
    return out


class QwenYesNo:
    """Local Qwen3.5 yes/no scorer: P(yes) from the logits of the first answer token."""

    def __init__(self, model_id="Qwen/Qwen3.5-9B", device="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes", "YES"]})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No", "NO"]})

    def _inputs(self, images, question):
        msgs = [{"role": "user", "content": [*({"type": "image", "image": im} for im in images),
                                              {"type": "text", "text": question + " Answer with exactly one word: yes or no."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Answer:"}]}]
        kw = dict(tokenize=True, return_dict=True, return_tensors="pt", continue_final_message=True)
        try:
            return self.proc.apply_chat_template(msgs, enable_thinking=False, **kw).to(self.device)
        except TypeError:
            return self.proc.apply_chat_template(msgs, **kw).to(self.device)

    def p_yes(self, images, question):
        with self.torch.no_grad():
            logits = self.model(**self._inputs(images, question)).logits[0, -1].float()
        y, n = self.torch.logsumexp(logits[self.yes], 0), self.torch.logsumexp(logits[self.no], 0)
        return float(self.torch.sigmoid(y - n))


# Used in the service: only the outsider question separated violations from clean videos
# (Qwen3.5-9B, 40 training videos: clean max p_yes 0.45, violations 0.51-0.69 -> threshold 0.48).
SERVICE_THRESHOLDS = {"count": 2.0, "outsider": 0.48, "secure": -1.0}
# Confirmation of rule events: only the handbag check separated (false X-14: 0.73, true X-40: 0.95).
# Cash-on-desk confirmation did not (true events scored lower than the false one) -> not used.
VERIFY_THRESHOLDS = {"verify_handbag": 0.85}


def verify_question(event):
    if event.rule == "R4" and "сумка" in event.description:
        return "verify_handbag"
    return None


def vlm_events(answers, thresholds):
    """answers: [{q, t, p_yes}] for one video -> extra Events.
    count/outsider: p_yes >= thr -> event; secure: p_yes < thr (did NOT secure) -> leave event."""
    ev = []
    for a in answers:
        thr = thresholds[a["q"]]
        if a["q"] == "count" and a["p_yes"] >= thr:
            ev.append(Event("no_count_or_authenticity_check", a["t"], 0.5 + 0.5 * a["p_yes"], "V1",
                            "Кассир пересчитывает купюры вручную, без счётной машинки."))
        elif a["q"] == "outsider" and a["p_yes"] >= thr:
            ev.append(Event("outsider_in_cash_zone", a["t"], 0.5 + 0.5 * a["p_yes"], "V2",
                            "Второй человек находится на стороне кассира за рабочим столом."))
        elif a["q"] == "secure" and a["p_yes"] < thr:
            ev.append(Event("leave_without_securing", max(a["t"] - 4, 0), 0.5 + 0.5 * (1 - a["p_yes"]), "V3",
                            "Кассир покидает помещение, не убрав ценности в сейф и не закрыв его."))
    return ev
