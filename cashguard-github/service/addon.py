"""Add-on for the ForteBank analyzer: extra checks with the already loaded Qwen, on top of the base pipeline.

Three blocks (each can be switched off; `ablation()` shows which ones really raise verdict accuracy):
  A. "audit"     — end-state audit: what is left open/unattended at the end of the clip
                   (laptop unlocked, valuables on an empty desk, door open, safe/drawer open);
                   walks back in time to find when it started (timecode).
  B. "checklist" — questions derived from the regulations (NBRK No.231/No.14), asked on a storyboard of
                   the whole clip, incl. violation types absent from the training videos
                   (photographing money, leaving mid-service, collector alone, cash into a pocket...).
  C. "safety"    — open question "is anything breaking the rules?" + mandatory second-question check,
                   so the model cannot simply make things up. Catches violation types nobody listed.
No fixed screen zones: works for any camera angle. Thresholds are set by meaning, not tuned per video.

Notebook (after `an = Analyzer(...)` exists; the model is NOT loaded again):
    import sys, importlib; sys.path.insert(0, "/workspace")
    import addon; importlib.reload(addon)
    an = addon.AnalyzerPlus(getattr(an, "base", an))
    addon.sanity(an, files)                                   # ~30 s check of the audit questions
    plus = run("plus_clean", lambda r: files[r.key])          # base + all blocks
    addon.ablation(RESULTS / "tta_plus_clean.csv")            # accuracy of every block combination
"""
import base64
import json
import os
import re
import time

import cv2
from PIL import Image

LOG = os.environ.get("ADDON_LOG", "/workspace/results/addon_log.jsonl")
TIME_BUDGET_S = 150          # whole add-on per video; keeps the API far below its 5-minute limit
MAX_BACK_STEPS = 15          # how far back (x2 s) the audit searches for the start of a condition

# ---------------------------------------------------------------------------------------------------------
# A. end-state audit: single yes/no questions on one full-resolution frame
Q = {
    "desk_unattended": "Is the desk with the laptop and the black cash counting machine unattended, "
                       "with nobody sitting or standing next to it?",
    "cash_on_desk": "Look at the desk surface. Are there banknotes (paper money) or a bundle of cash lying on the desk?",
    "bag_on_desk": "Look at the desk surface. Is there a bag or package (for example a green cash bag) lying on the desk?",
    "tablet_on_desk": "Look at the desk surface. Is there a tablet computer or a cash box lying on the desk?",
    "laptop_active": "Look at the laptop on the desk. Is its lid open and does the screen show an open program window "
                     "(for example a white document or a spreadsheet)? A closed lid, a dark screen or a "
                     "lock-screen wallpaper does not count.",
    "door_open": "Look at the glass door of the room. Is it standing open, leaving a gap in the glass wall (not shut)?",
    "safe_open": "Look at the grey metal safe. Is its door standing open?",
    "drawer_open": "Is a desk drawer or a cash drawer pulled open?",
}
THRESHOLDS = {
    "desk_unattended": 0.5, "cash_on_desk": 0.5, "bag_on_desk": 0.5, "tablet_on_desk": 0.6,
    "laptop_active": 0.6, "door_open": 0.5, "safe_open": 0.6, "drawer_open": 0.6,
}
ITEM_RU = {"cash_on_desk": "наличные", "bag_on_desk": "сумка/пакет с ценностями", "tablet_on_desk": "планшет/денежный ящик"}

# B. regulation checklist: one question about the whole storyboard; threshold is deliberately high
CHECKLIST = {
    "PHONE_PHOTO": "Does the cashier hold up a mobile phone to take a photo or a video of the money or the desk?",
    "LEAVE_DURING_SERVICE": "Does the cashier walk away from the desk or leave the room while a client is still "
                            "sitting or waiting at the desk?",
    "CASH_TO_OTHER": "Does the cashier hand banknotes or a cash bundle to a person who is not the client sitting "
                     "across the desk (for example to someone who just walked in)?",
    "CASH_CONCEALED": "Does the cashier put banknotes into her pocket, into her clothes or into a personal bag?",
    "SECOND_CLIENT": "While the cashier is serving one client, does another visitor come up to the desk and hand "
                     "something to the cashier or take something from her?",
    "OUTSIDER_CASHIER_SIDE": "Does a client or any other person walk around the desk to the cashier's side, or reach "
                             "into the cashier's work area or the safe?",
    "COLLECTOR_ALONE": "Does a cash collector bring or take a cash bag alone, without any other collector "
                       "accompanying him?",
    "BAG_OPENED_ALONE": "Does the cashier open or empty a cash collection bag when no collector is present in the room?",
}
CHECKLIST_THRESHOLD = 0.7
SAFETY_VERIFY_THRESHOLD = 0.8

RULES = {
    "LEFT_VALUABLES": ("Оставление ценностей без присмотра",
                       "Кассир покидает рабочее место, а на столе остаются {what}. Наличные и ценности должны быть "
                       "убраны в запираемое хранилище.",
                       [("НБРК № 231", "п. 20 ч. 2, п. 22")]),
    "PC_UNLOCKED": ("Компьютер оставлен незаблокированным",
                    "Кассир покидает рабочее место, оставив ноутбук открытым с рабочим экраном: посторонний может "
                    "получить доступ к банковской системе (длительность неактивности на видео не подтверждается, "
                    "фрагмент показывает риск).",
                    [("АРРФР № 53", "п. 71 пп. 9")]),
    "DOOR_OPEN": ("Дверь кассы оставлена открытой",
                  "Дверь помещения кассы остаётся открытой, пока кассир работает.",
                  [("НБРК № 14", "п. 8, п. 16 пп. 1")]),
    "STORAGE_OPEN": ("Сейф или денежный ящик оставлен открытым",
                     "Сейф или денежный ящик остаётся открытым; наличные должны храниться в устройствах, "
                     "закрывающихся на ключ.",
                     [("НБРК № 231", "п. 20 ч. 2")]),
    "PHONE_PHOTO": ("Съёмка денег на мобильный телефон",
                    "Кассир фотографирует или снимает наличные/рабочее место на мобильный телефон.",
                    [("НБРК № 231", "п. 12 ч. 2")]),
    "LEAVE_DURING_SERVICE": ("Уход с рабочего места во время обслуживания",
                             "Кассир покидает рабочее место, не закончив обслуживание клиента.",
                             [("НБРК № 231", "п. 22")]),
    "CASH_TO_OTHER": ("Передача ценностей третьему лицу",
                      "Кассир передаёт наличные человеку, который не является обслуживаемым клиентом.",
                      [("НБРК № 231", "п. 15, п. 14")]),
    "CASH_CONCEALED": ("Сокрытие наличных",
                       "Кассир убирает наличные в карман, одежду или личную сумку.",
                       [("НБРК № 231", "п. 12, п. 14")]),
    "SECOND_CLIENT": ("Допуск второго клиента во время обслуживания",
                      "Во время обслуживания одного клиента к кассиру подходит и обслуживается другой посетитель.",
                      [("НБРК № 14", "п. 2 пп. 9, п. 8")]),
    "OUTSIDER_CASHIER_SIDE": ("Посторонний в рабочей зоне кассира",
                              "Посторонний человек заходит на сторону кассира или тянется в его рабочую зону/к сейфу.",
                              [("НБРК № 14", "п. 8, п. 16 пп. 1")]),
    "COLLECTOR_ALONE": ("Нарушение порядка инкассации: инкассатор без бригады",
                        "Инкассаторскую сумку приносит/забирает один инкассатор, бригада должна быть не менее трёх человек.",
                        [("НБРК № 231", "п. 98")]),
    "BAG_OPENED_ALONE": ("Вскрытие инкассаторской сумки без инкассатора",
                         "Кассир вскрывает инкассаторскую сумку в отсутствие работника инкассации.",
                         [("НБРК № 231", "п. 24 пп. 3, п. 25")]),
    "OTHER": ("Иное возможное нарушение кассовой дисциплины",
              "{what}",
              [("НБРК № 231", "п. 14 (организация работы исключает условия для хищения)")]),
}
DOC_TITLES = {
    "НБРК № 231": "Правила осуществления кассовых операций и операций по инкассации… (постановление Правления НБРК "
                  "от 29.11.2019 № 231), https://adilet.zan.kz/rus/docs/V1900019680",
    "НБРК № 14": "Правила организации охраны и устройства помещений банков… (постановление Правления НБРК "
                 "от 24.02.2020 № 14), https://adilet.zan.kz/rus/docs/V2000020075",
    "АРРФР № 53": "Требования к обеспечению информационной безопасности банков… (постановление Правления АРРФР "
                  "от 03.04.2026 № 53), https://adilet.zan.kz/rus/docs/V2600038336",
}

SAFETY_PROMPT = (
    "These frames are from one CCTV video of a bank cashier's room, in time order; the time in seconds is printed "
    "in the top-left corner of each frame. Is there a moment where the cashier or another person clearly breaks "
    "cash-handling rules or the security of the cash room (for example: money left unattended, money hidden or "
    "handed to a stranger, an outsider at the cashier's side, a safe, drawer or door left open, a phone used to "
    "film money)? Normal work — counting money, serving a seated client, typing — is NOT a violation.\n"
    "If yes, answer exactly in one line:  YES | <seconds> | <short English sentence of what is visible> | "
    "<the same sentence in Russian>\nIf not, answer exactly:  NO")


class YesNo:
    """Yes/no probabilities and short answers from the already loaded Qwen of the base analyzer."""

    def __init__(self, vlm):
        self.model, self.proc = vlm.model, vlm.proc
        tok = self.proc.tokenizer

        def ids(words):
            return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
        self.yes = list(getattr(vlm, "yes", None) or ids(["yes", "Yes", " yes", " Yes", "YES"]))
        self.no = list(getattr(vlm, "no", None) or ids(["no", "No", " no", " No", "NO"]))

    def _inputs(self, images, text):
        import torch
        msgs = [{"role": "user", "content": [*({"type": "image", "image": im} for im in images),
                                             {"type": "text", "text": text}]}]
        kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            inputs = self.proc.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            inputs = self.proc.apply_chat_template(msgs, **kw)
        dev, dt = self.model.device, self.model.dtype
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(dev, dtype=dt) if torch.is_floating_point(v) else v.to(dev)
        return inputs

    def p_yes(self, images, question):
        import torch
        inputs = self._inputs(images, question + " Answer only yes or no.")
        with torch.inference_mode():
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs)
        logits = out.logits[0, -1].float()
        return float(torch.sigmoid(torch.logsumexp(logits[self.yes], 0) - torch.logsumexp(logits[self.no], 0)))

    def generate(self, images, prompt, max_new_tokens=70):
        import torch
        inputs = self._inputs(images, prompt)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------------------------------------
def read_frames(path, step=1.0):
    """Frames every `step` seconds plus the last frame: [(t, bgr)]."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    every = max(1, int(round(fps * step)))
    frames, i = [], 0
    while True:
        if not cap.grab():
            break
        if i % every == 0:
            ok, img = cap.retrieve()
            if ok:
                frames.append((i / fps, img))
        i += 1
    total = i or n
    if frames and total - 1 - int(frames[-1][0] * fps) > every // 2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 2))
        ok, img = cap.read()
        if ok:
            frames.append(((total - 2) / fps, img))
    cap.release()
    return frames


def to_pil(bgr, width=None):
    if width and bgr.shape[1] > width:
        bgr = cv2.resize(bgr, (width, int(round(bgr.shape[0] * width / bgr.shape[1]))), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def stamp(bgr, t):
    img = bgr.copy()
    s = max(1.0, img.shape[1] / 300)
    cv2.rectangle(img, (0, 0), (int(100 * s), int(30 * s)), (0, 0, 0), -1)
    cv2.putText(img, f"t={t:.0f}s", (int(4 * s), int(22 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 255),
                max(1, int(2 * s)))
    return img


def jpeg_b64(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def video_key(path_or_name):
    """'Х-26.mov' / 'X-26_mirror.mov' / 'X-4+X-47' -> ('X-26', 'clean'|'mirror'|...)."""
    name = os.path.basename(str(path_or_name)).split("+")[0]
    m = re.match(r"\s*[XХxх]\s*[-_]?\s*(\d+)(?:_([a-z_]+))?", name)
    if not m:
        return name, "clean"
    variant = (m.group(2) or "clean").rstrip("_")
    return f"X-{int(m.group(1))}", variant


class _Budget(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------
class AnalyzerPlus:
    """Base Analyzer + add-on blocks. Any add-on failure falls back to the base result."""

    def __init__(self, base, blocks=("audit", "checklist", "safety"), disable=(), thresholds=None, disable_tta=True):
        self.base = base
        self.yn = YesNo(base.vlm)
        self.blocks = set(blocks)
        self.disable = set(disable)            # checklist codes to skip, e.g. {"COLLECTOR_ALONE"}
        self.thr = dict(THRESHOLDS, **(thresholds or {}))
        if disable_tta and getattr(base, "tta", False):
            base.tta = False
            print("TTA выключен (он ухудшал результат)")

    def __getattr__(self, name):          # everything else behaves like the base analyzer
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def analyze(self, path, *args, **kwargs):
        res = self.base.analyze(path, *args, **kwargs)
        base_verdict = res.get("verdict")
        try:
            extra = self.extra_checks(path, base_verdict)
        except Exception as e:  # noqa: BLE001 - the add-on must never break the service
            extra = []
            notes = res.get("notes")
            if isinstance(notes, list):
                notes.append(f"addon error: {e}")
        if extra:
            have = {v.get("type") for v in res.get("violations") or []}
            res["violations"] = list(res.get("violations") or []) + [v for v in extra if v["type"] not in have]
            res["verdict"] = "violation"
        return res

    # -----------------------------------------------------------------------------------------------------
    def extra_checks(self, path, base_verdict=None):
        t0 = time.time()
        frames = read_frames(path)
        rec = {"video": os.path.basename(str(path)), "base_verdict": base_verdict, "found": {}, "probs": {}}
        found = []
        if len(frames) >= 4:
            for block, fn in (("audit", self._audit), ("checklist", self._checklist), ("safety", self._safety)):
                if block not in self.blocks:
                    continue
                try:
                    got = fn(frames, t0, rec)
                except _Budget:
                    rec.setdefault("notes", []).append(f"{block}: time budget exceeded")
                    got = []
                rec["found"][block] = [(c, round(t, 1)) for c, t, _ in got]
                found += got
        out, seen = [], []
        for code, t, what in found:
            if any(c == code or (code == "OTHER" and abs(tt - t) < 3) for c, tt in seen):
                continue
            seen.append((code, t))
            out.append(self._violation(frames, code, t, what))
        rec["sec"] = round(time.time() - t0, 1)
        self._log(rec)
        return out

    def _p(self, rec, key, images, question, t0):
        if key not in rec["probs"]:
            if time.time() - t0 > TIME_BUDGET_S:
                raise _Budget()
            rec["probs"][key] = round(self.yn.p_yes(images, question), 3)
        return rec["probs"][key]

    # A ---------------------------------------------------------------------------------------------------
    def _audit(self, frames, t0, rec):
        def yes(i, q):
            return self._p(rec, f"{frames[i][0]:.0f}s:{q}", [to_pil(frames[i][1])], Q[q], t0) > self.thr[q]

        last = len(frames) - 1
        prev = max(0, last - 2)
        back = list(range(last, -1, -2))[:MAX_BACK_STEPS]
        found = []
        absent = lambda i: yes(i, "desk_unattended")  # noqa: E731
        if absent(last) and absent(prev):
            val = lambda i: absent(i) and any(yes(i, q) for q in ITEM_RU)  # noqa: E731
            if val(last) and val(prev):
                what = ", ".join(ITEM_RU[q] for q in ITEM_RU if yes(last, q) or yes(prev, q))
                found.append(("LEFT_VALUABLES", self._start(frames, back, val), what))
            pc = lambda i: absent(i) and yes(i, "laptop_active")  # noqa: E731
            if pc(last) and pc(prev):
                found.append(("PC_UNLOCKED", self._start(frames, back, pc), None))
        else:
            door = lambda i: yes(i, "door_open")  # noqa: E731
            if door(last) and door(prev):
                found.append(("DOOR_OPEN", self._start(frames, back, door), None))
        storage = lambda i: yes(i, "safe_open") or yes(i, "drawer_open")  # noqa: E731
        if storage(last) and storage(prev) and storage(max(0, last - 4)):
            found.append(("STORAGE_OPEN", self._start(frames, back, storage), None))
        return found

    @staticmethod
    def _start(frames, back, cond):
        """Earliest time from which cond holds continuously until the end (midpoint to the first failing sample)."""
        t_ok, t_fail = frames[back[0]][0], None
        for i in back:
            if cond(i):
                t_ok = frames[i][0]
            else:
                t_fail = frames[i][0]
                break
        return (t_ok + t_fail) / 2 if t_fail is not None else t_ok

    # B ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _storyboard(frames):
        n = int(min(12, max(6, round(frames[-1][0] / 3))))
        idx = sorted({round(k * (len(frames) - 1) / (n - 1)) for k in range(n)})
        return idx

    def _checklist(self, frames, t0, rec):
        idx = self._storyboard(frames)
        board = [to_pil(stamp(frames[i][1], frames[i][0]), 336) for i in idx]
        found = []
        for code, q in CHECKLIST.items():
            if code in self.disable:
                continue
            if self._p(rec, f"board:{code}", board, q, t0) <= CHECKLIST_THRESHOLD:
                continue
            # localise: the storyboard frame where the same question is most clearly "yes"
            best = max(idx, key=lambda i: self._p(rec, f"{frames[i][0]:.0f}s:{code}", [to_pil(frames[i][1])], q, t0))
            found.append((code, frames[best][0], None))
        return found

    # C ---------------------------------------------------------------------------------------------------
    def _safety(self, frames, t0, rec):
        if time.time() - t0 > TIME_BUDGET_S:
            raise _Budget()
        idx = self._storyboard(frames)
        board = [to_pil(stamp(frames[i][1], frames[i][0]), 336) for i in idx]
        text = self.yn.generate(board, SAFETY_PROMPT)
        rec["safety_raw"] = text[:300]
        parts = [p.strip() for p in text.split("|")]
        if not parts or not parts[0].upper().startswith("YES") or len(parts) < 3:
            return []
        try:
            t = float(re.findall(r"[\d.]+", parts[1])[0])
        except (IndexError, ValueError):
            return []
        claim_en = parts[2]
        claim_ru = parts[3] if len(parts) > 3 and parts[3] else parts[2]
        near = sorted(range(len(frames)), key=lambda i: abs(frames[i][0] - t))[:3]
        p = self._p(rec, "safety:verify", [to_pil(frames[i][1]) for i in sorted(near)],
                    f"Look at these frames carefully. Is it true that {claim_en.rstrip('.')}?", t0)
        if p <= SAFETY_VERIFY_THRESHOLD:
            return []
        t = min(max(0.0, t), frames[-1][0])
        return [("OTHER", t, f"Возможное нарушение: {claim_ru}")]

    # -----------------------------------------------------------------------------------------------------
    def _violation(self, frames, code, t, what):
        name, desc, refs = RULES[code]
        desc = desc.format(what=what or "")
        basis = "; ".join(f"{DOC_TITLES[d]}, {pt}" for d, pt in refs)
        i_t = min(range(len(frames)), key=lambda k: abs(frames[k][0] - t))
        ev = sorted({i_t, min(len(frames) - 1, i_t + 1)})
        return {
            "type": name,
            "timecode_sec": round(float(t), 1),
            "description": f"{desc} Основание: {basis}.",
            "evidence_frames": [{"timestamp_sec": round(frames[k][0], 1), "image_base64": jpeg_b64(frames[k][1])}
                                for k in ev],
        }

    @staticmethod
    def _log(rec):
        try:
            os.makedirs(os.path.dirname(LOG), exist_ok=True)
            with open(LOG, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------
def ablation(run_csv, log=LOG, variant=None, checklist_threshold=None):
    """Verdict accuracy for every combination of blocks, from one run (no re-run needed).

    run_csv — the CSV written by run() (columns video, true); variant is taken from the file names in the log
    ('clean', 'mirror', ...); by default the variant of the first video in run_csv.
    checklist_threshold — re-score block B with another threshold using the logged probabilities.
    """
    import pandas as pd
    df = pd.read_csv(run_csv)
    recs = {}
    with open(log) as f:
        for line in f:
            r = json.loads(line)
            recs[video_key(r["video"])] = r        # the latest record of each (video, variant) wins
    thr = checklist_threshold
    rows = []
    for v, truth in zip(df["video"], df["true"]):
        key = video_key(v)[0]
        var = variant
        if var is None:
            cands = [k for k in recs if k[0] == key]
            var = cands[-1][1] if cands else "clean"
        r = recs.get((key, var))
        if r is None:
            continue
        found = dict(r.get("found", {}))
        if thr is not None:
            found["checklist"] = [c for c in CHECKLIST if r["probs"].get(f"board:{c}", 0) > thr]
        rows.append({"video": key, "true": bool(truth), "base": r.get("base_verdict") == "violation",
                     **{b: bool(found.get(b)) for b in ("audit", "checklist", "safety")},
                     "hits": "; ".join(f"{b}:{x[0] if isinstance(x, (list, tuple)) else x}"
                                       for b in ("audit", "checklist", "safety") for x in found.get(b, []))})
    t = pd.DataFrame(rows)
    combos = {"база": [], "база + A": ["audit"], "база + B": ["checklist"], "база + C": ["safety"],
              "база + A + B": ["audit", "checklist"], "база + A + C": ["audit", "safety"],
              "база + A + B + C": ["audit", "checklist", "safety"]}
    out = []
    for name, blocks in combos.items():
        pred = t["base"].copy()
        for b in blocks:
            pred |= t[b]
        tp = int((pred & t.true).sum()); fn = int((~pred & t.true).sum())
        fp = int((pred & ~t.true).sum()); tn = int((~pred & ~t.true).sum())
        out.append({"вариант": name, "верно": f"{tp + tn}/{len(t)}", "точность": round((tp + tn) / max(1, len(t)), 3),
                    "найдено": f"{tp}/{tp + fn}", "ложных": f"{fp}/{fp + tn}"})
    res = pd.DataFrame(out).set_index("вариант")
    print(res.to_string())
    print("\nчто нашли блоки (видео с хотя бы одной находкой):")
    print(t[t.hits != ""][["video", "true", "base", "hits"]].to_string(index=False))
    return res


SANITY = [  # (video, second, question, expected) — frames checked by hand
    ("X-28", 12, "laptop_active", True), ("X-27", 24, "laptop_active", False), ("X-29", 15, "laptop_active", False),
    ("X-32", 15, "door_open", True), ("X-33", 12, "door_open", True), ("X-31", 20, "door_open", False),
    ("X-24", 12, "tablet_on_desk", True), ("X-23", 16, "tablet_on_desk", False),
    ("X-26", 15, "cash_on_desk", True), ("X-27", 24, "cash_on_desk", False),
    ("X-15", 18, "desk_unattended", True), ("X-21", 14, "desk_unattended", False),
]


def sanity(plus, files):
    """Ask each audit question on a frame with a known answer; prints P(yes) and OK/ERR."""
    ok = 0
    for vid, t, q, exp in SANITY:
        cap = cv2.VideoCapture(str(files[vid]))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        good, img = cap.read()
        cap.release()
        if not good:
            print(vid, t, q, "cannot read frame")
            continue
        pr = plus.yn.p_yes([to_pil(img)], Q[q])
        hit = (pr > plus.thr[q]) == exp
        ok += hit
        print(f"{vid:5} {t:3}s {q:16} ожидаем={'да ' if exp else 'нет'} P(да)={pr:.2f} {'OK' if hit else 'ОШИБКА'}")
    print(f"итого {ok}/{len(SANITY)}")
