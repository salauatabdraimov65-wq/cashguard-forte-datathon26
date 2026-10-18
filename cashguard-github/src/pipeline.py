"""analyze(video) -> result JSON in the format required by the task (section 10 of the brief).

Pipeline: sample frames -> YOLO11m people + BoT-SORT tracking, fine-tuned object detector
-> rule engine (src/rules.py) -> evidence frames (JPEG base64) + rule references (rules_norms.json).
"""
import base64
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.rules import VIOLATION_TYPES, detect_events, verdict

CLASSES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
           "id_document", "phone", "handbag", "cup"]
NORMS_PATH = Path(__file__).with_name("rules_norms.json")
HIGHLIGHT = {"cash", "collection_bag", "safe_open", "handbag", "cup", "phone"}


class Analyzer:
    def __init__(self, person_weights="yolo11m.pt", object_weights=None, device="0", fps=4.0, imgsz=896,
                 threshold=0.5, vlm_model_id=None, max_vlm_calls=30, pose_weights="yolo11m-pose.pt", time_budget_s=180,
                 tta=False):
        from ultralytics import YOLO
        self.person_model = YOLO(person_weights)
        self.pose_model = YOLO(pose_weights) if pose_weights else None
        self.obj_model = YOLO(str(object_weights)) if object_weights and Path(object_weights).exists() else None
        self.device, self.fps, self.imgsz, self.threshold = device, fps, imgsz, threshold
        self.norms = json.loads(NORMS_PATH.read_text()) if NORMS_PATH.exists() else {}
        self.vlm, self.max_vlm_calls, self._max_vlm_calls_cfg = None, max_vlm_calls, max_vlm_calls
        self.time_budget_s = time_budget_s
        self.tta = tta  # flip + multi-scale test-time augmentation for people/objects (not pose)
        if vlm_model_id:
            from src.vlm_checks import QwenYesNo
            self.vlm = QwenYesNo(vlm_model_id, device="cuda" if str(device).isdigit() else device)

    def verify(self, path, events):
        """Drop rule events that the VLM does not confirm (only checks with a proven threshold)."""
        from src.vlm_checks import QUESTIONS2, VERIFY_THRESHOLDS, verify_question
        kept = []
        for e in events:
            q = verify_question(e)
            if q and e.score >= self.threshold:
                imgs = [im for im in (self.grab(path, t) for t in (e.t - 0.5, e.t + 0.5, e.t + 1.5)) if im is not None]
                for im in imgs:
                    im.thumbnail((640, 640))
                if imgs and self.vlm.p_yes(imgs, QUESTIONS2[q]) < VERIFY_THRESHOLDS[q]:
                    continue
            kept.append(e)
        return kept

    def vlm_events(self, path, P, O, H):
        """Ask the VLM the outsider question on moments with >= 2 people in the room."""
        from src.vlm_checks import QUESTIONS, SERVICE_THRESHOLDS, candidates, vlm_events
        cands = [c for c in candidates(P, O, pd.DataFrame(), H, fps_sample=self.fps) if c["q"] == "outsider"]
        if len(cands) > self.max_vlm_calls:  # bound the processing time on long videos
            cands = [cands[i] for i in np.linspace(0, len(cands) - 1, self.max_vlm_calls).round().astype(int)]
        answers = []
        for c in cands:
            imgs = [im for im in (self.grab(path, t) for t in c["frames"]) if im is not None]
            for im in imgs:
                im.thumbnail((640, 640))
            if imgs:
                answers.append({"q": c["q"], "t": c["t"], "p_yes": self.vlm.p_yes(imgs, QUESTIONS[c["q"]])})
        return vlm_events(answers, SERVICE_THRESHOLDS)

    # ---- detection
    def timelines(self, path):
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        step = max(1, round(fps / self.fps))
        common = dict(stream=True, vid_stride=step, device=self.device, verbose=False, imgsz=self.imgsz)
        persons, objects = [], []
        for k, r in enumerate(self.person_model.track(source=str(path), classes=[0], conf=0.3,
                                                      tracker="botsort.yaml", augment=self.tta, **common)):
            t = k * step / fps
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [-1] * len(r.boxes)
            for tid, b, c in zip(ids, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                persons.append({"t": t, "track": tid, "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "conf": float(c)})
        if self.obj_model is not None:
            for k, r in enumerate(self.obj_model.predict(source=str(path), conf=0.25, augment=self.tta, **common)):
                t = k * step / fps
                for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    objects.append({"t": t, "cls": CLASSES[int(c)], "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                                    "conf": float(s)})
        pose = []
        if self.pose_model is not None:
            for k, r in enumerate(self.pose_model.predict(source=str(path), conf=0.3, **common)):
                if r.keypoints is None:
                    continue
                t = k * step / fps
                for b, kp in zip(r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()):
                    row = {"t": t, "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
                    for name, i in (("l_wrist", 9), ("r_wrist", 10)):
                        row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_c"] = kp[i]
                    pose.append(row)
        P = pd.DataFrame(persons, columns=["t", "track", "x0", "y0", "x1", "y1", "conf"])
        O = pd.DataFrame(objects, columns=["t", "cls", "x0", "y0", "x1", "y1", "conf"])
        self.last_pose = pd.DataFrame(pose)
        return P, O, H, fps

    # ---- evidence
    @staticmethod
    def grab(path, t):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
        ok, frame = cap.read()
        cap.release()
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None

    @staticmethod
    def annotate(img, O, t, tol=0.3):
        d = ImageDraw.Draw(img)
        near = O[(O.t - t).abs() <= tol] if len(O) else O
        for r in near.itertuples():
            if r.cls in HIGHLIGHT:
                d.rectangle([r.x0, r.y0, r.x1, r.y1], outline="red", width=3)
                d.text((r.x0 + 3, max(r.y0 - 12, 0)), r.cls, fill="red")
        d.text((6, 6), f"t = {t:.1f} s", fill="yellow")
        return img

    @staticmethod
    def to_b64(img, quality=85):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()

    # ---- main entry
    def analyze(self, path, job_id=None):
        t0 = time.time()
        P, O, H, fps = self.timelines(path)          # if the video cannot be read at all -> exception -> "failed"
        events = detect_events(P, O, H, fps_sample=self.fps, pose=getattr(self, "last_pose", None))
        notes = []
        if self.vlm is not None:
            # time budget: the jury limit is 5 min per video; the VLM is optional, rules always answer
            left = self.time_budget_s - (time.time() - t0)
            if left > 30:
                try:
                    self.max_vlm_calls = max(5, min(self._max_vlm_calls_cfg, int((left - 20) / 2.0)))
                    events = self.verify(path, events) + self.vlm_events(path, P, O, H)
                except Exception as e:  # never fail the whole job because of the VLM
                    notes.append(f"vlm skipped: {type(e).__name__}")
            else:
                notes.append("vlm skipped: time budget")
        v, kept = verdict(events, self.threshold)
        violations = []
        for e in kept:
            frames = []
            for dt in (-1.0, 0.0, 1.0):  # before / during / after
                img = self.grab(path, e.t + dt)
                if img is not None:
                    frames.append({"timestamp_sec": round(e.t + dt, 2),
                                   "image_base64": self.to_b64(self.annotate(img, O, e.t + dt))})
            violations.append({
                "type": VIOLATION_TYPES.get(e.type, e.type),
                "timecode_sec": round(float(e.t), 2),
                "description": e.description,
                "rule": self.norms.get(e.type, {}),
                "confidence": round(float(e.score), 2),
                "evidence_frames": frames,
            })
        return {"job_id": job_id, "status": "done", "verdict": v, "violations": violations,
                "processing_sec": round(time.time() - t0, 1), "notes": notes}
