"""CashGuard API for the jury and the website: base Analyzer + add-on block A (end-state audit), no TTA.

Contract (task section 10):
  POST /api/v1/analyze        multipart field "file" (.mov), header X-API-Key  -> {"job_id": "..."}
  GET  /api/v1/jobs/{job_id}  header X-API-Key -> {"job_id", "status": processing|done|failed, "verdict",
                                                    "violations": [{type, timecode_sec, description, evidence_frames}]}
  GET  /health                -> {"status": "ok", "model_loaded": true}

Run on the pod (all notebook kernels shut down, GPU free):
  cd /workspace && CASHGUARD_API_KEY=<key> nohup python -m uvicorn serve_plus:app --host 0.0.0.0 --port 8000 \
      > /workspace/serve_plus.log 2>&1 &
Works offline: models come from /workspace (weights) and /workspace/hf_cache (Qwen).
"""
import os
import sys
import tempfile
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

WORK = os.environ.get("WORK", "/workspace")
os.environ.setdefault("HF_HOME", os.path.join(WORK, "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # no internet needed (and not used) for processing
for p in (WORK, os.path.join(WORK, "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

import base64  # noqa: E402

import cv2  # noqa: E402
from fastapi import FastAPI, File, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

API_KEY = os.environ.get("CASHGUARD_API_KEY", "")
UPLOAD_DIR = os.path.join(WORK, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="CashGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS = {}
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1)   # one video at a time: Qwen fills the GPU, others wait in the queue
ANALYZER = None


def build_analyzer():
    """Exactly the configuration evaluated in 11_tta: 85% clean / 79% mirror with block A."""
    from src.pipeline import Analyzer
    import addon
    base = Analyzer(person_weights=os.path.join(WORK, "yolo11m.pt"),
                    pose_weights=os.path.join(WORK, "yolo11m-pose.pt"),
                    object_weights=os.path.join(WORK, "models", "cash_yolo11m_auto.pt"),
                    device="0", vlm_model_id="Qwen/Qwen3.5-9B", tta=False)
    return addon.AnalyzerPlus(base, blocks=("audit",))


@app.on_event("startup")
def _startup():
    global ANALYZER
    if not API_KEY:
        raise RuntimeError("задайте CASHGUARD_API_KEY")
    ANALYZER = build_analyzer()
    print("CashGuard: модель загружена", flush=True)


def _check_key(key):
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


def _frame_b64(path, t):
    """JPEG of the frame at t seconds (sequential read: seeking is unreliable for some .mov files)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target, i, img = int(round(t * fps)), 0, None
    while True:
        if not cap.grab():
            break
        if i >= target:
            ok, img = cap.retrieve()
            break
        i += 1
    cap.release()
    if img is None:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _normalize(res, path, job_id):
    """Base result -> exactly the jury schema; every violation gets 1-3 evidence frames."""
    viols = []
    for v in res.get("violations") or []:
        try:
            t = float(v.get("timecode_sec") or 0.0)
        except (TypeError, ValueError):
            t = 0.0
        ev = [{"timestamp_sec": round(float(e.get("timestamp_sec", t)), 1), "image_base64": e["image_base64"]}
              for e in (v.get("evidence_frames") or []) if e.get("image_base64")]
        if not ev:
            b = _frame_b64(path, t)
            ev = [{"timestamp_sec": round(t, 1), "image_base64": b}] if b else []
        viols.append({"type": v.get("type") or "Возможное нарушение кассовой дисциплины",
                      "timecode_sec": round(t, 1),
                      "description": v.get("description") or "",
                      "evidence_frames": ev[:3]})
    verdict = "violation" if viols or res.get("verdict") == "violation" else "no_violation"
    return {"job_id": job_id, "status": "done", "verdict": verdict, "violations": viols}


def _work(job_id, path):
    try:
        res = ANALYZER.analyze(path, job_id=job_id)
        out = _normalize(res, path, job_id)
    except Exception as e:  # noqa: BLE001 - report the failure to the client
        traceback.print_exc()
        out = {"job_id": job_id, "status": "failed", "error": str(e)}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    with JOBS_LOCK:
        JOBS[job_id] = out


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str = Header(None)):
    _check_key(x_api_key)
    job_id = uuid.uuid4().hex[:12]
    suffix = os.path.splitext(file.filename or "")[1] or ".mov"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
    with os.fdopen(fd, "wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    with JOBS_LOCK:
        JOBS[job_id] = {"job_id": job_id, "status": "processing"}
    POOL.submit(_work, job_id, path)
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}")
def job(job_id: str, x_api_key: str = Header(None)):
    _check_key(x_api_key)
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="job not found")
        return JOBS[job_id]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ANALYZER is not None}
