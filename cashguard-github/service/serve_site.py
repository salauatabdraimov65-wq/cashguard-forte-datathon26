"""Proven API (serve_plus: queue, fallback to base result, verified norms) + the team's website on one address.

Put this file into /workspace/forte-release/ next to forte_site.py and site-dist/. Run from /workspace:
  CASHGUARD_API_KEY="$(cat /workspace/api_key.txt)" nohup python -m uvicorn serve_site:app \
      --app-dir /workspace/forte-release --host 0.0.0.0 --port 8000 --workers 1 > /workspace/serve_site.log 2>&1 &
"""
import os
import sys

WORK = os.environ.get("WORK", "/workspace")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (WORK, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import forte_site  # noqa: E402
import serve_plus  # noqa: E402

forte_site.install_frontend(serve_plus.app, os.path.join(HERE, "site-dist"))
app = serve_plus.app
