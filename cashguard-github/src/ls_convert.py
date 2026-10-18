"""Label Studio JSON export -> COCO dict (only tasks that were actually reviewed and saved).

Tasks without a saved annotation are skipped: an unreviewed frame must not become a
"no objects here" training example.
"""
import json
from pathlib import Path

CLASSES = ["cash", "collection_bag", "safe_open", "safe_closed", "counting_machine",
           "id_document", "phone", "handbag", "cup"]


def ls_to_coco(tasks):
    if isinstance(tasks, (str, Path)):
        tasks = json.loads(Path(tasks).read_text())
    coco = {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)]}
    cat_id = {c: i + 1 for i, c in enumerate(CLASSES)}
    ann_id = 0
    for task in tasks:
        done = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not done:
            continue
        ann = max(done, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        data = task["data"]
        W, H = data.get("width"), data.get("height")
        img_id = len(coco["images"]) + 1
        coco["images"].append({"id": img_id, "file_name": data["fname"], "width": W, "height": H,
                               "reviewed_empty": not ann["result"]})
        for r in ann["result"]:
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            W, H = r.get("original_width", W), r.get("original_height", H)
            for lbl in v["rectanglelabels"]:
                if lbl not in cat_id:
                    continue
                ann_id += 1
                x, y, w, h = v["x"] * W / 100, v["y"] * H / 100, v["width"] * W / 100, v["height"] * H / 100
                coco["annotations"].append({"id": ann_id, "image_id": img_id, "category_id": cat_id[lbl],
                                            "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
    return coco
