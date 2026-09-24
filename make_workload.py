"""Build the benchmark workload: images + prompts, with SHA-256 manifest.

Photos are COCO val2017 (downloaded by URL, not redistributed); the chart and the
document page are rendered locally (document text = public-domain Pride and
Prejudice). Each item is one independent cluster for paired statistics.
"""
import hashlib, json, textwrap, urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
D = HERE / "data" / "images"
D.mkdir(parents=True, exist_ok=True)

COCO = {
    "coco_cats": "http://images.cocodataset.org/val2017/000000039769.jpg",
    "coco_kitchen": "http://images.cocodataset.org/val2017/000000397133.jpg",
    "coco_street": "http://images.cocodataset.org/val2017/000000252219.jpg",
    "coco_sports": "http://images.cocodataset.org/val2017/000000087038.jpg",
}
for name, url in COCO.items():
    p = D / f"{name}.jpg"
    if not p.exists():
        urllib.request.urlretrieve(url, p)

font = lambda s: ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", s)

# bar chart
img = Image.new("RGB", (800, 500), "white"); d = ImageDraw.Draw(img)
vals = {"Q1": 42, "Q2": 57, "Q3": 35, "Q4": 71, "Q5": 64}
d.text((250, 15), "Quarterly revenue (USD millions)", fill="black", font=font(24))
d.line((80, 440, 760, 440), fill="black", width=2); d.line((80, 60, 80, 440), fill="black", width=2)
for i, (k, v) in enumerate(vals.items()):
    x = 120 + i * 125; y = 440 - v * 5
    d.rectangle((x, y, x + 70, 440), fill=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][i])
    d.text((x + 20, y - 28), str(v), fill="black", font=font(20)); d.text((x + 18, 450), k, fill="black", font=font(20))
img.save(D / "chart_bars.png")

# document page (public-domain text)
book = (HERE / "data" / "book.txt").read_text(encoding="utf-8")
start = book.find("Chapter I.")
para = " ".join(book[start : start + 1400].split())
img = Image.new("RGB", (900, 1100), "white"); d = ImageDraw.Draw(img)
y = 40
for line in textwrap.wrap(para, 70):
    d.text((50, y), line, fill="black", font=font(22)); y += 32
img.save(D / "doc_page.png")

ITEMS = [
    ("coco_cats", "Describe this image in detail."),
    ("coco_kitchen", "Describe this image in detail."),
    ("coco_street", "What is happening in this image? Answer in a few sentences."),
    ("coco_sports", "Describe the scene, the people, and what they are doing."),
    ("chart_bars", "Read every value from this chart and explain the overall trend."),
    ("doc_page", "Transcribe all the text in this image exactly."),
]
workload = []
for name, prompt in ITEMS:
    p = next(D.glob(f"{name}.*"))
    workload.append(dict(id=name, image=str(p.relative_to(HERE)), prompt=prompt,
                         sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(HERE / "data" / "workload.json").write_text(json.dumps(workload, indent=1))
print(json.dumps(workload, indent=1))
