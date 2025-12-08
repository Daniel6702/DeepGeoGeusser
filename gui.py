from pathlib import Path

import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import folium
import branca.colormap as cm
import s2sphere
from transformers import AutoImageProcessor

from modules import HierarchicalConvNeXt

# ---------------- CONFIG ----------------
CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "checkpoint.pt"
PRETRAINED_MODEL_ID = "facebook/convnext-base-384"

LEVEL_TO_SHOW = 6
RESIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

S2_METADATA_PATH = Path("checkpoints/s2_metadata.pt")

# --------------- HELPER FNS ---------------
def probs_by_level(probs_fine, levels, parents, num_classes_per_level):
    fine_level = levels[-1]
    out = {fine_level: probs_fine}
    p = probs_fine.unsqueeze(0)  # (1, num_fine)
    for i, coarse in enumerate(levels[:-1]):
        parent = parents[(fine_level, coarse)]
        n = num_classes_per_level[i]
        pc = torch.zeros(1, n, device=probs_fine.device)
        pc.scatter_add_(1, parent.unsqueeze(0).expand(1, -1), p)
        out[coarse] = pc[0]
    return out


def s2_cell_corners(s2_id):
    cell = s2sphere.Cell(s2sphere.CellId(int(s2_id)))
    corners = []
    for i in range(4):
        v = cell.get_vertex(i)
        ll = s2sphere.LatLng.from_point(v)
        corners.append((float(ll.lat().degrees), float(ll.lng().degrees)))
    return corners


def make_prob_map(level, probs_per_level, idx2id):
    probs = probs_per_level[level].detach().cpu()
    ids = idx2id[level]
    m = folium.Map(location=[0, 0], zoom_start=2, tiles="cartodbpositron")
    cmap = cm.linear.YlOrRd_09.scale(float(probs.min()), float(probs.max()))

    lats, lons = [], []
    for idx, p in enumerate(probs):
        p = float(p)
        s2_id = ids[idx]
        corners = s2_cell_corners(s2_id)
        lats.extend([lat for lat, _ in corners])
        lons.extend([lon for _, lon in corners])

        color = cmap(p)
        folium.Polygon(
            locations=corners,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=f"{s2_id} p={p:.3f}",
        ).add_to(m)

    if lats:
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
    cmap.add_to(m)
    return m


# --------------- LOAD PRECOMPUTED S2 METADATA ---------------
print("Loading S2 metadata and initializing image processor...")

meta = torch.load(S2_METADATA_PATH, map_location="cpu")
S2_LEVELS = meta["S2_LEVELS"]
idx2id = meta["idx2id"]
NUM_CLASSES_PER_LEVEL = meta["num_classes_per_level"]
parent_table = meta["parent_table"]
PARENTS = {k: v.to(DEVICE) for k, v in parent_table.items()}

# (Optional sanity check)
assert LEVEL_TO_SHOW in S2_LEVELS, "LEVEL_TO_SHOW must be one of S2_LEVELS"

# Image processor
processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_ID, use_fast=True)
processor.do_resize = True
processor.size = {"shortest_edge": RESIZE}

# --------------- MODEL LOADING WITH SIMPLE CACHE ---------------
_current_model = None
_current_checkpoint = None


def load_model(checkpoint_path: Path):
    global _current_model, _current_checkpoint
    checkpoint_path = checkpoint_path.resolve()
    if (
        _current_model is not None
        and _current_checkpoint is not None
        and _current_checkpoint == checkpoint_path
    ):
        return _current_model

    print(f"Loading model from {checkpoint_path} ...")
    model = HierarchicalConvNeXt(
        pretrained_name=PRETRAINED_MODEL_ID,
        num_classes=NUM_CLASSES_PER_LEVEL[-1],  # fine level (S7)
        freeze=False,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _current_model = model
    _current_checkpoint = checkpoint_path
    return model


def run_inference_on_pil(
    pil_img: Image.Image, checkpoint_path: Path, level_to_show: int = LEVEL_TO_SHOW
):
    model = load_model(checkpoint_path)

    inputs = processor(pil_img, return_tensors="pt")
    pixel_vals = inputs["pixel_values"].to(DEVICE, memory_format=torch.channels_last)

    with torch.no_grad():
        logits = model(pixel_vals)
        probs_fine = torch.softmax(logits, dim=1)[0]

    probs_per_level = probs_by_level(
        probs_fine,
        S2_LEVELS,
        PARENTS,
        NUM_CLASSES_PER_LEVEL,
    )

    m = make_prob_map(
        level=level_to_show,
        probs_per_level=probs_per_level,
        idx2id=idx2id,
    )
    return m


# --------------- FLASK APP ---------------
app = Flask(__name__)

# simple template: checkpoint selector, pasted-image preview, map
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>GeoModel GUI</title>
    <style>
        body { font-family: sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; }
        #left { width: 35%; padding: 1rem; box-sizing: border-box; border-right: 1px solid #ccc; }
        #right { flex: 1; padding: 1rem; box-sizing: border-box; }
        #paste-area {
            border: 2px dashed #999;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-top: 1rem;
            color: #666;
        }
        #paste-area.highlight {
            border-color: #333;
            color: #000;
        }
        #preview-img { max-width: 100%; margin-top: 1rem; }
        label { font-weight: bold; display: block; margin-top: 0.5rem; }
        select {
            width: 100%; padding: 0.3rem; box-sizing: border-box; margin-top: 0.25rem;
        }
        #map-frame {
            width: 100%;
            height: 100%;
            border: none;
        }
    </style>
</head>
<body>
    <div id="left">
        <h2>DeepGeoGeusser GUI</h2>
        
        <label for="checkpoint-select">Checkpoint:</label>
        <select id="checkpoint-select">
            {% for ckpt in checkpoints %}
            <option value="{{ ckpt }}" {% if ckpt == default_ckpt %}selected{% endif %}>{{ ckpt }}</option>
            {% endfor %}
        </select>

        <label for="level-select">S2 level:</label>
        <select id="level-select">
            {% for lvl in s2_levels %}
            <option value="{{ lvl }}" {% if lvl == default_level %}selected{% endif %}>Level {{ lvl }}</option>
            {% endfor %}
        </select>

        <label>Paste image here:</label>
        <div id="paste-area" tabindex="0">
            Click here and press Ctrl+V to paste an image
        </div>

        <img id="preview-img" src="" alt="Pasted image will appear here">
    </div>

    <div id="right">
        <iframe id="map-frame"
            srcdoc="<p style='font-family:sans-serif;'>Paste an image to see the map.</p>">
        </iframe>
    </div>

<script>
    const pasteArea = document.getElementById("paste-area");
    const previewImg = document.getElementById("preview-img");
    const checkpointSelect = document.getElementById("checkpoint-select");
    const levelSelect = document.getElementById("level-select");
    const mapFrame = document.getElementById("map-frame");

    pasteArea.addEventListener("focus", () => {
        pasteArea.classList.add("highlight");
    });
    pasteArea.addEventListener("blur", () => {
        pasteArea.classList.remove("highlight");
    });

    window.addEventListener("paste", async (event) => {
        if (document.activeElement !== pasteArea) {
            return; // only when paste area has focus
        }

        const items = event.clipboardData.items;
        let imageFile = null;

        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf("image") === 0) {
                imageFile = items[i].getAsFile();
                break;
            }
        }

        if (!imageFile) {
            alert("No image data found in clipboard!");
            return;
        }

        // Local preview
        const imgURL = URL.createObjectURL(imageFile);
        previewImg.src = imgURL;

        // Send to backend
        const formData = new FormData();
        formData.append("image", imageFile, "pasted.png");
        formData.append("checkpoint", checkpointSelect.value);
        formData.append("level", levelSelect.value);

        try {
            const resp = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                alert("Error running model (HTTP " + resp.status + ")");
                return;
            }

            const data = await resp.json();
            // IMPORTANT: use iframe srcdoc so scripts run
            mapFrame.srcdoc = data.map_html;

        } catch (err) {
            console.error("Error calling /predict:", err);
            alert("Error calling /predict, see console for details.");
        }
    });
</script>
</body>
</html>
"""


def list_checkpoints():
    if not CHECKPOINT_DIR.exists():
        return [str(DEFAULT_CHECKPOINT)]
    return [str(p) for p in sorted(CHECKPOINT_DIR.glob("*.pt"))]


@app.route("/", methods=["GET"])
def index():
    checkpoints = list_checkpoints()
    return render_template_string(
        HTML_TEMPLATE,
        checkpoints=checkpoints,
        default_ckpt=str(DEFAULT_CHECKPOINT.resolve()),
        s2_levels=S2_LEVELS,
        default_level=LEVEL_TO_SHOW,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "no image"}), 400

        image_file = request.files["image"]
        pil_img = Image.open(image_file.stream).convert("RGB")

        ckpt_path_str = request.form.get("checkpoint", str(DEFAULT_CHECKPOINT))
        ckpt_path = Path(ckpt_path_str)

        level_str = request.form.get("level", str(LEVEL_TO_SHOW))
        try:
            level = int(level_str)
        except ValueError:
            level = LEVEL_TO_SHOW
        if level not in S2_LEVELS:
            level = LEVEL_TO_SHOW

        m = run_inference_on_pil(pil_img, ckpt_path, level_to_show=level)
        map_html = m.get_root().render()
        return jsonify({"map_html": map_html})
    except Exception as e:
        # This will also show up in your Flask logs
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Example: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
