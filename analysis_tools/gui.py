import io
import base64
from pathlib import Path

import torch
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import folium
import branca.colormap as cm
import s2sphere
from transformers import AutoImageProcessor

# ---- modules ----
from modules import (
    GeoWebDataset,
    build_s2_index_maps,
    build_parent_tables_from_maps,
    HierarchicalConvNeXt,
)

# ---------------- CONFIG ----------------
DATASET_PATH = Path("/home/austen/GeoDataset/dataset_sharded")
CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "checkpoint_1235.pt"
PRETRAINED_MODEL_ID = "facebook/convnext-base-384"
S2_LEVELS = list(range(3, 8))
LEVEL_TO_SHOW = 6
RESIZE = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def s2_id_to_latlon(s2_id):
    ll = s2sphere.CellId(int(s2_id)).to_lat_lng()
    return float(ll.lat().degrees), float(ll.lng().degrees)

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


# --------------- INITIALIZE MODEL SUPPORT STUFF ---------------
print("Initializing processor / dataset / S2 index...")

s2_labels_dir = DATASET_PATH / "s2_labels"
idx2id, id2idx, _ = build_s2_index_maps(s2_labels_dir, S2_LEVELS)

processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_ID, use_fast=True)
if RESIZE > 0:
    processor.do_resize = True
    processor.size = {"shortest_edge": RESIZE}
else:
    processor.do_resize = False

dataset = GeoWebDataset(
    DATASET_PATH,
    processor,
    levels=S2_LEVELS,
    shuffle=False,
    num_shards_limit=None,
    id2idx=id2idx,
)

parent_table = build_parent_tables_from_maps(idx2id, id2idx, S2_LEVELS)
PARENTS = {k: v.to(DEVICE) for k, v in parent_table.items()}
NUM_CLASSES_PER_LEVEL = dataset.num_classes_list

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
        num_classes=NUM_CLASSES_PER_LEVEL[-1],
        freeze=False,
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _current_model = model
    _current_checkpoint = checkpoint_path
    return model


# --------------- INFERENCE ON A PIL IMAGE ---------------
def run_inference_on_pil(pil_img: Image.Image, checkpoint_path: Path):
    model = load_model(checkpoint_path)

    # Preprocess with same processor as dataset
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
        level=LEVEL_TO_SHOW,
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
        <h2>GeoModel GUI</h2>
        <p>1. Choose checkpoint<br>2. Click paste area<br>3. Ctrl+V an image</p>

        <label for="checkpoint-select">Checkpoint:</label>
        <select id="checkpoint-select">
            {% for ckpt in checkpoints %}
            <option value="{{ ckpt }}" {% if ckpt == default_ckpt %}selected{% endif %}>{{ ckpt }}</option>
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
        map_html=None,
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

        m = run_inference_on_pil(pil_img, ckpt_path)
        map_html = m.get_root().render()
        return jsonify({"map_html": map_html})
    except Exception as e:
        # This will also show up in your Flask logs
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Example: http://127.0.0.1:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
