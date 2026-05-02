import torch
import numpy as np
from model import UNet
import torch.nn.functional as F
from PIL import Image as PILImage
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import io
import gradio as gr

# ── Model Loading ─────────────────────────────────────────────────────────────
device = torch.device("cpu")
model = UNet(in_channels=1, out_channels=2)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully")

# ── Helper Functions ──────────────────────────────────────────────────────────
def is_nifti(filepath):
    return filepath.endswith('.nii.gz') or filepath.endswith('.nii') or filepath.endswith('.gz')

def process_image(file_path):
    img = PILImage.open(file_path).convert('L')
    return np.array(img).astype(np.float32)

def process_nifti(file_path, slice_idx):
    volume = nib.load(file_path).get_fdata()
    total_slices = volume.shape[2]
    slice_idx = min(int(slice_idx), total_slices - 1)
    return volume[:, :, slice_idx].astype(np.float32)

def preprocess(ct_slice):
    mn, mx = ct_slice.min(), ct_slice.max()
    if mx > mn:
        ct_slice = (ct_slice - mn) / (mx - mn)
    t = torch.from_numpy(ct_slice).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(256, 256), mode='bilinear', align_corners=False)
    return t

# ── Gradio Event Functions ────────────────────────────────────────────────────
def update_info(file):
    if file is None:
        return gr.update(visible=False), gr.update(visible=False)
    if is_nifti(file.name):
        volume = nib.load(file.name).get_fdata()
        total = volume.shape[2]
        info = (f"✅ CT volume loaded — **{total} slices** (index 0 to {total-1}).\n\n"
                f"Use the slider to pick a slice. Middle slices ({total//3} to {2*total//3}) "
                f"tend to show the most vertebrae content.")
        return gr.update(value=info, visible=True), gr.update(minimum=0, maximum=total-1, value=total//2, visible=True)
    else:
        return gr.update(value="✅ Image loaded — no slice selection needed.", visible=True), gr.update(visible=False)

def predict(file, slice_idx):
    if file is None:
        return None, None, None, "Please upload a file first."
    
    # Load correct input type
    if is_nifti(file.name):
        ct_slice = process_nifti(file.name, slice_idx)
    else:
        ct_slice = process_image(file.name)
    
    # Preprocess and run inference
    tensor = preprocess(ct_slice)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze().numpy()
    
    # Prepare CT display image (resize to 256x256)
    mn, mx = ct_slice.min(), ct_slice.max()
    ct_display = (ct_slice - mn) / (mx - mn)
    ct_pil = PILImage.fromarray((ct_display * 255).astype(np.uint8)).resize((256, 256), PILImage.BILINEAR)
    ct_display_256 = np.array(ct_pil)

    # Prediction mask image
    pred_img = PILImage.fromarray((pred * 255).astype(np.uint8))

    # Overlay — pixel level, guaranteed alignment
    ct_rgb = np.stack([ct_display_256] * 3, axis=-1).copy()
    mask = pred == 1
    ct_rgb[mask, 0] = np.clip(ct_rgb[mask, 0] * 0.4 + 180, 0, 255).astype(np.uint8)
    ct_rgb[mask, 1] = (ct_rgb[mask, 1] * 0.4).astype(np.uint8)
    ct_rgb[mask, 2] = (ct_rgb[mask, 2] * 0.4).astype(np.uint8)
    overlay_img = PILImage.fromarray(ct_rgb.astype(np.uint8))

    # Metrics
    coverage = (pred == 1).sum() / pred.size * 100
    metrics = (
        f"Model Architecture  : U-Net (31M parameters)\n"
        f"Dataset             : VerSe19 — Train: 67 | Val: 37 | Test: 37\n\n"
        f"--- This Prediction ---\n"
        f"Vertebrae Coverage  : {coverage:.2f}%\n"
        f"Vertebrae Pixels    : {(pred == 1).sum()}\n"
        f"Background Pixels   : {(pred == 0).sum()}\n\n"
        f"--- Model Performance (Test Set) ---\n"
        f"Dice Score          : 0.8964 ± 0.0701\n"
        f"IoU Score           : 0.8186\n"
        f"Accuracy            : 98.07%"
    )

    return ct_pil, pred_img, overlay_img, metrics

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Vertebrae Segmentation") as demo:
    gr.Markdown("# 🦴 Vertebrae Segmentation")
    gr.Markdown("Automatic vertebrae detection from CT scans using a U-Net trained on the VerSe19 dataset.")

    gr.Markdown("### Step 1 — Upload your CT scan")
    file_input = gr.File(label="Upload (.nii.gz or PNG/JPEG)",
                         file_types=[".nii.gz", ".nii", ".gz", ".png", ".jpg", ".jpeg"])

    slice_info = gr.Markdown(visible=False)
    
    gr.Markdown("### Step 2 — Select Slice (for .nii.gz only)")
    slice_slider = gr.Slider(minimum=0, maximum=200, value=20, step=1,
                             label="Slice Index", visible=False)

    run_btn = gr.Button("▶ Run Segmentation", variant="primary", size="lg")

    gr.Markdown("### Step 3 — Results")
    ct_out      = gr.Image(label="CT Scan — Original input slice")
    pred_out    = gr.Image(label="Segmentation Mask — White regions = detected vertebrae")
    overlay_out = gr.Image(label="Overlay — Red highlights show vertebrae detected by the model")

    gr.Markdown("### Step 4 — Model Metrics")
    metrics_out = gr.Textbox(label="Results", lines=10)

    file_input.change(fn=update_info, inputs=file_input, outputs=[slice_info, slice_slider])
    run_btn.click(fn=predict, inputs=[file_input, slice_slider],
                  outputs=[ct_out, pred_out, overlay_out, metrics_out])

demo.launch(theme=gr.themes.Soft())