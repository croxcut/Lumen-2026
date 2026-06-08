"""
LUMEN — LungMaxViT GUI  v3
Hybrid Explainable Deep Learning System for Pulmonary Abnormality Detection
Run: python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import csv
from datetime import datetime
from PIL import Image, ImageTk
import numpy as np

try:
    import torch
    from torchvision import transforms
    from model import LungMaxViTHybrid
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    IMPORT_ERROR    = str(e)

CHECKPOINT = r"best_model_finetuned.pth"
OUTPUT_DIR = "lumen_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_KEYS  = ['ILD', 'Lung Cancer', 'Normal', 'Tuberculosis', 'COVID', 'Pneumonia']
LABELS      = [
    'Pulmonary Fibrosis/ILD',
    'Lung Cancer/Nodules',
    'Normal',
    'Tuberculosis',
    'COVID-19',
    'Pneumonia',
]
NUM_LABELS  = 6
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "predictions.csv")

THRESHOLDS = {
    'ILD': 0.65, 'Lung Cancer': 0.28, 'Normal': 0.81,
    'Tuberculosis': 0.88, 'COVID': 0.50, 'Pneumonia': 0.68,
}

MODEL_METRICS = {
    'Pulmonary Fibrosis/ILD':  {'f1': '0.9919', 'auc': '0.9999'},
    'Lung Cancer/Nodules':     {'f1': '0.9804', 'auc': '0.9996'},
    'Normal':                  {'f1': '0.9257', 'auc': '0.9960'},
    'Tuberculosis':            {'f1': '1.0000', 'auc': '1.0000'},
    'COVID-19':                {'f1': '0.9543', 'auc': '0.9974'},
    'Pneumonia':               {'f1': '0.9949', 'auc': '1.0000'},
}

DISEASE_DESCRIPTIONS = {
    'Pulmonary Fibrosis/ILD': (
        "Interstitial lung disease (ILD) encompasses a group of disorders causing progressive "
        "scarring of lung tissue. The scarring reduces the lungs' ability to transfer oxygen into "
        "the bloodstream. Common causes include prolonged exposure to hazardous materials, "
        "autoimmune conditions, and certain medications. Symptoms include progressive breathlessness "
        "and a dry, persistent cough. Early detection is critical as fibrosis is largely irreversible."
    ),
    'Lung Cancer/Nodules': (
        "Lung cancer is the leading cause of cancer-related mortality worldwide. Pulmonary nodules "
        "are small masses of tissue in the lung that may be benign or malignant. Risk factors include "
        "smoking, radon exposure, asbestos, and genetic predisposition. Early-stage detection "
        "significantly improves prognosis — five-year survival rates exceed 60% when caught at Stage I "
        "compared to under 10% at Stage IV. CT screening is recommended for high-risk individuals."
    ),
    'Normal': (
        "No significant pulmonary abnormalities are detected in this scan. The lung parenchyma, "
        "airways, and pleural spaces appear within normal radiological limits. While this finding "
        "is reassuring, it should be interpreted alongside clinical presentation, patient history, "
        "and any relevant symptoms. Routine follow-up per clinical guidelines is still advised "
        "for high-risk populations."
    ),
    'Tuberculosis': (
        "Tuberculosis (TB) is a contagious bacterial infection caused by Mycobacterium tuberculosis, "
        "primarily affecting the lungs. It remains one of the top infectious disease killers globally. "
        "Radiological features include upper-lobe infiltrates, cavitations, and hilar lymphadenopathy. "
        "Active TB requires a multi-drug antibiotic regimen for 6–9 months. Prompt identification "
        "and isolation are essential to prevent community spread."
    ),
    'COVID-19': (
        "COVID-19 pneumonia caused by SARS-CoV-2 presents with characteristic bilateral ground-glass "
        "opacities and consolidations on CT, predominantly in peripheral and lower-lobe distributions. "
        "Severity ranges from mild respiratory symptoms to acute respiratory distress syndrome (ARDS). "
        "CT findings can persist weeks after clinical recovery. Imaging plays a key role in assessing "
        "disease extent and monitoring progression or resolution."
    ),
    'Pneumonia': (
        "Pneumonia is an acute infection of the lung parenchyma caused by bacteria, viruses, or fungi. "
        "It presents radiologically as lobar or segmental consolidation, air bronchograms, and "
        "increased opacity. Common pathogens include Streptococcus pneumoniae, influenza viruses, and "
        "in immunocompromised patients, Pneumocystis jirovecii. Treatment depends on causative organism, "
        "severity, and patient comorbidities. Rapid diagnosis is essential to initiate appropriate therapy."
    ),
}

LABEL_ICONS = {
    'Pulmonary Fibrosis/ILD': '🫧',
    'Lung Cancer/Nodules':    '🔴',
    'Normal':                 '✅',
    'Tuberculosis':           '🦠',
    'COVID-19':               '🌐',
    'Pneumonia':              '💧',
}

BG_MAIN      = "#FFFFFF"
BG_PANEL     = "#F4FBF6"
BG_CARD      = "#FFFFFF"
BG_CARD_ALT  = "#F0F9F4"
GREEN_DARK   = "#1B6B3A"
GREEN_MED    = "#2E9E5B"
GREEN_LIGHT  = "#4CAF50"
GREEN_PALE   = "#E8F5EC"
GREEN_ACCENT = "#00C853"
ACCENT_RED   = "#D32F2F"
ACCENT_AMBER = "#F57C00"
BORDER_LIGHT = "#E0F0E8"
BORDER_MED   = "#B2D8BE"
TEXT_DARK    = "#1A2E22"
TEXT_MED     = "#3D5944"
TEXT_LIGHT   = "#7A9E84"
WHITE        = "#FFFFFF"

FONT_BRAND   = ("Segoe UI", 22, "bold")
FONT_TITLE   = ("Segoe UI", 16, "bold")
FONT_HEAD    = ("Segoe UI", 13, "bold")
FONT_SUBHEAD = ("Segoe UI", 11, "bold")
FONT_BODY    = ("Segoe UI", 10)
FONT_SMALL   = ("Segoe UI", 9)
FONT_TINY    = ("Segoe UI", 8)
FONT_MONO    = ("Consolas", 10)
FONT_MONO_LG = ("Consolas", 13, "bold")

TOOL_BTN_WIDTH = 90

model_instance = None

def load_model():
    global model_instance
    if not MODEL_AVAILABLE:
        return False
    try:
        model_instance = LungMaxViTHybrid(num_labels=NUM_LABELS)
        ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        model_instance.load_state_dict(ckpt['model_state_dict'])
        model_instance.eval()
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False

transform = None
if MODEL_AVAILABLE:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def preprocess_for_inference(image_path):
    img = Image.open(image_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    arr = np.clip(arr, p2, p98)
    if p98 - p2 > 0:
        arr = (arr - p2) / (p98 - p2) * 255.0
    return Image.fromarray(arr.astype(np.uint8)).convert('RGB')

def is_ct_scan(image_path):
    try:
        raw = Image.open(image_path).convert('RGB')
        arr = np.array(raw, dtype=np.float32)
        r, g, b  = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        avg_diff = (np.mean(np.abs(r - g)) + np.mean(np.abs(r - b)) + np.mean(np.abs(g - b))) / 3
        if avg_diff >= 15:
            return False
        processed = preprocess_for_inference(image_path)
        gray      = np.array(processed.convert('L'), dtype=np.float32)
        if (gray > 10).sum() / gray.size < 0.02:
            return False
        if gray.std() < 20:
            return False
        return True
    except Exception:
        return False

def predict(image_path):
    if model_instance is None:
        raise RuntimeError("Model not loaded")
    image  = preprocess_for_inference(image_path)
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_instance(tensor)
        probs   = torch.sigmoid(outputs).squeeze().tolist()
    results = {}
    for label, key, prob in zip(LABELS, LABEL_KEYS, probs):
        results[label] = {
            'prob':     round(prob, 4),
            'detected': prob >= THRESHOLDS.get(key, 0.5),
            'key':      key,
        }
    return results

def save_prediction(image_path, results):
    row = {'filename': os.path.basename(image_path),
           'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    for label, data in results.items():
        row[label] = data['prob']
    file_exists = os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def run_gradcam(image_path, output_path):
    if not MODEL_AVAILABLE:
        return False
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        for param in model_instance.parameters():
            param.requires_grad = True

        image   = Image.open(image_path).convert('RGB')
        rgb_img = np.array(image.resize((224, 224))) / 255.0
        tensor  = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model_instance(tensor)
            probs   = torch.sigmoid(outputs).squeeze().tolist()

        target_layer = [model_instance.backbone.stem.conv2]

        fig = plt.figure(figsize=(22, 8))
        fig.patch.set_facecolor(BG_MAIN)

        ax_orig = fig.add_subplot(1, NUM_LABELS + 1, 1)
        ax_orig.imshow(rgb_img, cmap='gray')
        ax_orig.set_title('Original\nScan', fontsize=11, fontweight='bold',
                          color=GREEN_DARK, pad=8)
        ax_orig.axis('off')

        for i, (label, key) in enumerate(zip(LABELS, LABEL_KEYS)):
            prob     = probs[i]
            detected = prob >= THRESHOLDS.get(key, 0.5)
            cam      = GradCAMPlusPlus(model=model_instance, target_layers=target_layer)
            gcam     = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(i)])[0]
            viz      = show_cam_on_image(rgb_img.astype(np.float32), gcam, use_rgb=True)
            ax = fig.add_subplot(1, NUM_LABELS + 1, i + 2)
            ax.imshow(viz)
            ax.set_title(f"{label}\n{prob:.1%} {'✓' if detected else '✗'}",
                         fontsize=9, fontweight='bold',
                         color=GREEN_LIGHT if detected else TEXT_LIGHT, pad=6)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(GREEN_LIGHT if detected else BORDER_MED)
                spine.set_linewidth(2.5 if detected else 1)
                spine.set_visible(True)

        plt.suptitle(f'Attention Map — {os.path.basename(image_path)}',
                     fontsize=13, fontweight='bold', color=GREEN_DARK, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=BG_MAIN)
        plt.close()
        return True
    except Exception as e:
        print(f"GradCAM error: {e}")
        return False

def run_shap(image_path, output_path):
    if not MODEL_AVAILABLE:
        return False
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        image   = Image.open(image_path).convert('RGB').resize((224, 224))
        rgb_img = np.array(image) / 255.0
        tensor  = transform(image).unsqueeze(0)

        model_instance.eval()
        with torch.no_grad():
            outputs = model_instance(tensor)
            probs   = torch.sigmoid(outputs).squeeze().tolist()

        background  = torch.zeros_like(tensor)
        explainer   = shap.GradientExplainer(model_instance, background)
        shap_values = explainer.shap_values(tensor)

        shap_cmap = LinearSegmentedColormap.from_list(
            'lumen_shap', ['#FFFFFF', '#C8E6C9', '#4CAF50', '#F57C00', '#D32F2F'], N=256)

        fig = plt.figure(figsize=(NUM_LABELS * 3.6, 8))
        fig.patch.set_facecolor(BG_MAIN)

        for i, (label, key) in enumerate(zip(LABELS, LABEL_KEYS)):
            prob     = probs[i]
            detected = prob >= THRESHOLDS.get(key, 0.5)
            fg_color = GREEN_LIGHT if detected else TEXT_LIGHT
            border_c = GREEN_LIGHT if detected else BORDER_MED

            ax_orig = fig.add_subplot(2, NUM_LABELS, i + 1)
            ax_orig.imshow(rgb_img, cmap='gray')
            ax_orig.set_title(f"{label}\n{prob:.1%}  {'✓' if detected else '✗'}",
                              fontsize=8.5, fontweight='bold', color=fg_color, pad=5)
            ax_orig.axis('off')
            for spine in ax_orig.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_c)
                spine.set_linewidth(2.0 if detected else 0.8)

            ax_shap = fig.add_subplot(2, NUM_LABELS, NUM_LABELS + i + 1)
            sv           = shap_values[i][0]
            sv           = np.transpose(sv, (1, 2, 0))
            sv_magnitude = np.abs(sv).mean(axis=2)
            sv_magnitude = (sv_magnitude - sv_magnitude.min()) / (sv_magnitude.max() - sv_magnitude.min() + 1e-8)

            im = ax_shap.imshow(sv_magnitude, cmap=shap_cmap, vmin=0, vmax=1)
            ax_shap.set_title('Influence Map', fontsize=7.5, color=TEXT_MED, pad=4)
            ax_shap.axis('off')
            cbar = plt.colorbar(im, ax=ax_shap, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6, colors=TEXT_MED)
            cbar.outline.set_edgecolor(BORDER_LIGHT)
            cbar.set_ticks([0, 0.5, 1.0])
            cbar.set_ticklabels(['Low', 'Mid', 'High'])

        plt.suptitle(f'Feature Influence Analysis  ·  {os.path.basename(image_path)}',
                     fontsize=12, fontweight='bold', color=TEXT_DARK, y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor=BG_MAIN, edgecolor='none')
        plt.close()
        return True
    except Exception as e:
        print(f"SHAP error: {e}")
        import traceback; traceback.print_exc()
        return False

def run_bbox(image_path, output_path, cam_threshold=0.45):
    if not MODEL_AVAILABLE:
        return False, None, {}
    try:
        from bbox_detector import detect_boxes, draw_boxes_on_image, render_bbox_figure
        import matplotlib
        matplotlib.use('Agg')

        for param in model_instance.parameters():
            param.requires_grad = True

        image     = preprocess_for_inference(image_path)
        rgb_img   = np.array(image.resize((224, 224))) / 255.0
        img_uint8 = (rgb_img * 255).astype(np.uint8)
        tensor    = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model_instance(tensor)
            probs   = torch.sigmoid(outputs).squeeze().tolist()

        detection_results = detect_boxes(
            model_instance, tensor, probs,
            orig_size=(224, 224), cam_threshold=cam_threshold,
            image_rgb_uint8=img_uint8)

        fig = render_bbox_figure(rgb_img.astype(np.float32), detection_results)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_MAIN)
        import matplotlib.pyplot as plt; plt.close(fig)

        annotated = draw_boxes_on_image(img_uint8, detection_results, line_thickness=2)
        return True, annotated, detection_results
    except Exception as e:
        print(f"BBox error: {e}")
        import traceback; traceback.print_exc()
        return False, None, {}


class ProgressBar(tk.Canvas):
    def __init__(self, parent, width=200, height=5, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=BORDER_LIGHT, highlightthickness=0, **kwargs)
        self._val   = 0
        self._bar_w = width
        self._bar_h = height
        self.after_idle(self._draw)

    def set_value(self, val):
        self._val = max(0.0, min(1.0, val))
        self._draw()

    def _draw(self):
        try:
            self.delete('all')
        except tk.TclError:
            return
        w, h = self._bar_w, self._bar_h
        self.create_rectangle(0, 0, w, h, fill=BORDER_LIGHT, outline='')
        fill_w = int(w * self._val)
        if fill_w > 2:
            color = GREEN_LIGHT if self._val >= 0.5 else (
                    ACCENT_AMBER if self._val >= 0.25 else ACCENT_RED)
            self.create_rectangle(0, 0, fill_w, h, fill=color, outline='')

class ScanThumbnail(tk.Canvas):
    def __init__(self, parent, size=270, **kwargs):
        super().__init__(parent, width=size, height=size,
                         bg=BG_CARD_ALT, highlightthickness=0, **kwargs)
        self._size  = size
        self._photo = None
        self.after_idle(self._draw_empty)

    def _draw_empty(self):
        try:
            self.delete('all')
        except tk.TclError:
            return
        s = self._size
        self.create_rectangle(2, 2, s-2, s-2, outline=BORDER_MED, width=1, dash=(4, 4))
        self.create_text(s//2, s//2 - 14, text="🩻", font=("Segoe UI", 28),
                         fill=TEXT_LIGHT)
        self.create_text(s//2, s//2 + 22, text="No scan uploaded",
                         fill=TEXT_LIGHT, font=FONT_SMALL)

    def set_image(self, pil_img):
        s     = self._size
        img   = pil_img.resize((s, s), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photo = photo
        self.delete('all')
        self.create_image(0, 0, anchor='nw', image=photo)
        self.create_rectangle(2, 2, s-2, s-2, outline=GREEN_MED, width=2)

    def clear(self):
        self._photo = None
        self._draw_empty()


class FlatButton(tk.Frame):
    def __init__(self, parent, text, command=None, primary=False,
                 width=180, height=38, danger=False):
        try:
            parent_bg = parent.cget('bg')
        except Exception:
            parent_bg = BG_MAIN
        super().__init__(parent, bg=parent_bg, width=width, height=height, cursor='hand2')
        self.pack_propagate(False)
        self._cmd     = command
        self._primary = primary
        self._danger  = danger
        self._hover   = False
        self._enabled = True
        self._lbl = tk.Label(self, text=text, bg=WHITE, fg=TEXT_MED,
                             font=FONT_BODY, cursor='hand2')
        self._lbl.place(relx=0.5, rely=0.5, anchor='center')
        self._refresh()
        for w in (self, self._lbl):
            w.bind('<Enter>',           self._on_enter)
            w.bind('<Leave>',           self._on_leave)
            w.bind('<ButtonPress-1>',   self._on_press)
            w.bind('<ButtonRelease-1>', self._on_release)

    def _colors(self, pressed=False):
        if not self._enabled:
            return BG_CARD_ALT, BORDER_LIGHT, TEXT_LIGHT, FONT_BODY
        if self._primary:
            bg = GREEN_MED if (self._hover and not pressed) else GREEN_DARK
            return bg, GREEN_DARK, WHITE, ("Segoe UI", 10, "bold")
        if self._danger:
            return ("#FFEBEE" if self._hover else WHITE), ACCENT_RED, ACCENT_RED, FONT_BODY
        if self._hover:
            return GREEN_PALE, GREEN_MED, GREEN_DARK, FONT_BODY
        return WHITE, BORDER_MED, TEXT_MED, FONT_BODY

    def _refresh(self, pressed=False):
        bg, border, fg, font = self._colors(pressed)
        self.configure(bg=bg, highlightbackground=border,
                       highlightthickness=1, highlightcolor=border)
        self._lbl.configure(bg=bg, fg=fg, font=font)

    def configure_text(self, text):
        self._lbl.configure(text=text)

    def set_enabled(self, enabled):
        self._enabled = enabled
        self._hover   = False
        self._refresh()
        cur = 'hand2' if enabled else ''
        self.configure(cursor=cur)
        self._lbl.configure(cursor=cur)

    def _on_enter(self, e):
        if self._enabled:
            self._hover = True
            self._refresh()

    def _on_leave(self, e):
        self._hover = False
        self._refresh()

    def _on_press(self, e):
        if self._enabled:
            self._refresh(pressed=True)

    def _on_release(self, e):
        self._refresh()
        if self._enabled and self._cmd:
            self._cmd()


class ToolCard(tk.Frame):
    def __init__(self, parent, icon, title, desc, command, bg=BG_CARD, **kwargs):
        super().__init__(parent, bg=BORDER_LIGHT, padx=1, pady=1, **kwargs)
        inner = tk.Frame(self, bg=bg)
        inner.pack(fill='both', expand=True)

        left = tk.Frame(inner, bg=GREEN_PALE, width=48)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)
        tk.Label(left, text=icon, font=("Segoe UI", 16),
                 bg=GREEN_PALE, fg=GREEN_DARK).place(relx=0.5, rely=0.5, anchor='center')

        body = tk.Frame(inner, bg=bg)
        body.pack(side='left', fill='both', expand=True, padx=12, pady=10)
        tk.Label(body, text=title, font=FONT_SUBHEAD, bg=bg, fg=TEXT_DARK).pack(anchor='w')
        tk.Label(body, text=desc,  font=FONT_TINY,    bg=bg, fg=TEXT_LIGHT).pack(anchor='w', pady=(1, 0))

        self.btn = FlatButton(inner, text="Run", command=command,
                              width=TOOL_BTN_WIDTH, height=36)
        self.btn.pack(side='right', padx=12, pady=10)
        self.btn.set_enabled(False)

    def set_enabled(self, v):
        self.btn.set_enabled(v)

    def set_running(self, running):
        self.btn.configure_text("⏳" if running else "Run")
        self.btn.set_enabled(not running)


class LUMENApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LUMEN — Lung Scan Analysis System")
        self.geometry("1440x920")
        self.minsize(1200, 780)
        self.configure(bg=BG_MAIN)

        self.image_path   = None
        self.results      = None
        self.gradcam_path = None
        self.shap_path    = None
        self.bbox_path    = None
        self.model_loaded = False
        self.ct_valid     = False
        self._scan_count  = 0

        self._style_ttk()
        self._build_ui()
        self._load_model_async()

    def _style_ttk(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('G.Vertical.TScrollbar',
                    background=GREEN_PALE, troughcolor=BG_PANEL,
                    bordercolor=BORDER_LIGHT, arrowcolor=GREEN_MED, relief='flat')

    def _card(self, parent, bg=BG_CARD, pad=0):
        outer = tk.Frame(parent, bg=BORDER_LIGHT, padx=1, pady=1)
        inner = tk.Frame(outer, bg=bg)
        inner.pack(fill='both', expand=True, padx=pad, pady=pad)
        return outer, inner

    def _section(self, parent, text, top=12):
        f = tk.Frame(parent, bg=BG_MAIN)
        f.pack(fill='x', pady=(top, 6))
        tk.Frame(f, bg=GREEN_MED, width=4, height=14).pack(side='left')
        tk.Label(f, text=f"  {text}", font=FONT_SUBHEAD,
                 bg=BG_MAIN, fg=GREEN_DARK).pack(side='left')

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER_LIGHT, height=1).pack(fill='x')

    def _build_ui(self):
        self._build_topbar()
        body = tk.Frame(self, bg=BG_PANEL)
        body.pack(fill='both', expand=True)

        left = tk.Frame(body, bg=WHITE, width=300)
        left.pack(side='left', fill='y')
        left.pack_propagate(False)
        tk.Frame(body, bg=BORDER_LIGHT, width=1).pack(side='left', fill='y')

        mid = tk.Frame(body, bg=BG_PANEL, width=360)
        mid.pack(side='left', fill='y', padx=0)
        mid.pack_propagate(False)
        tk.Frame(body, bg=BORDER_LIGHT, width=1).pack(side='left', fill='y')

        right = tk.Frame(body, bg=WHITE)
        right.pack(side='left', fill='both', expand=True)

        self._build_left(left)
        self._build_mid(mid)
        self._build_right(right)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=GREEN_DARK, height=56)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        logo_f = tk.Frame(bar, bg=GREEN_DARK)
        logo_f.pack(side='left', padx=20, pady=10)
        tk.Label(logo_f, text="⊕", font=("Segoe UI", 18, "bold"),
                 bg=GREEN_DARK, fg=GREEN_ACCENT).pack(side='left', padx=(0, 8))
        tk.Label(logo_f, text="LUMEN", font=FONT_BRAND,
                 bg=GREEN_DARK, fg=WHITE).pack(side='left')
        tk.Label(logo_f, text="  Lung Scan Analysis", font=("Segoe UI", 10),
                 bg=GREEN_DARK, fg="#A5D6B0").pack(side='left', padx=(4, 0))

        right_f = tk.Frame(bar, bg=GREEN_DARK)
        right_f.pack(side='right', padx=20)

        self._scan_counter_var = tk.StringVar(value="0 scans")
        tk.Label(right_f, textvariable=self._scan_counter_var,
                 font=FONT_SMALL, bg=GREEN_DARK, fg="#A5D6B0").pack(side='right', padx=(16, 0))

        self.status_dot   = tk.Label(right_f, text="●", font=("Segoe UI", 9),
                                      bg=GREEN_DARK, fg=ACCENT_AMBER)
        self.status_label = tk.Label(right_f, text="Loading model…",
                                      font=FONT_SMALL, bg=GREEN_DARK, fg="#A5D6B0")
        self.status_dot.pack(side='right', padx=(0, 4))
        self.status_label.pack(side='right')

    def _build_left(self, parent):
        hdr = tk.Frame(parent, bg=GREEN_PALE, height=48)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Upload Scan", font=FONT_SUBHEAD,
                 bg=GREEN_PALE, fg=GREEN_DARK).pack(side='left', padx=16, pady=14)

        wrap = tk.Frame(parent, bg=WHITE)
        wrap.pack(fill='both', expand=True, padx=16, pady=12)

        self.thumb_canvas = ScanThumbnail(wrap, size=268)
        self.thumb_canvas.pack(pady=(0, 10))

        self.ct_badge_var = tk.StringVar(value="")
        self.ct_badge     = tk.Label(wrap, textvariable=self.ct_badge_var,
                                      font=FONT_SMALL, bg=WHITE, fg=TEXT_MED)
        self.ct_badge.pack(pady=(0, 2))

        self.fname_var = tk.StringVar(value="No file selected")
        tk.Label(wrap, textvariable=self.fname_var, font=FONT_TINY,
                 bg=WHITE, fg=TEXT_LIGHT, wraplength=260).pack(pady=(0, 10))

        self.browse_btn = FlatButton(wrap, text="📂  Browse Scan", command=self._browse_file,
                                      width=268, height=38)
        self.browse_btn.pack(pady=(0, 6))

        self.analyze_btn = FlatButton(wrap, text="▶  Run Analysis", command=self._analyze,
                                       primary=True, width=268, height=42)
        self.analyze_btn.set_enabled(False)
        self.analyze_btn.pack(pady=(0, 10))

        self.progress_var = tk.StringVar(value="")
        self.progress_lbl = tk.Label(wrap, textvariable=self.progress_var,
                                      font=FONT_SMALL, bg=WHITE, fg=GREEN_MED, wraplength=260)
        self.progress_lbl.pack(pady=(0, 10))

        self._divider(wrap)

        tk.Label(wrap, text="Model Info", font=FONT_SUBHEAD,
                 bg=WHITE, fg=GREEN_DARK).pack(anchor='w', pady=(10, 6))
        for key, val in [("Model", "LungMaxViT Hybrid"), ("AUC", "0.9988"),
                          ("Conditions", "6 types"), ("Input", "224×224 px")]:
            row = tk.Frame(wrap, bg=WHITE)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=key, font=FONT_TINY, bg=WHITE,
                     fg=TEXT_LIGHT, width=12, anchor='w').pack(side='left')
            tk.Label(row, text=val, font=FONT_TINY, bg=WHITE, fg=TEXT_DARK).pack(side='left')

        bottom = tk.Frame(parent, bg=WHITE)
        bottom.pack(fill='x', side='bottom', padx=16, pady=12)
        self.reset_btn = FlatButton(bottom, text="↺  New Analysis", command=self._reset,
                                     width=268, height=34)
        self.reset_btn.pack()

    def _build_mid(self, parent):
        hdr = tk.Frame(parent, bg=GREEN_PALE, height=48)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Analysis Tools", font=FONT_SUBHEAD,
                 bg=GREEN_PALE, fg=GREEN_DARK).pack(side='left', padx=16, pady=14)

        wrap = tk.Frame(parent, bg=BG_PANEL)
        wrap.pack(fill='both', expand=True, padx=12, pady=12)

        self._tool_cards = {}
        tools = [
            ('🔍', "Attention Map",    "Grad-CAM++ visualization",      self._show_gradcam, 'gradcam'),
            ('[ ]', "Region Detection", "Locate abnormal regions",        self._show_bbox,    'bbox'),
            ('📊', "SHAP Analysis",    "Feature influence heatmaps",     self._show_shap,    'shap'),
        ]
        for icon, title, desc, cmd, key in tools:
            card = ToolCard(wrap, icon=icon, title=title, desc=desc, command=cmd, bg=BG_CARD)
            card.pack(fill='x', pady=(0, 8))
            self._tool_cards[key] = card

        self._divider(wrap)

        tk.Label(wrap, text="Result Preview", font=FONT_SUBHEAD,
                 bg=BG_PANEL, fg=GREEN_DARK).pack(anchor='w', pady=(10, 6))

        self._preview_title_var = tk.StringVar(value="")
        tk.Label(wrap, textvariable=self._preview_title_var,
                 font=FONT_TINY, bg=BG_PANEL, fg=GREEN_MED).pack(anchor='w')

        self._viz_preview_label = tk.Label(wrap, bg=BG_CARD_ALT,
                                            text="Run a tool to\npreview results here",
                                            font=FONT_SMALL, fg=TEXT_LIGHT,
                                            width=32, height=10)
        self._viz_preview_label.pack(pady=(4, 10), fill='x')

        self._divider(wrap)

        self.export_btn = FlatButton(wrap, text="💾  Export CSV", command=self._export_csv,
                                      width=336, height=34)
        self.export_btn.set_enabled(False)
        self.export_btn.pack(pady=(10, 0))

    def _build_right(self, parent):
        hdr = tk.Frame(parent, bg=WHITE, height=72)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        self._divider(hdr)
        inner_hdr = tk.Frame(hdr, bg=WHITE)
        inner_hdr.pack(fill='both', expand=True, padx=24, pady=12)

        self.results_header = tk.Label(inner_hdr, text="Ready to analyze",
                                        font=FONT_TITLE, bg=WHITE, fg=TEXT_LIGHT)
        self.results_header.pack(side='left')
        self.results_sub = tk.Label(inner_hdr, text="",
                                     font=FONT_SMALL, bg=WHITE, fg=TEXT_LIGHT)
        self.results_sub.pack(side='left', padx=(12, 0), pady=(4, 0))

        self._divider(parent)

        wrap = tk.Frame(parent, bg=WHITE)
        wrap.pack(fill='both', expand=True)

        self.results_canvas = tk.Canvas(wrap, bg=WHITE, highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient='vertical',
                           command=self.results_canvas.yview,
                           style='G.Vertical.TScrollbar')
        self.results_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.results_canvas.pack(side='left', fill='both', expand=True)

        self.results_inner = tk.Frame(self.results_canvas, bg=WHITE)
        self._cwin = self.results_canvas.create_window((0, 0), window=self.results_inner, anchor='nw')

        self.results_inner.bind('<Configure>', lambda e: self.results_canvas.configure(
            scrollregion=self.results_canvas.bbox('all')))
        self.results_canvas.bind('<Configure>', lambda e: self.results_canvas.itemconfig(
            self._cwin, width=e.width))
        self.results_canvas.bind('<MouseWheel>', lambda e: self.results_canvas.yview_scroll(
            int(-1 * (e.delta / 120)), "units"))

        self._show_placeholder()

    def _show_placeholder(self):
        for w in self.results_inner.winfo_children():
            w.destroy()

        pad = tk.Frame(self.results_inner, bg=WHITE)
        pad.pack(fill='x', padx=24, pady=24)

        outer, inner = self._card(pad, bg=BG_CARD_ALT)
        outer.pack(fill='x', pady=(0, 20))
        tk.Label(inner, text="🫁", font=("Segoe UI", 40),
                 bg=BG_CARD_ALT).pack(pady=(20, 8))
        tk.Label(inner, text="No Analysis Yet",
                 font=FONT_HEAD, bg=BG_CARD_ALT, fg=TEXT_DARK).pack()
        tk.Label(inner, text="Upload a chest CT scan and press Run Analysis",
                 font=FONT_SMALL, bg=BG_CARD_ALT, fg=TEXT_LIGHT).pack(pady=(4, 20))

        tk.Label(pad, text="Detectable Conditions",
                 font=FONT_SUBHEAD, bg=WHITE, fg=GREEN_DARK).pack(anchor='w', pady=(0, 10))
        grid = tk.Frame(pad, bg=WHITE)
        grid.pack(fill='x')
        for col in range(3):
            grid.columnconfigure(col, weight=1)
        for i, label in enumerate(LABELS):
            m   = MODEL_METRICS.get(label, {})
            c   = i % 3
            r   = i // 3
            outer, inner = self._card(grid, bg=BG_CARD)
            outer.grid(row=r, column=c, padx=5, pady=5, sticky='ew')
            icon = LABEL_ICONS.get(label, '◆')
            top = tk.Frame(inner, bg=BG_CARD)
            top.pack(fill='x', padx=10, pady=(10, 4))
            tk.Label(top, text=icon, font=("Segoe UI", 18),
                     bg=BG_CARD).pack(side='left')
            tk.Label(top, text=f"AUC {m.get('auc','—')}",
                     font=FONT_TINY, bg=BG_CARD, fg=GREEN_MED).pack(side='right', pady=(6, 0))
            tk.Label(inner, text=label, font=FONT_SMALL, bg=BG_CARD,
                     fg=TEXT_DARK, wraplength=160, justify='left').pack(anchor='w', padx=10, pady=(0, 10))

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Chest CT Scan",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("All files", "*.*")])
        if path:
            self._set_image(path)

    def _set_image(self, path):
        self.image_path   = path
        self.results      = None
        self.gradcam_path = None
        self.shap_path    = None
        self.bbox_path    = None

        try:
            img = Image.open(path).convert('RGB')
            self.thumb_canvas.set_image(img)
        except Exception:
            self.thumb_canvas.clear()

        self.fname_var.set(os.path.basename(path))

        try:
            self.ct_valid = is_ct_scan(path)
        except Exception:
            self.ct_valid = False

        if self.ct_valid:
            self.ct_badge_var.set("✓ Valid CT scan")
            self.ct_badge.configure(fg=GREEN_MED)
            if self.model_loaded:
                self.analyze_btn.set_enabled(True)
        else:
            self.ct_badge_var.set("⚠  May not be a CT scan")
            self.ct_badge.configure(fg=ACCENT_AMBER)
            if self.model_loaded:
                self.analyze_btn.set_enabled(True)

        self.results_header.configure(text=os.path.basename(path), fg=TEXT_DARK)
        self.results_sub.configure(text="Press Run Analysis to begin", fg=TEXT_LIGHT)
        self._show_placeholder()

        for card in self._tool_cards.values():
            card.set_enabled(False)
        self.export_btn.set_enabled(False)
        self.progress_var.set("")

    def _analyze(self):
        if not self.image_path:
            return
        self.analyze_btn.set_enabled(False)
        self.analyze_btn.configure_text("⏳  Analyzing…")
        self.progress_var.set("Running inference…")
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        try:
            results = predict(self.image_path)
            self.results = results
            save_prediction(self.image_path, results)
            self.after(0, self._show_results)
        except Exception as e:
            self.after(0, lambda: self._show_error(str(e)))

    def _show_results(self):
        self.analyze_btn.set_enabled(True)
        self.analyze_btn.configure_text("▶  Run Analysis")
        self.progress_var.set("✓ Analysis complete")
        self._scan_count += 1
        self._scan_counter_var.set(f"{self._scan_count} scan{'s' if self._scan_count > 1 else ''}")

        for card in self._tool_cards.values():
            card.set_enabled(True)
        self.export_btn.set_enabled(True)

        fname = os.path.basename(self.image_path)
        self.results_header.configure(text="Analysis Complete", fg=GREEN_DARK)
        self.results_sub.configure(text=fname, fg=TEXT_MED)

        for w in self.results_inner.winfo_children():
            w.destroy()

        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['prob'], reverse=True)
        detected_list  = [(l, d) for l, d in sorted_results if d['detected']]
        other_list     = [(l, d) for l, d in sorted_results if not d['detected']]

        pad = tk.Frame(self.results_inner, bg=WHITE)
        pad.pack(fill='x', padx=24, pady=(20, 0))

        outer, inner = self._card(pad, bg=GREEN_PALE if detected_list else BG_CARD_ALT)
        outer.pack(fill='x', pady=(0, 16))

        top_row = tk.Frame(inner, bg=inner['bg'])
        top_row.pack(fill='x', padx=16, pady=(12, 6))
        label_text = f"{len(detected_list)} finding{'s' if len(detected_list) != 1 else ''} detected" \
                     if detected_list else "No significant findings"
        tk.Label(top_row, text=label_text, font=FONT_SUBHEAD,
                 bg=inner['bg'], fg=GREEN_DARK if detected_list else TEXT_MED).pack(side='left')
        tk.Label(top_row, text=datetime.now().strftime('%H:%M:%S'),
                 font=FONT_TINY, bg=inner['bg'], fg=TEXT_LIGHT).pack(side='right')

        if detected_list:
            pills = tk.Frame(inner, bg=inner['bg'])
            pills.pack(anchor='w', padx=16, pady=(0, 10))
            for label, data in detected_list:
                icon = LABEL_ICONS.get(label, '◆')
                pill = tk.Frame(pills, bg=WHITE, highlightbackground=GREEN_MED,
                                highlightthickness=1)
                pill.pack(side='left', padx=(0, 6), pady=2)
                tk.Label(pill, text=f" {icon} {label} ", font=FONT_TINY,
                         bg=WHITE, fg=GREEN_DARK, padx=6, pady=3).pack()
        else:
            tk.Label(inner, text="All markers below detection thresholds",
                     font=FONT_SMALL, bg=inner['bg'], fg=TEXT_LIGHT).pack(anchor='w', padx=16, pady=(0, 12))

        if detected_list:
            tk.Label(pad, text="Detected", font=FONT_SUBHEAD,
                     bg=WHITE, fg=GREEN_DARK).pack(anchor='w', pady=(0, 8))
            for label, data in detected_list:
                self._make_result_card(pad, label, data['prob'], detected=True)

        tk.Label(pad, text="All Scores", font=FONT_SUBHEAD,
                 bg=WHITE, fg=TEXT_LIGHT).pack(anchor='w', pady=(16, 8))
        for label, data in sorted_results:
            self._make_mini_row(pad, label, data['prob'], data['detected'])

    def _make_result_card(self, parent, label, prob, detected):
        outer, inner = self._card(parent, bg=BG_CARD)
        outer.pack(fill='x', pady=(0, 8))

        tk.Frame(inner, bg=GREEN_MED if detected else BORDER_LIGHT, width=5).pack(side='left', fill='y')

        body = tk.Frame(inner, bg=BG_CARD)
        body.pack(side='left', fill='both', expand=True, padx=14, pady=12)

        top = tk.Frame(body, bg=BG_CARD)
        top.pack(fill='x')

        icon = LABEL_ICONS.get(label, '◆')
        tk.Label(top, text=icon, font=("Segoe UI", 18), bg=BG_CARD).pack(side='left', padx=(0, 10))

        info = tk.Frame(top, bg=BG_CARD)
        info.pack(side='left', fill='x', expand=True)
        tk.Label(info, text=label, font=FONT_SUBHEAD, bg=BG_CARD, fg=TEXT_DARK).pack(anchor='w')
        m = MODEL_METRICS.get(label, {})
        tk.Label(info, text=f"F1 {m.get('f1','—')}  ·  AUC {m.get('auc','—')}",
                 font=FONT_TINY, bg=BG_CARD, fg=TEXT_LIGHT).pack(anchor='w')

        right_col = tk.Frame(top, bg=BG_CARD)
        right_col.pack(side='right', padx=(10, 0))
        tk.Label(right_col, text=f"{prob:.1%}", font=FONT_MONO_LG,
                 bg=BG_CARD, fg=GREEN_DARK if detected else TEXT_LIGHT).pack(anchor='e')
        badge_text = "POSITIVE" if detected else "NEGATIVE"
        badge_bg   = GREEN_PALE if detected else BG_CARD_ALT
        badge_fg   = GREEN_DARK if detected else TEXT_LIGHT
        badge = tk.Label(right_col, text=f" {badge_text} ", font=FONT_TINY,
                         bg=badge_bg, fg=badge_fg, padx=4, pady=2)
        badge.pack(anchor='e', pady=(2, 0))

        bar_f = tk.Frame(body, bg=BG_CARD)
        bar_f.pack(fill='x', pady=(8, 0))
        bar = ProgressBar(bar_f, height=5)
        bar.pack(fill='x')
        bar.set_value(prob)

        desc_text = DISEASE_DESCRIPTIONS.get(label, "")
        if desc_text:
            desc_container = tk.Frame(body, bg=BG_CARD_ALT,
                                      highlightbackground=BORDER_MED,
                                      highlightthickness=1)
            desc_container.pack(fill='x', pady=(10, 0))
            header_row = tk.Frame(desc_container, bg=GREEN_PALE)
            header_row.pack(fill='x')
            tk.Label(header_row, text="ℹ  Clinical Overview",
                     font=FONT_TINY, bg=GREEN_PALE, fg=GREEN_DARK,
                     padx=10, pady=5).pack(side='left')
            tk.Label(desc_container, text=desc_text,
                     font=FONT_TINY, bg=BG_CARD_ALT, fg=TEXT_MED,
                     wraplength=520, justify='left',
                     padx=10, pady=8).pack(anchor='w')

    def _make_mini_row(self, parent, label, prob, detected):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill='x', pady=2)

        icon = LABEL_ICONS.get(label, '◆')
        fg   = GREEN_DARK if detected else TEXT_LIGHT

        tk.Label(row, text=icon, font=("Segoe UI", 11), bg=BG_CARD,
                 width=2).pack(side='left', padx=(10, 4), pady=8)
        tk.Label(row, text=label, font=FONT_SMALL, bg=BG_CARD,
                 fg=TEXT_MED if not detected else TEXT_DARK,
                 width=24, anchor='w').pack(side='left')

        bar = ProgressBar(row, width=130, height=4)
        bar.pack(side='left', padx=8)
        bar.set_value(prob)

        tk.Label(row, text=f"{prob:.1%}", font=FONT_MONO,
                 bg=BG_CARD, fg=fg, width=6).pack(side='left')
        dot = "●" if detected else "○"
        tk.Label(row, text=dot, font=FONT_TINY, bg=BG_CARD,
                 fg=fg).pack(side='left', padx=(4, 10))

    def _run_tool(self, key, label, fn):
        if not self.image_path:
            return
        card = self._tool_cards[key]
        card.set_running(True)
        self.progress_var.set(f"Running {label}…")

        def thread():
            try:
                fn()
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: [card.set_running(False),
                                       self.progress_var.set("✓ Done")])

        threading.Thread(target=thread, daemon=True).start()

    def _show_gradcam(self):
        def fn():
            out  = os.path.join(OUTPUT_DIR, f"attention_{os.path.splitext(os.path.basename(self.image_path))[0]}.png")
            ok   = run_gradcam(self.image_path, out)
            if ok:
                self.gradcam_path = out
                self.after(0, lambda: [self._update_preview(out, "Attention Map"),
                                       self._open_viz(out, "Attention Map")])
            else:
                self.after(0, lambda: messagebox.showerror("Error",
                    "GradCAM failed. Install: pip install grad-cam"))
        self._run_tool('gradcam', 'Attention Map', fn)

    def _show_bbox(self):
        def fn():
            out     = os.path.join(OUTPUT_DIR, f"regions_{os.path.splitext(os.path.basename(self.image_path))[0]}.png")
            ok, *_  = run_bbox(self.image_path, out)
            if ok:
                self.bbox_path = out
                self.after(0, lambda: [self._update_preview(out, "Region Detection"),
                                       self._open_viz(out, "Region Detection")])
            else:
                self.after(0, lambda: messagebox.showerror("Error",
                    "BBox failed. Ensure bbox_detector.py exists."))
        self._run_tool('bbox', 'Region Detection', fn)

    def _show_shap(self):
        def fn():
            out = os.path.join(OUTPUT_DIR, f"shap_{os.path.splitext(os.path.basename(self.image_path))[0]}.png")
            ok  = run_shap(self.image_path, out)
            if ok:
                self.shap_path = out
                self.after(0, lambda: [self._update_preview(out, "SHAP Analysis"),
                                       self._open_viz(out, "SHAP Analysis")])
            else:
                self.after(0, lambda: messagebox.showerror("Error",
                    "SHAP failed. Install: pip install shap"))
        self._run_tool('shap', 'SHAP Analysis', fn)

    def _update_preview(self, path, title):
        try:
            img   = Image.open(path)
            img.thumbnail((266, 180), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._viz_thumb = photo
            self._preview_title_var.set(title)
            self._viz_preview_label.configure(image=photo, text="", compound='center')
            self._viz_preview_label.image = photo
        except Exception as e:
            print(f"Preview error: {e}")

    def _open_viz(self, path, title):
        win = tk.Toplevel(self)
        win.title(title)
        win.configure(bg=WHITE)
        win.geometry("1160x760")

        bar = tk.Frame(win, bg=GREEN_DARK, height=52)
        bar.pack(fill='x')
        bar.pack_propagate(False)
        tk.Label(bar, text=title, font=FONT_SUBHEAD,
                 bg=GREEN_DARK, fg=WHITE).pack(side='left', padx=20, pady=14)
        tk.Button(bar, text="✕", font=("Segoe UI", 12), bg=GREEN_DARK, fg=WHITE,
                  relief='flat', command=win.destroy, cursor='hand2',
                  activebackground=GREEN_MED, activeforeground=WHITE).pack(side='right', padx=16)

        try:
            img   = Image.open(path)
            img.thumbnail((1120, 660))
            photo = ImageTk.PhotoImage(img)
            lbl   = tk.Label(win, image=photo, bg=WHITE)
            lbl.image = photo
            lbl.pack(padx=16, pady=16)
        except Exception as e:
            tk.Label(win, text=f"Could not display:\n{e}",
                     bg=WHITE, fg=TEXT_LIGHT).pack(pady=20)

        tk.Label(win, text=f"Saved → {path}", font=FONT_TINY,
                 bg=WHITE, fg=TEXT_LIGHT).pack(pady=(0, 10))

    def _export_csv(self):
        if not os.path.exists(OUTPUT_CSV):
            messagebox.showinfo("Export", "No results to export yet.")
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="lumen_results.csv")
        if dest:
            import shutil
            shutil.copy(OUTPUT_CSV, dest)
            messagebox.showinfo("Export", f"Saved to:\n{dest}")

    def _show_error(self, msg):
        self.analyze_btn.set_enabled(True)
        self.analyze_btn.configure_text("▶  Run Analysis")
        self.progress_var.set("")
        messagebox.showerror("Error", msg)

    def _reset(self):
        self.image_path   = None
        self.results      = None
        self.gradcam_path = None
        self.shap_path    = None
        self.bbox_path    = None
        self.ct_valid     = False

        self.thumb_canvas.clear()
        self.fname_var.set("No file selected")
        self.ct_badge_var.set("")
        self.analyze_btn.set_enabled(False)
        self.analyze_btn.configure_text("▶  Run Analysis")
        self.progress_var.set("")

        for card in self._tool_cards.values():
            card.set_enabled(False)
        self.export_btn.set_enabled(False)

        self.results_header.configure(text="Ready to analyze", fg=TEXT_LIGHT)
        self.results_sub.configure(text="", fg=TEXT_LIGHT)
        self._preview_title_var.set("")
        self._viz_preview_label.configure(image='',
            text="Run a tool to\npreview results here")
        self._show_placeholder()

    def _load_model_async(self):
        threading.Thread(target=self._load_model_thread, daemon=True).start()

    def _load_model_thread(self):
        if not MODEL_AVAILABLE:
            self.after(0, lambda: [
                self.status_dot.configure(fg=ACCENT_RED),
                self.status_label.configure(text="Model unavailable")])
            return
        ok = load_model()
        if ok:
            self.model_loaded = True
            self.after(0, lambda: [
                self.status_dot.configure(fg=GREEN_ACCENT),
                self.status_label.configure(text="System ready")])
        else:
            self.after(0, lambda: [
                self.status_dot.configure(fg=ACCENT_RED),
                self.status_label.configure(text="Checkpoint missing")])


if __name__ == "__main__":
    app = LUMENApp()
    app.mainloop()