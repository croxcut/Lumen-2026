LUMEN - Lung Scan Analysis System

LUMEN is a desktop application for analyzing chest CT scans using the LungMaxViT hybrid deep learning model. It detects six pulmonary conditions and provides explainability tools including Grad-CAM attention maps, bounding box region detection, and SHAP feature influence analysis.

REQUIREMENTS

Python 3.12.13
torch 2.3.0+cu121
torchvision 0.18.0+cu121
timm 1.0.3
pandas 2.2.2
numpy 1.26.4
pillow 10.3.0
scikit-learn 1.5.0
opencv-python 4.10.0
shap 0.45.1
grad-cam 1.5.0


INSTALLATION

Install PyTorch with CUDA 12.1 first:

pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121

Then install the remaining packages:

pip install timm==1.0.3 pandas==2.2.2 numpy==1.26.4 pillow==10.3.0 scikit-learn==1.5.0 opencv-python==4.10.0 shap==0.45.1 grad-cam==1.5.0


SETUP

Place the trained checkpoint file best_model_finetuned.pth in the same directory as gui.py. The file bbox_detector.py must also be in the same directory if you want to use Region Detection.


HOW TO RUN

python gui.py


USAGE

1. Click Browse Scan and select a chest CT image (JPG, PNG, BMP, or TIFF).
2. Click Run Analysis to get predictions for all six conditions.
3. After analysis, use the tools in the Analysis Tools panel:
   - Attention Map runs Grad-CAM++ to highlight where the model focused.
   - Region Detection draws bounding boxes around detected abnormal areas.
   - SHAP Analysis generates per-class feature influence heatmaps. - this is fucked up sooo idk :D*
4. Click Export CSV to save all prediction scores to a CSV file.
5. Click New Analysis to reset and load a new scan.


DETECTABLE CONDITIONS

Pulmonary Fibrosis / ILD (AUC 0.9999)
Lung Cancer / Nodules (AUC 0.9996)
Normal (AUC 0.9960)
Tuberculosis (AUC 1.0000)
COVID-19 (AUC 0.9974)
Pneumonia (AUC 1.0000)


OUTPUT

All results are saved to a folder called lumen_results in the working directory. Prediction scores are appended to lumen_results/predictions.csv. Visualization images from the analysis tools are also saved there.


NOTES

The model runs on CPU by default. GPU is used automatically if available through the training scripts but the GUI inference is CPU-bound for compatibility.
SHAP analysis can take several minutes depending on your hardware.
The app will warn if an uploaded image does not appear to be a grayscale CT scan, but will still allow analysis.
