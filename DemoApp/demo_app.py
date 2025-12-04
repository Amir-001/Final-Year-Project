import os
import io
import numpy as np
import onnxruntime
from flask import Flask, request, send_file, render_template_string, jsonify
from PIL import Image, ImageDraw # Pillow for image manipulation in display/overlay
import rasterio
import base64
import torch # For resizing with interpolate, consistent with your training
import torch.nn.functional as F

ONNX_MODEL_PATH = "moe_system_final.onnx" # The exported ONNX model file

# These are the 1-based indices to read all 6 bands from such a TIFF.
S2_BANDS_TO_READ_FROM_TIFF_1_BASED = [1, 2, 3, 4, 5, 6]

# 0-based indices into the 6-band array (after reading) for R, G, B display.
RGB_VISUALIZATION_BAND_INDICES_0_BASED = [2, 1, 0]

MODEL_INPUT_NUM_BANDS = 6
MODEL_INPUT_IMG_SIZE = 256
NUM_CLASSES = 2
CLASS_NAMES = ["non_cereal", "cereal"]

# Overlay settings - Highlighting "cereal" (class 1) in RED
HIGHLIGHT_CLASS_VALUE = 1 # 0 for "non_cereal", 1 for "cereal"
HIGHLIGHT_CLASS_NAME = CLASS_NAMES[HIGHLIGHT_CLASS_VALUE] # This will be "cereal"
OVERLAY_COLOR = (255, 0, 0) # Red (R, G, B)
OVERLAY_ALPHA = 128 # Opacity (0-255)
# --- End Constants ---

# --- Load ONNX Model ---
try:
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    print(f"ONNX model loaded successfully from {ONNX_MODEL_PATH} using CPUExecutionProvider.")
except Exception as e:
    print(f"FATAL: Error loading ONNX model: {e}\nMake sure '{ONNX_MODEL_PATH}' exists and is a valid ONNX file.")
    ort_session = None
# --- End Load ONNX Model ---

# --- Preprocessing Function ---
def preprocess_uploaded_tif(file_storage):
    file_bytes = io.BytesIO(file_storage.read())
    file_storage.seek(0)
    try:
        with rasterio.open(file_bytes) as src:
            print(f"Opened TIFF: {src.count} bands, {src.height}x{src.width} pixels, dtype: {src.dtypes[0] if src.count > 0 else 'N/A'}")

            if src.count == 0:
                raise ValueError("Uploaded TIFF file has no bands.")
            if src.count < MODEL_INPUT_NUM_BANDS:
                raise ValueError(f"TIFF file has only {src.count} bands, but {MODEL_INPUT_NUM_BANDS} are expected (as per S2_BANDS_TO_READ_FROM_TIFF_1_BASED setting).")

            selected_6bands_data = src.read(S2_BANDS_TO_READ_FROM_TIFF_1_BASED).astype(np.float32)
            
            if selected_6bands_data.shape[0] != MODEL_INPUT_NUM_BANDS:
                 raise ValueError(f"Error after reading bands: Expected {MODEL_INPUT_NUM_BANDS} bands, but got {selected_6bands_data.shape[0]}. Check S2_BANDS_TO_READ_FROM_TIFF_1_BASED or TIFF file content.")

            img_tensor = torch.from_numpy(selected_6bands_data).unsqueeze(0)
            resized_tensor = F.interpolate(img_tensor,
                                           size=(MODEL_INPUT_IMG_SIZE, MODEL_INPUT_IMG_SIZE),
                                           mode='bilinear',
                                           align_corners=False)
            
            model_input_numpy_single = resized_tensor.squeeze(0).numpy()
            resized_6bands_for_display = model_input_numpy_single.copy()
            model_input_numpy_batched = np.expand_dims(model_input_numpy_single, axis=0)

            return model_input_numpy_batched, resized_6bands_for_display
            
    except rasterio.errors.RasterioIOError as e:
        print(f"RasterioIOError during TIFF preprocessing: {e}")
        raise ValueError(f"Could not read the uploaded TIFF file. It might be corrupted or not a valid GeoTIFF. Error: {e}")
    except Exception as e:
        print(f"Unexpected error during TIFF preprocessing: {e}")
        raise ValueError(f"An unexpected error occurred while processing the TIFF file. Error: {e}")

# --- Postprocessing Function ---
def postprocess_output(onnx_output_logits_batch):
    logits = onnx_output_logits_batch[0]
    pred_mask = np.argmax(logits, axis=1)
    pred_mask_2d = pred_mask.squeeze(axis=0)
    return pred_mask_2d.astype(np.uint8)

# --- Function to create displayable RGB ---
def create_rgb_display(image_patch_6bands_resized, band_indices_rgb_0_based, contrast_stretch_percentile=98):
    display_error_message = None
    if image_patch_6bands_resized.shape[0] != MODEL_INPUT_NUM_BANDS:
        msg = f"RGB Display: Expected {MODEL_INPUT_NUM_BANDS} bands, got {image_patch_6bands_resized.shape[0]}."
        print(f"Warning: {msg}")
        rgb_display_arr = np.zeros((image_patch_6bands_resized.shape[1], image_patch_6bands_resized.shape[2], 3), dtype=np.uint8)
        return Image.fromarray(rgb_display_arr), msg

    valid_indices = [idx for idx in band_indices_rgb_0_based if 0 <= idx < MODEL_INPUT_NUM_BANDS]
    if len(valid_indices) < 3:
        msg = f"RGB Display: Not enough valid R,G,B indices in RGB_VISUALIZATION_BAND_INDICES_0_BASED ({band_indices_rgb_0_based}) for the {MODEL_INPUT_NUM_BANDS}-band input. Using grayscale of first band."
        print(f"Warning: {msg}")
        gray_band = image_patch_6bands_resized[0, :, :]
        finite_gray_band = gray_band[np.isfinite(gray_band)]
        if finite_gray_band.size == 0: stretched_gray = np.zeros_like(gray_band)
        else:
            p_low, p_high = np.percentile(finite_gray_band, [100 - contrast_stretch_percentile, contrast_stretch_percentile])
            stretched_gray = np.clip((gray_band - p_low) / (p_high - p_low + 1e-7), 0, 1) * 255
        rgb_display_arr = np.stack([stretched_gray] * 3, axis=-1).astype(np.uint8)
        display_error_message = msg
        return Image.fromarray(rgb_display_arr), display_error_message

    rgb_bands_to_display = image_patch_6bands_resized[valid_indices, :, :]
    stretched_bands = []
    for i in range(rgb_bands_to_display.shape[0]):
        band = rgb_bands_to_display[i, :, :]
        finite_band = band[np.isfinite(band)]
        if finite_band.size == 0: stretched_bands.append(np.zeros_like(band)); continue
        p_low, p_high = np.percentile(finite_band, [100 - contrast_stretch_percentile, contrast_stretch_percentile])
        if p_high <= p_low: stretched_band = np.zeros_like(band) if p_low == 0 else np.ones_like(band) * (255 if band.max() > 0 else 0)
        else: stretched_band = np.clip((band - p_low) / (p_high - p_low + 1e-7), 0, 1)
        stretched_bands.append((stretched_band * 255).astype(np.uint8))
    
    rgb_display_arr = np.stack(stretched_bands, axis=-1)
    return Image.fromarray(rgb_display_arr), display_error_message

# --- Function to create overlay image (Highlights HIGHLIGHT_CLASS_VALUE in OVERLAY_COLOR) ---
def create_overlay(original_rgb_pil, mask_np):
    original_rgba = original_rgb_pil.convert("RGBA")
    overlay_img = Image.new("RGBA", original_rgb_pil.size, (0,0,0,0))
    color_tuple_with_alpha = OVERLAY_COLOR + (OVERLAY_ALPHA,)
    target_pixels = (mask_np == HIGHLIGHT_CLASS_VALUE)
    if np.any(target_pixels):
        highlight_color_layer = Image.new("RGBA", original_rgb_pil.size, color_tuple_with_alpha)
        pil_mask = Image.fromarray(target_pixels.astype(np.uint8) * 255, mode='L')
        # Corrected compositing: original -> highlight_color_layer masked by pil_mask -> then alpha_composite if needed
        # For a simple overlay of colored pixels:
        # Create an empty transparent image, paste color where mask is true, then alpha composite with original
        temp_overlay = Image.new("RGBA", original_rgb_pil.size) # Transparent
        temp_overlay.paste(color_tuple_with_alpha, (0,0), pil_mask) # Paste color using mask
        highlighted_img = Image.alpha_composite(original_rgba, temp_overlay)
    else:
        highlighted_img = original_rgba # No change if no target pixels
    return highlighted_img.convert("RGB")
# --- End Functions ---

# --- Flask App Definition ---
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>MoE Cereal Segmentation Demo</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #eef2f7; color: #333; display: flex; flex-direction: column; align-items: center; padding-top: 20px; min-height:100vh; }
        .container { background-color: #ffffff; padding: 25px 40px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); width: 90%; max-width: 950px; margin-bottom:30px;}
        h1 { color: #2c3e50; text-align: center; margin-bottom: 25px; font-weight:600; }
        p { font-size: 1rem; color: #555; line-height: 1.7; }
        .important { color: #c0392b; font-weight: 500; background-color:#fdf3f2; border: 1px solid #f5c6cb; padding:12px; border-radius:6px; margin-bottom:20px; font-size:0.95rem;}
        .important strong { color: #a94442; }
        label { display: block; margin-bottom: 8px; font-weight: 500; color: #333; }
        input[type="file"] { border: 2px dashed #bdc3c7; padding: 20px; border-radius: 6px; margin-bottom: 25px; width: calc(100% - 44px); background-color:#f8f9fa; text-align:center; cursor:pointer; }
        input[type="file"]::file-selector-button { background-color: #3498db; color: white; padding: 10px 18px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.95em; margin-right:10px; }
        input[type="submit"] { display:block; width:100%; background-color: #e74c3c; color: white; padding: 14px 25px; border: none; border-radius: 6px; cursor: pointer; font-size: 1.1em; font-weight:500; transition: background-color 0.2s ease-in-out; }
        input[type="submit"]:hover { background-color: #c0392b; }
        .results { margin-top: 35px; display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px; }
        .image-container { text-align: center; border: 1px solid #dfe4ea; padding: 15px; background: #fdfdfd; border-radius: 8px; }
        .image-container h2 { font-size: 1.15em; color: #34495e; margin-bottom:12px; font-weight:600; }
        .image-container img { max-width: 100%; height: auto; border: 1px solid #ced4da; border-radius:4px; background-color:#eee; min-height:50px;}
        .footer { margin-top: auto; padding:20px; text-align: center; font-size: 0.85em; color: #7f8c8d; width:100%; background-color:#dfe6e9;}
        .processing-message { font-size: 1.1em; color: #2980b9; text-align:center; margin-top:20px; font-weight:500; }
        .error-message { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 15px; border-radius: 6px; margin-top: 20px; text-align:center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MoE Cereal Segmentation Demo</h1>
        <p class="important">Upload a <strong>GeoTIFF (.tif, .tiff)</strong> file. <br>
           The application expects the TIFF to contain exactly <strong>{{ model_input_num_bands }} bands</strong> in the correct order for the model.
           (These should be the specific Sentinel-2 bands the model was trained on BLUE, GREEN, RED, NIR b8, SWIR1, SWIR2).<br>
           The input image will be resized to <strong>{{ model_img_size }}x{{ model_img_size }}</strong> pixels before processing.
        </p>
        <form method=post enctype=multipart/form-data action="/predict_and_show" onsubmit="showProcessingMessage()">
          <label for="fileUpload">Select GeoTIFF file (containing the {{ model_input_num_bands }} required bands in order):</label>
          <input type=file name=file id="fileUpload" accept=".tif,.tiff" required>
          <input type=submit value="Segment '{{ highlighted_class_name_str }}' (Highlight in Red)">
        </form>
        
        <div id="processingMessage" class="processing-message" style="display:none;">Processing your image, please wait... This might take a moment with CPU inference.</div>

        {% if error_msg %}
            <div class="error-message">{{ error_msg }}</div>
        {% endif %}

        {% if original_img_b64 and overlay_img_b64 %}
        <div class="results">
            <div class="image-container">
                <h2>Original (RGB Composite)</h2>
                {% if rgb_display_note %} <p style="font-size:0.8em; color:orange;">Note: {{ rgb_display_note }} </p> {% endif %}
                <img src="data:image/png;base64,{{ original_img_b64 }}" alt="Original Input Image">
            </div>
            <div class="image-container">
                <h2>Segmentation Overlay (Class: '{{ highlighted_class_name_str }}' Highlighted in Red)</h2>
                <img src="data:image/png;base64,{{ overlay_img_b64 }}" alt="Segmentation Overlay Result">
            </div>
        </div>
        {% endif %}
    </div>
    <div class="footer"><p>MoE Final Year Project Demo &copy; 2024</p></div>

    <script>
        function showProcessingMessage() {
            // Check if file input has a file
            var fileInput = document.getElementById('fileUpload');
            if (fileInput.files.length > 0) {
                 document.getElementById('processingMessage').style.display = 'block';
            } else {
                // Optionally alert user or handle empty submission client-side,
                // though server-side validation is also present.
                // alert("Please select a file first.");
                // return false; // To prevent form submission if desired
            }
            return true; // Allow form submission
        }
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE,
                                  s2_bands_list_str=str(S2_BANDS_TO_READ_FROM_TIFF_1_BASED), # You can remove if not used in HTML
                                  model_img_size=MODEL_INPUT_IMG_SIZE,
                                  model_input_num_bands=MODEL_INPUT_NUM_BANDS,
                                  highlighted_class_name_str=HIGHLIGHT_CLASS_NAME)

@app.route('/predict_and_show', methods=['POST'])
def predict_and_show():
    render_vars = {
        "s2_bands_list_str": str(S2_BANDS_TO_READ_FROM_TIFF_1_BASED), # You can remove if not used in HTML
        "model_img_size": MODEL_INPUT_IMG_SIZE,
        "model_input_num_bands": MODEL_INPUT_NUM_BANDS,
        "highlighted_class_name_str": HIGHLIGHT_CLASS_NAME
    }
    if ort_session is None:
        render_vars["error_msg"] = "ONNX Model not loaded. Check server logs."
        return render_template_string(HTML_TEMPLATE, **render_vars), 500
    
    if 'file' not in request.files:
        render_vars["error_msg"] = "No file part. Please select a file."
        return render_template_string(HTML_TEMPLATE, **render_vars), 400
    file = request.files['file']
    if file.filename == '':
        render_vars["error_msg"] = "No selected file. Please select a file."
        return render_template_string(HTML_TEMPLATE, **render_vars), 400
    if not (file.filename.lower().endswith('.tif') or file.filename.lower().endswith('.tiff')):
        render_vars["error_msg"] = "Invalid file type. Please upload a .tif or .tiff GeoTIFF file."
        return render_template_string(HTML_TEMPLATE, **render_vars), 400

    try:
        model_input_data_batched, resized_6bands_for_display = preprocess_uploaded_tif(file)
        print(f"Input data shape for ONNX prediction: {model_input_data_batched.shape}")

        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: model_input_data_batched}
        ort_outs = ort_session.run(None, ort_inputs)
        print("ONNX Inference successful.")

        predicted_mask_2d_np = postprocess_output(ort_outs)
        
        unique_classes, counts = np.unique(predicted_mask_2d_np, return_counts=True)
        class_counts_dict = dict(zip([CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in unique_classes], counts))
        print(f"Unique classes in predicted mask: {class_counts_dict}")
        pixels_highlighted = np.sum(predicted_mask_2d_np == HIGHLIGHT_CLASS_VALUE)
        print(f"Number of pixels predicted as '{HIGHLIGHT_CLASS_NAME}' (class {HIGHLIGHT_CLASS_VALUE}): {pixels_highlighted}")

        original_rgb_pil, rgb_display_note_str = create_rgb_display(resized_6bands_for_display, RGB_VISUALIZATION_BAND_INDICES_0_BASED)
        overlay_pil = create_overlay(original_rgb_pil, predicted_mask_2d_np)

        original_img_io = io.BytesIO()
        original_rgb_pil.save(original_img_io, 'PNG')
        original_img_io.seek(0)
        render_vars["original_img_b64"] = base64.b64encode(original_img_io.getvalue()).decode('utf-8')

        overlay_img_io = io.BytesIO()
        overlay_pil.save(overlay_img_io, 'PNG')
        overlay_img_io.seek(0)
        render_vars["overlay_img_b64"] = base64.b64encode(overlay_img_io.getvalue()).decode('utf-8')
        
        if rgb_display_note_str:
            render_vars["rgb_display_note"] = rgb_display_note_str
        
        return render_template_string(HTML_TEMPLATE, **render_vars)

    except ValueError as e:
        print(f"ValueError during processing: {e}")
        render_vars["error_msg"] = str(e)
        return render_template_string(HTML_TEMPLATE, **render_vars), 400
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        import traceback
        traceback.print_exc()
        render_vars["error_msg"] = "An unexpected error occurred. Please check the file or server logs."
        return render_template_string(HTML_TEMPLATE, **render_vars), 500

if __name__ == '__main__':
    print(f"Starting Flask app. Access at http://0.0.0.0:5000 or http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)