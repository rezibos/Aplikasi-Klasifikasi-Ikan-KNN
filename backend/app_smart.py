# ============================================================================
# 🐟 API KLASIFIKASI IKAN - VERSI SMART
# Backend yang cepat dan akurat seperti manusia!
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# ============================================================================
# KONFIGURASI
# ============================================================================

# Path model (sesuaikan dengan lokasi model Anda)
MODEL_PATH = './models/model_best.h5'
CLASS_MAP_PATH = './models/class_mapping.json'

# Image size (sesuai dengan training)
IMG_SIZE = 224  # Untuk MobileNetV2 (model fixed)

# Load model dan class mapping
print("🚀 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

try:
    with open(CLASS_MAP_PATH, 'r', encoding='utf-8') as f:
        class_mapping = json.load(f)
    print(f"✅ Class mapping loaded: {len(class_mapping)} classes")
except Exception as e:
    print(f"❌ Error loading class mapping: {e}")
    class_mapping = {}

# ============================================================================
# SMART PREPROCESSING - Seperti Mata Manusia
# ============================================================================

def smart_preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Preprocess gambar dengan teknik yang meniru cara manusia melihat
    """
    # Konversi ke numpy array jika belum
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Pastikan RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # 1. CLAHE - Perbaiki kontras
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    
    # 2. Denoising - Hilangkan noise
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # 3. Sharpening - Pertegas tepi
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    image = cv2.filter2D(image, -1, kernel)
    
    # 4. Resize dengan interpolasi terbaik
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    # 5. Normalize
    image = image.astype(np.float32) / 255.0
    
    return image

def simple_preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Preprocessing sederhana (backup jika smart preprocessing terlalu lambat)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    
    return image

# ============================================================================
# PREDIKSI DENGAN CONFIDENCE ANALYSIS
# ============================================================================

def predict_with_analysis(image, use_smart_preprocessing=True):
    """
    Prediksi dengan analisis confidence yang detail
    """
    if model is None:
        return {
            'error': 'Model not loaded',
            'success': False
        }
    
    try:
        # Preprocessing
        if use_smart_preprocessing:
            processed_image = smart_preprocess_image(image)
        else:
            processed_image = simple_preprocess_image(image)
        
        # Expand dimensions untuk batch
        input_image = np.expand_dims(processed_image, axis=0)
        
        # Prediksi
        predictions = model.predict(input_image, verbose=0)[0]
        
        # Ambil top 5 prediksi
        top_indices = np.argsort(predictions)[::-1][:5]
        
        results = []
        for idx in top_indices:
            class_id = str(idx)
            confidence = float(predictions[idx])
            
            if class_id in class_mapping:
                class_info = class_mapping[class_id]
                name_en = class_info.get('name_en', f'Class {idx}')
                name_id = class_info.get('name_id', name_en)
            else:
                name_en = f'Class {idx}'
                name_id = name_en
            
            results.append({
                'class_id': int(idx),
                'name_en': name_en,
                'name_id': name_id,
                'confidence': confidence,
                'percentage': f'{confidence * 100:.2f}%'
            })
        
        # Analisis confidence
        top_confidence = results[0]['confidence']
        second_confidence = results[1]['confidence'] if len(results) > 1 else 0
        confidence_gap = top_confidence - second_confidence
        
        # Tentukan certainty level
        if top_confidence >= 0.95:
            certainty = 'very_high'
            certainty_text = 'Sangat Yakin'
            certainty_emoji = '🎯'
        elif top_confidence >= 0.85:
            certainty = 'high'
            certainty_text = 'Yakin'
            certainty_emoji = '✅'
        elif top_confidence >= 0.70:
            certainty = 'medium'
            certainty_text = 'Cukup Yakin'
            certainty_emoji = '👍'
        elif top_confidence >= 0.50:
            certainty = 'low'
            certainty_text = 'Kurang Yakin'
            certainty_emoji = '⚠️'
        else:
            certainty = 'very_low'
            certainty_text = 'Tidak Yakin'
            certainty_emoji = '❓'
        
        # Rekomendasi
        recommendations = []
        if top_confidence < 0.70:
            recommendations.append('Coba foto dari sudut yang lebih jelas')
            recommendations.append('Pastikan pencahayaan cukup')
        if confidence_gap < 0.15:
            recommendations.append('Model ragu antara beberapa jenis')
            recommendations.append('Coba foto yang lebih dekat atau detail')
        
        return {
            'success': True,
            'prediction': results[0]['name_id'],
            'prediction_en': results[0]['name_en'],
            'confidence': top_confidence,
            'percentage': f'{top_confidence * 100:.2f}%',
            'certainty': certainty,
            'certainty_text': certainty_text,
            'certainty_emoji': certainty_emoji,
            'confidence_gap': confidence_gap,
            'top_5_predictions': results,
            'recommendations': recommendations,
            'preprocessing': 'smart' if use_smart_preprocessing else 'simple'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def home():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'online',
        'message': '🐟 Smart Fish Classifier API',
        'version': '2.0 Smart',
        'model_loaded': model is not None,
        'classes_loaded': len(class_mapping),
        'img_size': IMG_SIZE,
        'features': [
            'Smart Preprocessing (CLAHE + Denoising + Sharpening)',
            'EfficientNetB4 Architecture',
            'Top-5 Predictions',
            'Confidence Analysis',
            'Recommendations'
        ]
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint prediksi gambar
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided',
            'success': False
        }), 400
    
    try:
        # Ambil file
        file = request.files['image']
        
        # Ambil parameter preprocessing (default: smart)
        use_smart = request.form.get('smart_preprocessing', 'true').lower() == 'true'
        
        # Baca gambar
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Prediksi
        result = predict_with_analysis(image, use_smart_preprocessing=use_smart)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """
    Endpoint untuk mendapatkan daftar kelas
    """
    classes_list = []
    for class_id, class_info in class_mapping.items():
        if isinstance(class_info, dict):
            classes_list.append({
                'id': int(class_id),
                'name_en': class_info.get('name_en', ''),
                'name_id': class_info.get('name_id', '')
            })
        else:
            classes_list.append({
                'id': int(class_id),
                'name_en': class_info,
                'name_id': class_info
            })
    
    # Sort by ID
    classes_list.sort(key=lambda x: x['id'])
    
    return jsonify({
        'success': True,
        'total_classes': len(classes_list),
        'classes': classes_list
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Endpoint untuk informasi model
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 500
    
    try:
        # Get model summary
        total_params = model.count_params()
        
        return jsonify({
            'success': True,
            'model_name': model.name if hasattr(model, 'name') else 'Unknown',
            'total_parameters': int(total_params),
            'input_shape': [int(x) if x is not None else None for x in model.input_shape],
            'output_shape': [int(x) if x is not None else None for x in model.output_shape],
            'num_classes': len(class_mapping),
            'img_size': IMG_SIZE,
            'preprocessing': 'Smart (CLAHE + Denoising + Sharpening)'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🐟 SMART FISH CLASSIFIER API")
    print("="*70)
    print(f"Model: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"Classes: {len(class_mapping)}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Preprocessing: Smart (CLAHE + Denoising + Sharpening)")
    print("="*70)
    print("\n🚀 Starting server on http://localhost:5000")
    print("📚 API Endpoints:")
    print("   GET  /              - Health check")
    print("   POST /api/predict   - Prediksi gambar")
    print("   GET  /api/classes   - Daftar kelas")
    print("   GET  /api/model-info - Info model")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
