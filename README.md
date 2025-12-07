# 🐟 Aplikasi Klasifikasi Ikan

Aplikasi web untuk mengklasifikasikan jenis ikan menggunakan Deep Learning (ResNet50) dengan Python Flask sebagai backend dan React sebagai frontend.

## 📋 Fitur

- ✅ Klasifikasi 9 jenis ikan
- ✅ Antarmuka web yang user-friendly dalam bahasa Indonesia
- ✅ Drag & drop upload gambar
- ✅ Deteksi ikan tidak dikenal
- ✅ Menampilkan confidence score dan top 3 prediksi
- ✅ Responsive design

## 🎯 Dataset

Dataset yang digunakan: [Fish Classification - Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

Jenis ikan yang dapat diklasifikasi:
1. Ikan Sprat Laut Hitam (Black Sea Sprat)
2. Ikan Gilthead (Gilt-Head Bream)
3. Ikan Kembung (Hourse Mackerel)
4. Ikan Kuniran (Red Mullet)
5. Ikan Kakap Merah (Red Sea Bream)
6. Ikan Kakap Putih (Sea Bass)
7. Udang (Shrimp)
8. Ikan Kuniran Bergaris (Striped Red Mullet)
9. Ikan Trout (Trout)

## 🛠️ Teknologi yang Digunakan

### Backend:
- Python 3.8+
- TensorFlow 2.15
- Flask
- ResNet50 (Transfer Learning)

### Frontend:
- React 18
- Node.js
- Axios
- CSS3

## 📦 Instalasi

### Prasyarat:
- Python 3.8 atau lebih tinggi
- Node.js 14 atau lebih tinggi
- pip
- npm atau yarn

### Langkah 1: Clone atau Download Project

```bash
cd "/home/firaz/Data - Belajar/TUGAS KULIAH SEMUA/Semester 3/Kecerdasan Buatan/AI - KECERDASAN BUATAN/klasifikasi-ikan"
```

### Langkah 2: Download Dataset

1. Buka link dataset: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset
2. Download dataset (perlu akun Kaggle)
3. Extract file zip
4. Copy folder yang berisi subfolder jenis ikan ke `backend/dataset/`

Struktur folder yang benar:
```
backend/
├── dataset/
│   ├── Black Sea Sprat/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   └── ...
│   ├── Gilt-Head Bream/
│   ├── Hourse Mackerel/
│   ├── Red Mullet/
│   ├── Red Sea Bream/
│   ├── Sea Bass/
│   ├── Shrimp/
│   ├── Striped Red Mullet/
│   └── Trout/
├── app.py
├── train_model.py
└── requirements.txt
```

### Langkah 3: Setup Backend (Python)

```bash
# Masuk ke folder backend
cd backend

# Buat virtual environment (opsional tapi direkomendasikan)
python3 -m venv venv

# Aktifkan virtual environment
# Untuk Linux/Mac:
source venv/bin/activate
# Untuk Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Training model (ini akan memakan waktu, tergantung spesifikasi komputer)
# Pastikan dataset sudah ada di folder dataset/
python train_model.py

# Setelah training selesai, jalankan server Flask
python app.py
```

Server backend akan berjalan di `http://localhost:5000`

### Langkah 4: Setup Frontend (React)

Buka terminal baru (jangan tutup terminal backend):

```bash
# Masuk ke folder frontend
cd frontend

# Install dependencies
npm install

# Jalankan aplikasi React
npm start
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:3000`

## 🚀 Cara Menggunakan

1. Pastikan backend (Flask) berjalan di terminal pertama
2. Pastikan frontend (React) berjalan di terminal kedua
3. Buka browser dan akses `http://localhost:3000`
4. Upload gambar ikan dengan cara:
   - Klik tombol "Pilih Gambar"
   - Atau drag & drop gambar ke area upload
5. Klik tombol "🔍 Analisis Gambar"
6. Tunggu beberapa detik untuk hasil prediksi
7. Hasil akan menampilkan:
   - Nama ikan dalam bahasa Indonesia
   - Tingkat keyakinan (confidence score)
   - Top 3 prediksi alternatif

## ⚙️ Konfigurasi

### Backend (`backend/app.py`):

```python
# Ubah confidence threshold (default 0.5 = 50%)
CONFIDENCE_THRESHOLD = 0.5

# Ubah port
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Frontend (`frontend/src/App.js`):

```javascript
// Ubah URL backend jika berbeda
const API_URL = 'http://localhost:5000';
```

## 🔧 Training Model

Jika ingin melatih ulang model dengan parameter berbeda, edit `backend/train_model.py`:

```python
# Konfigurasi
IMG_SIZE = 224          # Ukuran gambar input
BATCH_SIZE = 32         # Batch size untuk training
EPOCHS = 50             # Jumlah epoch
LEARNING_RATE = 0.0001  # Learning rate
```

Lalu jalankan:
```bash
cd backend
python train_model.py
```

## 📊 Hasil Training

Model akan disimpan di:
- `backend/models/fish_classifier_best.h5` - Model dengan validation accuracy terbaik
- `backend/models/fish_classifier_final.h5` - Model di akhir training
- `backend/models/class_mapping.json` - Mapping kelas ke nama Indonesia

## ❗ Troubleshooting

### Error: Module not found
```bash
# Pastikan virtual environment aktif dan install ulang
pip install -r requirements.txt
```

### Error: Model tidak ditemukan
```bash
# Training model terlebih dahulu
cd backend
python train_model.py
```

### Error: Dataset tidak ditemukan
- Pastikan dataset sudah di-extract ke folder `backend/dataset/`
- Cek struktur folder sudah benar (ada subfolder untuk setiap jenis ikan)

### Error: Cannot connect to backend
- Pastikan Flask server sudah berjalan di `http://localhost:5000`
- Cek firewall tidak memblokir port 5000
- Periksa CORS sudah diaktifkan di backend

### Error: Port already in use
```bash
# Backend (Python)
# Ubah port di app.py atau kill proses yang menggunakan port 5000
lsof -ti:5000 | xargs kill -9

# Frontend (React)
# Ubah port dengan environment variable
PORT=3001 npm start
```

## 📝 Catatan Penting

1. **Training Model**: Proses training bisa memakan waktu 30 menit - 2 jam tergantung spesifikasi komputer. Untuk hasil terbaik, gunakan GPU.

2. **Ukuran Model**: Model yang dihasilkan cukup besar (~100MB). Pastikan ada cukup ruang disk.

3. **Dataset**: Dataset dari Kaggle cukup besar (~2GB). Download memerlukan koneksi internet yang stabil.

4. **Memory**: Training memerlukan RAM minimal 8GB. Jika RAM kurang, kurangi BATCH_SIZE di `train_model.py`.

## 🎓 Untuk Tugas Kuliah

Aplikasi ini cocok untuk:
- Tugas Kecerdasan Buatan
- Project Computer Vision
- Implementasi Deep Learning
- Studi kasus Transfer Learning

## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi/tugas kuliah.

## 👤 Kredit

- Dataset: [Kaggle - Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)
- Model: ResNet50 (Pre-trained on ImageNet)
- Framework: TensorFlow, Flask, React

## 🤝 Kontribusi

Jika menemukan bug atau ingin menambah fitur, silakan buat issue atau pull request.

---

**Selamat mencoba! 🚀**
