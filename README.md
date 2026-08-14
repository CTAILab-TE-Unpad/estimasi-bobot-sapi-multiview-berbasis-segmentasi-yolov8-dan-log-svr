# Estimasi Bobot Sapi Multiview Berbasis Segmentasi YOLOv8 dan Log-SVR

## 📌 Deskripsi Proyek

Proyek ini mengusulkan sebuah sistem visi komputer untuk mengestimasi bobot sapi secara nirsentuh (*contactless*) menggunakan citra dua dimensi (2D). Sistem ini memanfaatkan gambar sapi dari dua sudut pandang (*multiview*: tampak samping dan tampak punggung) untuk mengekstraksi dimensi tubuh (morfometrik) sapi yang secara biologis berkorelasi kuat dengan bobot badannya.

Repositori ini berisi keseluruhan *pipeline* mulai dari eksperimen Jupyter Notebook hingga kode *deployment* yang siap produksi (Web UI dan REST API).

## 🔬 Metodologi

Sistem ini berjalan di atas empat pilar utama pemrosesan citra dan *machine learning*:

1. **Deteksi Stiker & Kalibrasi Metrik**:
   Sistem memanfaatkan stiker referensi (4x4 cm) pada tubuh sapi untuk mendapatkan faktor skala piksel-ke-sentimeter. Deteksi dilakukan menggunakan model YOLO kustom (`best_sticker.pt`). Jika stiker gagal dideteksi pada salah satu sisi, sistem menggunakan pendekatan kalibrasi silang berdasarkan rasio Tinggi Gumba (*Withers Height*).
2. **Segmentasi Objek Sapi (YOLOv8)**:
   Model YOLOv8 Instance Segmentation (`yolov8l-seg.pt`) digunakan untuk mengisolasi piksel tubuh sapi dari latar belakang (*background removal*).
3. **Ekstraksi Fitur Morfometrik**:
   Melalui kontur dan titik ekstrem hasil segmentasi, sistem mengukur:

   - **Tampak Samping**: Panjang Badan (*Body Length*) dan Dalam Dada (*Depth/Withers Height*).
   - **Tampak Punggung**: Lebar Punggung (*Width*).
   - **Lingkar Dada (*Chest Girth*)**: Dihitung secara matematis dari *Depth* dan *Width* menggunakan rumusan aproksimasi keliling elips **Ramanujan**.
4. **Pemodelan Regresi (Log-SVR)**:
   Sistem menggunakan algoritma **Support Vector Regression (SVR)** dengan fungsi kernel *Radial Basis Function (RBF)*. Karena sifat biologis bobot yang non-linear dan heteroskedastik, nilai target (bobot asli) ditransformasi menggunakan logaritma natural (*Log-Space*) sebelum dilatih dan dikembalikan (*eksponensial*) saat inferensi.

## 📁 Struktur Direktori

Keseluruhan repositori ini diletakkan dalam satu lingkungan agar *self-contained*:

```text
script_final/
├── datasets/                                 # Kumpulan citra pengujian (Samping & Punggung)
├── models/
│   ├── yolov8l-seg.pt                        # Bobot model Segmentasi Sapi
│   ├── best_sticker.pt                       # Bobot model Deteksi Stiker
│   ├── svr_log_pipeline.joblib               # Pipeline Scikit-Learn SVR terlatih
│   └── model_metadata.joblib                 # Metadata pipeline SVR
├── api_inference_svr.py                      # Skrip backend REST API menggunakan FastAPI
├── web-inference-SVR.py                      # Skrip frontend Web UI menggunakan Streamlit
├── feature_extraction_with_calibration.ipynb # Eksperimen ekstraksi fitur & kalibrasi jembatan
├── nonlinear_regression_evaluation.ipynb     # Eksperimen model regresi Log-SVR
├── segmentation_test.ipynb                   # Eksperimen segmentasi mask YOLOv8
```

## 🚀 Cara Menjalankan

### 1. Antarmuka Web (Streamlit)

Antarmuka web interaktif yang memungkinkan pengguna mengunggah gambar sapi, melihat hasil deteksi *bounding-box*, topeng segmentasi, garis ukur morfometrik, dan prediksi berat.

```bash
# Pastikan Anda berada di direktori script_final
streamlit run web-inference-SVR.py
```

### 2. Layanan Backend API (FastAPI)

Layanan tanpa wajah (*headless microservice*) untuk integrasi ke sistem aplikasi lain (misalnya Mobile App) melalui *endpoints* RESTful.

```bash
# Pastikan Anda berada di direktori script_final
python api_inference_svr.py
```

*Server akan berjalan di port `8000`. Buka `http://localhost:8000/docs` di browser untuk mengakses antarmuka interaktif Swagger UI untuk pengujian API.*

---

**Pengembang**: [Bim Yusuf/ Lab CTAI]
**Bahasa**: Python 3.10+
