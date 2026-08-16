# 📋 STANDAR OPERASIONAL PROSEDUR (SOP)
## Pengambilan Data Foto & Estimasi Bobot Sapi (*proj-sapi*)

**Versi:** 1.0  
**Target Pengguna:** Operator Lapangan, Peternak, dan Pengambil Data  
**Tujuan:** Memandu proses penempelan stiker kalibrasi, pengambilan foto sapi (samping & belakang), penginputan ke aplikasi *proj-sapi*, hingga validasi timbangan secara presisi dan konsisten.

---

### 🧰 Perlengkapan yang Diperlukan:
1. **Stiker Kalibrasi** (Lingkaran atau Persegi dengan warna kontras: Kuning Stabilo / Magenta / Cyan).
2. **Double Tape Busa (*Foam Tape*)** atau perekat stiker yang kuat namun mudah dilepas.
3. **Kain Lap Pembersih** (Kering & bersih).
4. **Smartphone / Kamera Digital** (Lensa dalam kondisi bersih, disarankan resolusi kamera \(\ge 12\text{ MP}\)).
5. **Aplikasi / Web *proj-sapi***.
6. **Timbangan Hewan Fisik (Konvensional/Digital)** (Untuk validasi berat nyata).

---

## 📝 9 Langkah Prosedur Kerja Lapangan

### 1. Menyiapkan Stiker Kalibrasi
* Ambil stiker kalibrasi yang sudah disiapkan.
* Pasang potongan *double tape* pada **minimal 3–4 titik** di sisi belakang stiker (terutama di bagian tepi dan tengah) atau buka lapisan pelindung perekatnya.
* **Pastikan stiker dalam keadaan rata sempurna (tidak melengkung, tidak terlipat, dan tidak bergelombang)** agar perhitungan skala pixel akurat.

---

### 2. Membersihkan Area Kulit Sapi
* Dekati sapi dengan tenang dan pastikan sapi dalam kondisi rileks.
* Bersihkan area kulit dan bulu yang akan ditempeli stiker dari **kotoran, debu, lumpur, serta air/keringat** menggunakan lap kering.
* Permukaan bulu yang kering dan bersih memastikan stiker menempel kuat dan batas warna stiker terbaca tajam oleh kamera.

---

### 3. Memasang Stiker Tampak Samping (*Side View*)
* Tempelkan stiker pada **sisi samping badan sapi** (pada area tengah badan / tulang rusuk tengah yang relatif rata).
* Pasang stiker **sejajar dengan garis horizontal tubuh sapi**.
* Tekan perlahan stiker dari bagian tengah ke arah luar agar menempel rata tanpa rongga udara.

---

### 4. Mengambil Foto Tampak Samping (*Side View*)
* Posisikan diri Anda berdiri **tegak lurus (sudut 90°)** tepat menghadap badan sapi dan stiker.
* Jaga ketinggian kamera sejajar dengan bagian tengah tubuh sapi (jangan memotret dari sudut atas/bawah yang miring).
* **Syarat Mutlak Komposisi Foto Samping:**
  * Seluruh tubuh sapi **wajib masuk 100% ke dalam bingkai foto**: mulai dari ujung moncong/kepala, punuk, keempat telapak kaki yang menjejak tanah, hingga ujung ekor.

#### 📐 Panduan Jarak & Komposisi Foto Samping:
| Kondisi Foto | Status | Dampak pada Sistem |
| :--- | :---: | :--- |
| **Terlalu Dekat** | ❌ Ditolak | Ada anggota tubuh yang terpotong (misal kaki atau kepala hilang). Sistem gagal mengukur panjang badan. |
| **Terlalu Jauh** | ❌ Ditolak | Sapi tampak terlalu kecil di layar; resolusi pixel stiker menjadi terlalu rendah sehingga skala tidak akurat. |
| **Jarak Tepat (Ideal)** | ✅ **Diterima** | **Sapi mengisi sekitar 70% – 85% area layar foto**. Seluruh batas tubuh dan stiker terlihat sangat tajam dan jelas. |

---

### 5. Memasang Stiker & Mengambil Foto Tampak Belakang (*Back View*)
* Pasang stiker kalibrasi di area **bokong / panggul sapi** menghadap lurus ke arah belakang dengan metode penempelan yang sama (rata dan tidak bergelombang).
* Berdiri di belakang sapi dengan jarak aman (\(\pm 2\text{ m} - 3\text{ m}\)) dan posisi kamera **tegak lurus (sudut 90°)** terhadap lebar bokong sapi.
* Ambil foto tampak belakang: pastikan seluruh lebar panggul, kedua kaki belakang, dan stiker kalibrasi masuk jelas ke dalam frame foto.

---

### 6. Memasukkan Foto ke Aplikasi *proj-sapi*
* Buka antarmuka aplikasi **proj-sapi** pada perangkat/browser Anda.
* Unggah berkas foto yang baru diambil:
  1. Input **Foto Tampak Samping (*Side Image*)**.
  2. Input **Foto Tampak Belakang (*Back Image*)**.
* Pada menu konfigurasi bentuk stiker (**Sticker Shape**), pilih sesuai stiker yang Anda pakai:
  * Pilih **`Circle`** jika menggunakan stiker lingkaran.
  * Pilih **`Square`** jika menggunakan stiker kotak/persegi.

---

### 7. Memasukkan Ukuran Stiker Kalibrasi
* Masukkan nilai ukuran stiker kalibrasi asli dalam satuan centimeter (cm):
  * Jika memilih **`Circle`**: Masukkan **Diameter (cm)** stiker (contoh: `11.0`, `14.0`, atau `19.0`).
  * Jika memilih **`Square`**: Masukkan nilai **Panjang/Lebar Sisi (cm)** (contoh: `10.16`).
* Tekan tombol **"Predict / Hitung Bobot"**.

---

### 8. Mencatat Hasil Estimasi Sistem
* Catat seluruh hasil perhitungan yang ditampilkan oleh sistem:
  * **Estimasi Bobot Sapi (*Predicted Weight*)**: dalam satuan kilogram (kg).
  * **Ukuran Morfometrik Tubuh**:
    1. *Body Length* (Panjang Badan) dalam cm.
    2. *Withers Height* (Tinggi Pundak) dalam cm.
    3. *Chest Girth* (Lingkar Dada) dalam cm.

---

### 9. Validasi dengan Timbangan Konvensional
* Segera giring sapi menuju ke alat **timbangan ternak fisik / digital konvensional**.
* Timbang sapi dan catat bobot nyata (*Ground Truth Weight*) pada form pencatatan.
* Bandingkan hasil prediksi sistem dengan bobot timbangan nyata untuk keperluan dokumentasi, verifikasi, dan monitoring harian.
* Lepaskan stiker kalibrasi dari tubuh sapi secara perlahan setelah proses selesai.

---

## ⚠️ Rangkuman Pantangan & Tips Lapangan
1. ❌ **Dilarang mengambil foto dengan sudut miring / menyerong**: Karena perspektif miring mengubah bentuk dan panjang proporsi tubuh sapi.
2. ❌ **Dilarang membiarkan stiker tertutup kotoran**: Stiker harus bersih agar algoritma segmentasi YOLOv8 dapat mendeteksi kontur stiker secara utuh.
3. ✅ **Pastikan pencahayaan cukup**: Hindari backlight ekstrem atau bayangan gelap pekat yang menutupi garis tepi tubuh sapi.
