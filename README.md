# 🏨 Hotel Booking Cancellation Prediction

Proyek ini bertujuan untuk memprediksi kemungkinan pembatalan pemesanan hotel menggunakan Machine Learning, sehingga pihak hotel dapat meminimalkan kerugian finansial akibat pembatalan mendadak.

---

## 1. 🎯 Business Problem Understanding

Pihak hotel sering mengalami kerugian karena:
- **False Negative (FN)**: Pemesanan tidak terdeteksi akan dibatalkan → kamar tidak dijual kembali → rugi.
- **False Positive (FP)**: Pemesanan diprediksi batal padahal tidak → terjadi overbooking → tamu kecewa.

**Biaya:**
- FN = €160 per malam  
- FP = €25 per malam

🎯 Tujuan: Meminimalkan total kerugian dari kesalahan prediksi pembatalan.

---

## 2. 📊 Data Understanding

Dataset berisi informasi pemesanan hotel seperti:
- Negara asal tamu
- Jenis pasar (market segment)
- Tipe pelanggan
- Jumlah pembatalan sebelumnya
- Permintaan khusus, dll.

🎯 Target: `is_canceled` → 0 = hadir, 1 = batal

---

## 3. 🧹 Data Cleaning

Langkah-langkah:
- Menangani missing values
- Menangani Duplicates
- Menangani Outliers

---

## 4. 📈 Data Analysis

Dilakukan analisis eksploratif (EDA) untuk:
- Menemukan fitur penting yang memengaruhi pembatalan
- Visualisasi distribusi data dan korelasi antar fitur

📌 Insight:
- Pelanggan dengan **previous cancellations** tinggi cenderung batal
- **Non-refundable deposit** berhubungan kuat dengan pembatalan
- Booking dari negara/segmen tertentu lebih berisiko dibatalkan

---

## 5. 🤖 Modelling and Evaluation

- Model terbaik: **Gradient Boosting Classifier**
- Evaluasi dengan **F2-Score** karena penalti FN (kehilangan pendapatan) lebih besar daripada FP
- Perbandingan kerugian:
  - **Tanpa model ML:** rugi ~ €38,950
  - **Dengan model ML:** rugi ~ €23,535
  - **Efisiensi:** Pengurangan kerugian sebesar **~39.56%**

---

## 6. ✅ Conclusion

Model Machine Learning berhasil membantu hotel memprediksi pembatalan dengan lebih akurat, dan secara signifikan menurunkan kerugian bisnis dari booking yang tidak jadi hadir.

---

## 7. 💡 Recommendation

- Gunakan model ini untuk memfilter booking yang berisiko batal dan tawarkan kebijakan khusus (seperti deposit tambahan).
- Kembangkan model dengan data real-time dan fitur tambahan seperti tanggal booking atau jenis kamar.
- Lakukan retraining model secara berkala agar tetap relevan dengan pola pemesanan terbaru.

---

## 🚀 Streamlit App (Demo)

Proyek ini telah dikembangkan menjadi aplikasi interaktif menggunakan **Streamlit** dan dapat langsung dicoba melalui link berikut:

🔗 [Coba Aplikasinya di Sini](https://hotel-booking-cancellation-prediction-ml.streamlit.app/)  


