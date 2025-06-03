# Deteksi Keliling Luka Kronis v2 (Refactor)

Versi 2 ini merupakan refaktor dari implementasi sebelumnya untuk deteksi keliling luka kronis menggunakan Active Contour (Snake) dan variannya.

## Motivasi Refaktor

- Merapikan struktur kode agar lebih mudah dipahami dan dikembangkan
- Modularisasi dan pemisahan fungsi-fungsi utama
- Penambahan dokumentasi dan komentar kode
- Memudahkan penambahan fitur atau pengujian metode baru

## Struktur Folder

### `data/`

Folder untuk menyimpan seluruh data input dan output gambar

### `notebooks/`

Jupyter Notebook untuk eksperimen, testing, dan catatan

### `utils.py`

Berisi fungsi-fungsi pembantu yang umum dan reusable

### `active_contour.py`

Berisi fungsi-fungsi utama untuk deteksi keliling menggunakan active contour (snake)

### `cli.py`

Program utama untuk batch processing

## Status

Versi 2 masih dalam tahap pengembangan/refaktor. Masukan dan kontribusi sangat terbuka.

---

Proyek ini tetap mengacu pada lisensi GNU GPL v3.0 seperti versi sebelumnya.
