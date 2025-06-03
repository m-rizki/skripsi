# Deteksi Keliling Luka Kronis v2 (Refactor)

Versi 2 ini merupakan refaktor dari implementasi sebelumnya untuk deteksi keliling luka kronis menggunakan _Active Contour (Snake)_ dan variannya.

## Motivasi Refaktor

- Merapikan struktur kode agar lebih mudah dipahami dan dikembangkan
- Modularisasi dan pemisahan fungsi-fungsi utama
- Penambahan dokumentasi dan komentar kode
- Memudahkan penambahan fitur atau pengujian metode baru

## Struktur Folder

```bash
  |--data/              # Folder untuk menyimpan seluruh data input dan output gambar
  |--notebooks/         # Jupyter Notebook untuk eksperimen, testing, dan catatan
  |--utils.py           # Berisi fungsi-fungsi pembantu yang umum dan reusable
  |--active_contour.py  # Berisi fungsi-fungsi utama untuk deteksi keliling menggunakan active contour (snake)
  |--cli.py             # Program utama untuk batch processing
```

## Petunjuk penggunaan

Project ini menggunakan `venv` untuk mengelola _virtual environment_ dengan tujuan kompabilitas, `venv` membuat semua _environment_ yg digunakan terisolasi, sehingga tidak mengganggu versi python di perangkat anda.

membuat _virtual environment_ dengan nama `env` (akan membuat folder _environment_ yg terisolasi)

```bash
# powershell
python -m venv env
```

Cara mengaktifkan _virtual environment_ di terminal anda sebagai berikut :

```bash
# powershell
.\env\Scripts\activate
```

menginstal semua _package_ Python yang tercantum di dalam file `requirements.txt`

```bash
# powershell mode virtual environment
pip install -r requirements.txt
```

menyimpan daftar semua _package_ Python (beserta versinya) yang saat ini terinstal di _environment_ kamu

```bash
# powershell mode virtual environment
pip freeze > requirements.txt
```

## Status

Versi 2 masih dalam tahap pengembangan/refaktor. Masukan dan kontribusi sangat terbuka.

---

Proyek ini tetap mengacu pada lisensi GNU GPL v3.0 seperti versi sebelumnya.
