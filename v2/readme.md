# Deteksi Keliling Luka Kronis v2 (Refactor)

Versi 2 ini merupakan refaktor dari implementasi sebelumnya untuk deteksi keliling luka kronis menggunakan _Active Contour (Snake)_ dan variannya.

## Motivasi Refaktor

- Merapikan struktur kode agar lebih mudah dipahami dan dikembangkan
- Modularisasi dan pemisahan fungsi-fungsi utama
- Penambahan dokumentasi dan komentar kode
- Memudahkan penambahan fitur atau pengujian metode baru

## Struktur Folder

```bash
v2/
|
|- data/                            # Folder untuk menyimpan seluruh dataset
|
|- notebooks/                       # Jupyter Notebook untuk eksperimen, testing, dan catatan
|   |- 01_acm_integer.ipynb
|   |- 02_acm_interpolation.ipynb
|
|- src/                             # Folder kode sumber utama
|   |- __init__.py                  # Menandai src sebagai package
|   |- utils.py                     # Fungsi-fungsi pembantu (reusable)
|   |- active_contour.py            # Fungsi-fungsi utama untuk deteksi keliling menggunakan active contour (snake)
|   |- processing.py                # Fungsi-fungsi pemrosesan snake
|
|- main.py                          # Entry point utama
|- requirements.txt                 # Daftar dependency
|- readme.md                        # Penjelasan proyek
```

## Struktur folder dataset

```bash
|- data/
|   |- luka_hitam/                            # Folder dataset gambar luka dengan karakteristik warna hitam
|       |- images/                            # Subfolder berisi file gambar yg siap di proses
|       |   |- 5.jpg                          # Contoh nama file gambar luka
|       |
|       |- output/                            # Subfolder output
|       |   |- integer/                       # Subfolder berisi file output hasil deteksi menggunakan acm integer
|       |   |   |- 5.jpg                      # Contoh nama file gambar hasil deteksi
|       |   |
|       |   |- interpolation/                 # Subfolder berisi file output hasil deteksi menggunakan acm interpolasi
|       |
|       |- validation/                        # Subfolder validasi
|       |   |- integer /                      # Subfolder berisi file output hasil deteksi menggunakan acm integer
|       |   |   |- 5_val.jpg                  # Contoh nama file gambar hasil validasi (berdasarkan region)
|       |   |   |- 5_region_result.jpg        # Contoh nama file gambar region kurva hasil deteksi
|       |   |   |- 5_region_groundtruth.jpg   # Contoh nama file gambar region kurva groundtruth
|       |   |
|       |   |- interpolation                  # Subfolder berisi file output hasil deteksi menggunakan acm interpolasi
|       |
|       |- integer.csv                        # File CSV berisi parameter dan hasil deteksi dengan acm integer
|       |- interpolation.csv                  # File CSV berisi parameter dan hasil deteksi dengan interpolasi
|
|   |- luka_kuning/                           # Folder dataset gambar luka dengan karakteristik warna kuning
|   |- luka_merah/                            # Folder dataset gambar luka dengan karakteristik warna merah
```

## Petunjuk penggunaan

Project ini menggunakan `venv` untuk mengelola _virtual environment_ dengan tujuan kompabilitas, `venv` membuat semua _environment_ yg digunakan terisolasi, sehingga tidak mengganggu versi python di perangkat anda.

membuat _virtual environment_ dengan nama `env` (akan membuat folder _environment_ yg terisolasi)

```bash
# powershell
python -m venv env
```

Cara mengaktifkan _virtual environment_ di terminal sebagai berikut :

```bash
# powershell
.\env\Scripts\activate
```

menginstal semua _package_ Python yang tercantum di dalam file `requirements.txt`

```bash
# powershell mode virtual environment
pip install -r requirements.txt
```

menyimpan daftar semua _package_ Python (beserta versinya) yang saat ini terinstal di _environment_

```bash
# powershell mode virtual environment
pip freeze > requirements.txt
```

## Status

Versi 2 masih dalam tahap pengembangan/refaktor. Masukan dan kontribusi sangat terbuka.

---

Proyek ini tetap mengacu pada lisensi GNU GPL v3.0 seperti versi sebelumnya.
