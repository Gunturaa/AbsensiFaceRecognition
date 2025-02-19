# Sistem Absensi Wajah Berbasis Website

Proyek ini adalah sistem absensi sederhana yang menggunakan teknologi pengenalan wajah (face recognition) berbasis website. Sistem ini dibangun dengan menggunakan Python, OpenCV, Flask, dan library `face_recognition`. Proyek ini memungkinkan pengguna untuk melakukan absensi dengan mendeteksi wajah mereka melalui kamera.

## Fitur
- **Pendaftaran Wajah**: Pengguna dapat mendaftarkan wajah mereka ke sistem.
- **Absensi Otomatis**: Sistem akan mengenali wajah yang sudah terdaftar dan mencatat absensi secara otomatis.
- **Antarmuka Website**: Pengguna dapat mengakses sistem melalui antarmuka website yang dibangun dengan Flask.
- **Riwayat Absensi**: Menyimpan dan menampilkan riwayat absensi pengguna.

## Teknologi yang Digunakan
- **Python**: Bahasa pemrograman utama yang digunakan.
- **OpenCV**: Library untuk pengolahan gambar dan video.
- **Flask**: Framework web untuk membangun antarmuka website.
- **face_recognition**: Library untuk pengenalan wajah.
- **SQLite**: Database untuk menyimpan data pengguna dan riwayat absensi.

## Instalasi

1. **Clone Repository**
   ```bash
   git clone https://github.com/username/repository-name.git
   cd repository-name

   python -m venv venv
   source venv/bin/activate  # Untuk Windows: venv\Scripts\activate
   python app.py
