
import streamlit as st
import pandas as pd
import numpy as np
import uuid

from datetime import datetime, timezone
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)


# =====================================================
# 1. KONFIGURASI APLIKASI
# =====================================================

st.set_page_config(
    page_title="Prediksi dan Analisis Mahasiswa",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Aplikasi Prediksi & Analisis Mahasiswa")

st.write(
    "Aplikasi Machine Learning untuk melakukan "
    "Regresi Linear, Klasifikasi, Clustering, "
    "dan Retraining pada data mahasiswa."
)

KOLOM_WAJIB = [
    "nim",
    "jenis_kelamin",
    "umur",
    "semester",
    "ipk",
    "kehadiran",
    "nilai_tugas",
    "nilai_uts",
    "nilai_uas",
    "status_kelulusan"
]

FITUR_REGRESI = [
    "kehadiran",
    "nilai_tugas",
    "nilai_uts",
    "nilai_uas"
]

FITUR_KLASIFIKASI = [
    "umur",
    "semester",
    "kehadiran",
    "nilai_tugas",
    "nilai_uts",
    "nilai_uas"
]

FITUR_CLUSTERING = [
    "ipk",
    "kehadiran",
    "nilai_tugas",
    "nilai_uts",
    "nilai_uas"
]


# =====================================================
# 2. KONEKSI MONGODB
# =====================================================

@st.cache_resource
def koneksi_mongodb(uri):
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=10000
    )

    client.admin.command("ping")

    db = client["db_machine_learning"]

    return client, db


db = None
mongo_error = None

try:
    mongo_uri = st.secrets["mongo"]["uri"]

    client, db = koneksi_mongodb(mongo_uri)

    st.sidebar.success("MongoDB berhasil terhubung!")

except Exception as e:
    mongo_error = str(e)

    st.sidebar.error("MongoDB belum terhubung.")

    st.sidebar.caption(
        "Periksa konfigurasi secrets.toml dan koneksi Atlas."
    )


# =====================================================
# 3. FUNGSI DATASET
# =====================================================

def validasi_dataset(data):

    kolom_hilang = [
        kolom for kolom in KOLOM_WAJIB
        if kolom not in data.columns
    ]

    return kolom_hilang


def bersihkan_data(data):

    data = data.copy()

    kolom_numerik = [
        "umur",
        "semester",
        "ipk",
        "kehadiran",
        "nilai_tugas",
        "nilai_uts",
        "nilai_uas"
    ]

    for kolom in kolom_numerik:
        data[kolom] = pd.to_numeric(
            data[kolom],
            errors="coerce"
        )

    return data


def ambil_dataset_mongodb():

    if db is None:
        return None

    koleksi = db["mahasiswa"]

    dokumen_terbaru = koleksi.find_one(
        {},
        sort=[("_id", -1)]
    )

    if dokumen_terbaru is None:
        return None

    batch_id = dokumen_terbaru.get("_batch_id")

    if batch_id:
        dokumen = list(
            koleksi.find(
                {"_batch_id": batch_id},
                {"_id": 0, "_batch_id": 0}
            )
        )
    else:
        dokumen = list(
            koleksi.find(
                {},
                {"_id": 0, "_batch_id": 0}
            )
        )

    if not dokumen:
        return None

    data = pd.DataFrame(dokumen)

    if validasi_dataset(data):
        return None

    return bersihkan_data(data)


# =====================================================
# 4. MEMUAT DATASET
# =====================================================

if "dataset_mahasiswa" not in st.session_state:

    data_awal = None

    try:
        data_awal = ambil_dataset_mongodb()
    except Exception:
        data_awal = None

    if data_awal is None:

        try:
            data_awal = pd.read_csv(
                "dataset_mahasiswa_300.csv"
            )

            if not validasi_dataset(data_awal):
                data_awal = bersihkan_data(data_awal)
            else:
                data_awal = None

        except Exception:
            data_awal = None

    st.session_state["dataset_mahasiswa"] = data_awal


# =====================================================
# 5. UPLOAD DATASET DARI SIDEBAR
# =====================================================

st.sidebar.header("Upload Dataset")

file_upload = st.sidebar.file_uploader(
    "Pilih dataset mahasiswa CSV",
    type=["csv"]
)

if file_upload is not None:

    try:
        data_upload = pd.read_csv(file_upload)

        kolom_hilang = validasi_dataset(data_upload)

        if kolom_hilang:
            st.sidebar.error(
                "Kolom dataset tidak lengkap: "
                + ", ".join(kolom_hilang)
            )

        else:
            st.session_state["dataset_mahasiswa"] = (
                bersihkan_data(data_upload)
            )

            st.sidebar.success(
                "Dataset berhasil dibaca!"
            )

    except Exception as e:
        st.sidebar.error(
            f"Gagal membaca dataset: {e}"
        )


df = st.session_state["dataset_mahasiswa"]


# =====================================================
# 6. INFORMASI DATASET
# =====================================================

st.sidebar.divider()
st.sidebar.subheader("Informasi Dataset")

if df is not None:

    st.sidebar.metric(
        "Jumlah Mahasiswa",
        len(df)
    )

    st.sidebar.metric(
        "Jumlah Kolom",
        len(df.columns)
    )

    st.sidebar.caption(
        "Dataset siap digunakan untuk analisis."
    )

else:

    st.sidebar.warning(
        "Silakan upload dataset mahasiswa CSV."
    )


# =====================================================
# 7. FUNGSI PENYIMPANAN HASIL ANALISIS
# =====================================================

def simpan_hasil(nama_model, metrik):

    if db is None:
        st.error("MongoDB belum terhubung.")
        return

    dokumen = {
        "nama_model": nama_model,
        "metrik": metrik,
        "tanggal": datetime.now(timezone.utc)
    }

    try:
        db["hasil_analisis"].insert_one(dokumen)

        st.success(
            "Hasil analisis berhasil disimpan ke MongoDB."
        )

    except Exception as e:
        st.error(f"Gagal menyimpan hasil: {e}")


# =====================================================
# 8. FUNGSI TRAINING REGRESI LINEAR
# =====================================================

def training_regresi(data):

    data_model = data[
        FITUR_REGRESI + ["ipk"]
    ].copy()

    for kolom in FITUR_REGRESI + ["ipk"]:
        data_model[kolom] = pd.to_numeric(
            data_model[kolom],
            errors="coerce"
        )

    data_model = data_model.dropna()

    if len(data_model) < 5:
        raise ValueError(
            "Data regresi tidak cukup untuk training."
        )

    X = data_model[FITUR_REGRESI]
    y = data_model["ipk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    prediksi = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediksi)
    mse = mean_squared_error(y_test, prediksi)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediksi)

    hasil = pd.DataFrame({
        "IPK Aktual": y_test.values,
        "IPK Prediksi": prediksi
    })

    metrik = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    return model, metrik, hasil


# =====================================================
# 9. FUNGSI TRAINING KLASIFIKASI
# =====================================================

def training_klasifikasi(data):

    data_model = data[
        FITUR_KLASIFIKASI + ["status_kelulusan"]
    ].copy()

    for kolom in FITUR_KLASIFIKASI:
        data_model[kolom] = pd.to_numeric(
            data_model[kolom],
            errors="coerce"
        )

    data_model["status_kelulusan"] = (
        data_model["status_kelulusan"].astype("string")
    )

    data_model = data_model.dropna()

    if len(data_model) < 5:
        raise ValueError(
            "Data klasifikasi tidak cukup."
        )

    X = data_model[FITUR_KLASIFIKASI]

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        data_model["status_kelulusan"]
    )

    if len(encoder.classes_) < 2:
        raise ValueError(
            "Status kelulusan harus memiliki minimal dua kelas."
        )

    jumlah_kelas = pd.Series(y).value_counts()

    stratifikasi = (
        y if jumlah_kelas.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratifikasi
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Data training harus memiliki minimal dua kelas."
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )

    model.fit(X_train, y_train)

    prediksi = model.predict(X_test)

    akurasi = accuracy_score(y_test, prediksi)

    presisi = precision_score(
        y_test,
        prediksi,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediksi,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediksi,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        prediksi,
        labels=np.arange(len(encoder.classes_))
    )

    laporan = classification_report(
        y_test,
        prediksi,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    hasil = pd.DataFrame({
        "Status Aktual": encoder.inverse_transform(y_test),
        "Status Prediksi": encoder.inverse_transform(prediksi)
    })

    metrik = {
        "Accuracy": float(akurasi),
        "Precision": float(presisi),
        "Recall": float(recall),
        "F1 Score": float(f1)
    }

    return model, encoder, metrik, cm, laporan, hasil


# =====================================================
# 10. FUNGSI TRAINING CLUSTERING
# =====================================================

def training_clustering(data, jumlah_cluster=3):

    data_model = data[FITUR_CLUSTERING].copy()

    for kolom in FITUR_CLUSTERING:
        data_model[kolom] = pd.to_numeric(
            data_model[kolom],
            errors="coerce"
        )

    data_model = data_model.dropna()

    if len(data_model) <= jumlah_cluster:
        raise ValueError(
            "Jumlah data harus lebih banyak daripada jumlah cluster."
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(data_model)

    model = KMeans(
        n_clusters=jumlah_cluster,
        random_state=42,
        n_init=10
    )

    label = model.fit_predict(X_scaled)

    hasil = data_model.copy()
    hasil["Cluster"] = label

    silhouette = silhouette_score(
        X_scaled,
        label
    )

    rata_rata = (
        hasil.groupby("Cluster")[FITUR_CLUSTERING]
        .mean()
    )

    metrik = {
        "Jumlah Cluster": int(jumlah_cluster),
        "Silhouette Score": float(silhouette)
    }

    return model, scaler, metrik, hasil, rata_rata


# =====================================================
# 11. MEMBUAT ENAM MENU
# =====================================================

tab_regresi, tab_klasifikasi, tab_clustering, tab_dataset, tab_retraining, tab_komentar = st.tabs([
    "Regresi Linear",
    "Klasifikasi",
    "Clustering",
    "Dataset MongoDB",
    "Retraining",
    "Komentar Publik"
])


# =====================================================
# 12. MENU REGRESI LINEAR
# =====================================================

with tab_regresi:

    st.header("📈 Analisis Regresi Linear")

    st.write(
        "Memprediksi IPK mahasiswa berdasarkan kehadiran "
        "dan nilai akademik."
    )

    if df is not None:

        st.write("Fitur:", FITUR_REGRESI)
        st.write("Target: ipk")

        if st.button(
            "Mulai Training Regresi Linear",
            key="btn_regresi"
        ):

            try:
                model, metrik, hasil = training_regresi(df)

                st.session_state["hasil_regresi"] = (
                    model, metrik, hasil
                )

            except Exception as e:
                st.error(f"Training gagal: {e}")

        if "hasil_regresi" in st.session_state:

            model, metrik, hasil = (
                st.session_state["hasil_regresi"]
            )

            st.success("Training Regresi Linear berhasil!")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("MAE", f"{metrik['MAE']:.4f}")
            c2.metric("MSE", f"{metrik['MSE']:.4f}")
            c3.metric("RMSE", f"{metrik['RMSE']:.4f}")
            c4.metric("R²", f"{metrik['R2']:.4f}")

            st.subheader("Hasil Prediksi IPK")
            st.dataframe(hasil, use_container_width=True)

            if st.button(
                "Simpan Hasil Regresi ke MongoDB",
                key="simpan_regresi"
            ):
                simpan_hasil(
                    "Linear Regression",
                    metrik
                )

    else:
        st.warning("Upload dataset mahasiswa terlebih dahulu.")


# =====================================================
# 13. MENU KLASIFIKASI
# =====================================================

with tab_klasifikasi:

    st.header("🎯 Analisis Klasifikasi Logistic Regression")

    st.write(
        "Memprediksi status kelulusan mahasiswa "
        "berdasarkan karakteristik akademik."
    )

    if df is not None:

        st.write("Fitur:", FITUR_KLASIFIKASI)
        st.write("Target: status_kelulusan")

        if st.button(
            "Mulai Training Klasifikasi",
            key="btn_klasifikasi"
        ):

            try:
                hasil_training = training_klasifikasi(df)

                st.session_state["hasil_klasifikasi"] = (
                    hasil_training
                )

            except Exception as e:
                st.error(f"Training gagal: {e}")

        if "hasil_klasifikasi" in st.session_state:

            model, encoder, metrik, cm, laporan, hasil = (
                st.session_state["hasil_klasifikasi"]
            )

            st.success("Training Klasifikasi berhasil!")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{metrik['Accuracy']:.2%}")
            c2.metric("Precision", f"{metrik['Precision']:.2%}")
            c3.metric("Recall", f"{metrik['Recall']:.2%}")
            c4.metric("F1 Score", f"{metrik['F1 Score']:.2%}")

            st.subheader("Hasil Prediksi")
            st.dataframe(hasil, use_container_width=True)

            st.subheader("Confusion Matrix")

            st.dataframe(
                pd.DataFrame(
                    cm,
                    index=encoder.classes_,
                    columns=encoder.classes_
                ),
                use_container_width=True
            )

            st.subheader("Classification Report")

            st.dataframe(
                pd.DataFrame(laporan).transpose(),
                use_container_width=True
            )

            if st.button(
                "Simpan Hasil Klasifikasi ke MongoDB",
                key="simpan_klasifikasi"
            ):
                simpan_hasil(
                    "Logistic Regression",
                    metrik
                )

    else:
        st.warning("Upload dataset mahasiswa terlebih dahulu.")


# =====================================================
# 14. MENU CLUSTERING
# =====================================================

with tab_clustering:

    st.header("👥 Analisis Clustering K-Means")

    st.write(
        "Mengelompokkan mahasiswa berdasarkan "
        "IPK dan nilai akademik."
    )

    if df is not None:

        jumlah_cluster = st.slider(
            "Jumlah Cluster",
            min_value=2,
            max_value=5,
            value=3
        )

        if st.button(
            "Mulai Training K-Means",
            key="btn_clustering"
        ):

            try:
                hasil_training = training_clustering(
                    df,
                    jumlah_cluster
                )

                st.session_state["hasil_clustering"] = (
                    hasil_training
                )

            except Exception as e:
                st.error(f"Clustering gagal: {e}")

        if "hasil_clustering" in st.session_state:

            model, scaler, metrik, hasil, rata_rata = (
                st.session_state["hasil_clustering"]
            )

            st.success("Training K-Means berhasil!")

            st.metric(
                "Silhouette Score",
                f"{metrik['Silhouette Score']:.4f}"
            )

            st.subheader("Hasil Pengelompokan")

            st.dataframe(
                hasil,
                use_container_width=True
            )

            st.subheader("Jumlah Mahasiswa per Cluster")

            st.bar_chart(
                hasil["Cluster"].value_counts().sort_index()
            )

            st.subheader("Rata-rata Setiap Cluster")

            st.dataframe(
                rata_rata,
                use_container_width=True
            )

            st.subheader("Visualisasi Cluster")

            st.scatter_chart(
                hasil,
                x="kehadiran",
                y="ipk",
                color="Cluster"
            )

            if st.button(
                "Simpan Hasil Clustering ke MongoDB",
                key="simpan_clustering"
            ):
                simpan_hasil(
                    "K-Means",
                    metrik
                )

    else:
        st.warning("Upload dataset mahasiswa terlebih dahulu.")


# =====================================================
# 15. MENU DATASET MONGODB
# =====================================================

with tab_dataset:

    st.header("🗄️ Penyimpanan Dataset ke MongoDB")

    st.write(
        "Upload dataset dari sidebar, kemudian simpan "
        "ke MongoDB Atlas."
    )

    if df is not None:

        st.subheader("Preview Dataset")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.write("Jumlah Baris:", len(df))
        st.write("Jumlah Kolom:", len(df.columns))

        if db is not None:

            if st.button(
                "Simpan Dataset ke MongoDB",
                key="simpan_dataset"
            ):

                try:
                    batch_id = str(uuid.uuid4())

                    records = df.to_dict(
                        orient="records"
                    )

                    for record in records:
                        record["_batch_id"] = batch_id

                    db["mahasiswa"].insert_many(records)

                    db["riwayat_dataset"].insert_one({
                        "batch_id": batch_id,
                        "jumlah_data": len(records),
                        "nama_dataset": "Dataset Mahasiswa",
                        "tanggal_upload": datetime.now(
                            timezone.utc
                        )
                    })

                    st.success(
                        f"Berhasil menyimpan {len(records)} "
                        "data mahasiswa ke MongoDB."
                    )

                    st.info(
                        "Dataset tersimpan. Hindari menekan "
                        "tombol simpan berulang kali agar tidak "
                        "membuat data duplikat."
                    )

                except Exception as e:
                    st.error(
                        f"Gagal menyimpan dataset: {e}"
                    )

        else:
            st.warning("MongoDB belum terhubung.")

    else:
        st.warning(
            "Silakan upload dataset CSV melalui sidebar."
        )


# =====================================================
# 16. MENU RETRAINING
# =====================================================

with tab_retraining:

    st.header("🔄 Retraining Machine Learning")

    st.write(
        "Retraining digunakan untuk melatih ulang "
        "ketiga model menggunakan dataset yang sedang aktif."
    )

    if df is not None:

        st.info(
            f"Dataset aktif berisi {len(df)} baris data mahasiswa."
        )

        st.write(
            "Retraining akan menjalankan Regresi Linear, "
            "Logistic Regression, dan K-Means."
        )

        jumlah_cluster_retraining = st.slider(
            "Jumlah Cluster untuk Retraining",
            min_value=2,
            max_value=5,
            value=3,
            key="cluster_retraining"
        )

        if st.button(
            "🚀 Jalankan Retraining Semua Model",
            key="btn_retraining"
        ):

            hasil_retraining = {}

            try:
                model, metrik, hasil = training_regresi(df)

                hasil_retraining["Regresi Linear"] = metrik

                st.session_state["hasil_regresi"] = (
                    model, metrik, hasil
                )

            except Exception as e:
                hasil_retraining["Regresi Linear"] = {
                    "error": str(e)
                }

            try:
                hasil_training = training_klasifikasi(df)

                model, encoder, metrik, cm, laporan, hasil = (
                    hasil_training
                )

                hasil_retraining["Logistic Regression"] = metrik

                st.session_state["hasil_klasifikasi"] = (
                    hasil_training
                )

            except Exception as e:
                hasil_retraining["Logistic Regression"] = {
                    "error": str(e)
                }

            try:
                hasil_training = training_clustering(
                    df,
                    jumlah_cluster_retraining
                )

                model, scaler, metrik, hasil, rata_rata = (
                    hasil_training
                )

                hasil_retraining["K-Means"] = metrik

                st.session_state["hasil_clustering"] = (
                    hasil_training
                )

            except Exception as e:
                hasil_retraining["K-Means"] = {
                    "error": str(e)
                }

            st.session_state["hasil_retraining"] = (
                hasil_retraining
            )

        if "hasil_retraining" in st.session_state:

            st.subheader("Hasil Retraining")

            hasil_retraining = (
                st.session_state["hasil_retraining"]
            )

            for nama, metrik in hasil_retraining.items():

                st.markdown(f"**{nama}**")

                if "error" in metrik:
                    st.error(metrik["error"])
                else:
                    st.json(metrik)

            if st.button(
                "Simpan Hasil Retraining ke MongoDB",
                key="simpan_retraining"
            ):

                if db is not None:

                    try:
                        db["hasil_analisis"].insert_one({
                            "nama_model": "Retraining Semua Model",
                            "hasil": hasil_retraining,
                            "tanggal": datetime.now(
                                timezone.utc
                            )
                        })

                        st.success(
                            "Hasil retraining berhasil disimpan."
                        )

                    except Exception as e:
                        st.error(
                            f"Gagal menyimpan hasil: {e}"
                        )

                else:
                    st.error("MongoDB belum terhubung.")

    else:
        st.warning("Upload dataset terlebih dahulu.")


# =====================================================
# 17. MENU KOMENTAR PUBLIK
# =====================================================

with tab_komentar:

    st.header("💬 Komentar Publik")

    st.write(
        "Pengunjung dapat memberikan komentar, "
        "saran, dan masukan terhadap aplikasi."
    )

    with st.form("form_komentar"):

        nama = st.text_input(
            "Nama Pengunjung"
        )

        komentar = st.text_area(
            "Tuliskan Komentar"
        )

        kirim = st.form_submit_button(
            "Kirim Komentar"
        )

        if kirim:

            if nama.strip() and komentar.strip():

                if db is not None:

                    try:
                        db["komentar_publik"].insert_one({
                            "nama": nama.strip(),
                            "komentar": komentar.strip(),
                            "tanggal": datetime.now(
                                timezone.utc
                            )
                        })

                        st.success(
                            "Komentar berhasil disimpan!"
                        )

                    except Exception as e:
                        st.error(
                            f"Gagal menyimpan komentar: {e}"
                        )

                else:
                    st.error(
                        "MongoDB belum terhubung. "
                        "Komentar belum dapat disimpan."
                    )

            else:
                st.warning(
                    "Nama dan komentar wajib diisi."
                )

    st.divider()

    st.subheader("Daftar Komentar Pengunjung")

    if db is not None:

        try:
            daftar_komentar = list(
                db["komentar_publik"].find(
                    {},
                    {"_id": 0}
                ).sort("_id", -1).limit(100)
            )

            if daftar_komentar:

                for item in daftar_komentar:

                    with st.container(border=True):

                        st.write(
                            f"**{item.get('nama', 'Pengunjung')}**"
                        )

                        st.write(
                            item.get("komentar", "")
                        )

                        tanggal = item.get("tanggal")

                        if tanggal:
                            st.caption(str(tanggal))

            else:
                st.info(
                    "Belum ada komentar dari pengunjung."
                )

        except Exception as e:
            st.error(
                f"Gagal mengambil komentar: {e}"
            )

    else:
        st.warning(
            "Hubungkan MongoDB untuk menampilkan komentar."
        )


# =====================================================
# 18. FOOTER
# =====================================================

st.divider()

st.caption(
    "Aplikasi Prediksi & Analisis Mahasiswa | "
    "Streamlit, Scikit-learn, dan MongoDB Atlas"
)
