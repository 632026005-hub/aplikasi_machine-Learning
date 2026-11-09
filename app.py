import streamlit as st
import pandas as pd
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

# 1. Koneksi ke MongoDB
client = MongoClient(
    st.secrets["mongo"]["uri"],
    serverSelectionTimeoutMS=10000
)

db = client["db_machine_learning"]

coll_data = db["mahasiswa"]
coll_comments = db["komentar_publik"]

# Tes koneksi MongoDB
try:
    client.admin.command("ping")
    st.success("MongoDB berhasil terhubung!")
except Exception as e:
    st.error(f"Koneksi MongoDB gagal: {e}")
    st.stop()

# Streamlit Title
st.title("🎓 Aplikasi Prediksi & Analisis Mahasiswa")

# Sidebar Menu
menu = st.sidebar.selectbox("Pilih Menu", [
    "Data & Overview", 
    "Regresi Linear (Nilai Ujian)", 
    "Klasifikasi (Kelulusan)", 
    "Clustering (Pengelompokan)", 
    "Tambah Data (Auto Retrain)", 
    "Komentar Publik"
])

# Ambil Data dari MongoDB
def load_data():
    data = list(coll_data.find({}, {"_id": 0}))
    return pd.DataFrame(data)

df = load_data()
st.write("Kolom yang ditemukan:", df.columns.tolist())
st.write("Jumlah data:", len(df))
st.dataframe(df)
if df.empty:
    st.warning("Data mahasiswa belum tersedia di MongoDB.")
    st.stop()
    
# -------------------------------------------------------------
# MENU 1: DATA & OVERVIEW
# -------------------------------------------------------------
if menu == "Data & Overview":
    st.header("📊 Dataset Mahasiswa dari MongoDB")
    st.write(f"Total Record: **{len(df)}** data")
    st.dataframe(df)

# -------------------------------------------------------------
# MENU 2: REGRESI LINEAR
# -------------------------------------------------------------
elif menu == "Regresi Linear (Nilai Ujian)":
    st.header("📈 Prediksi Nilai Ujian (Regresi Linear)")
    X = df[['jam_belajar', 'kehadiran_persen']]
    y = df['nilai_ujian']
    
    model = LinearRegression()
    model.fit(X, y)
    
    jam = st.number_input("Jam Belajar per Hari", min_value=0.0, max_value=24.0, value=5.0)
    kehadiran = st.slider("Persentase Kehadiran (%)", 0, 100, 80)
    
    if st.button("Prediksi Nilai"):
        pred = model.predict([[jam, kehadiran]])[0]
        st.success(f"Estimasi Nilai Ujian: **{pred:.2f}**")

# -------------------------------------------------------------
# MENU 3: KLASIFIKASI
# -------------------------------------------------------------
elif menu == "Klasifikasi (Kelulusan)":
    st.header("🎯 Prediksi Status Kelulusan (Decision Tree)")
    X = df[['jam_belajar', 'kehadiran_persen', 'nilai_ujian']]
    y = df['status_kelulusan']
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    jam = st.number_input("Jam Belajar", min_value=0.0, max_value=24.0, value=6.0)
    kehadiran = st.slider("Kehadiran (%)", 0, 100, 85)
    nilai = st.number_input("Nilai Ujian", min_value=0.0, max_value=100.0, value=75.0)
    
    if st.button("Cek Kelulusan"):
        pred = model.predict([[jam, kehadiran, nilai]])[0]
        if pred == "Lulus":
            st.balloons()
            st.success(f"Hasil Prediksi: **{pred}** 🎉")
        else:
            st.error(f"Hasil Prediksi: **{pred}** ⚠️")

# -------------------------------------------------------------
# MENU 4: CLUSTERING
# -------------------------------------------------------------
elif menu == "Clustering (Pengelompokan)":
    st.header("📌 Pengelompokan Mahasiswa (K-Means)")
    k = st.slider("Pilih Jumlah Cluster (K)", 2, 5, 3)
    
    X = df[['jam_belajar', 'kehadiran_persen', 'nilai_ujian']]
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
    
    df_cluster = df.copy()
    df_cluster['Cluster'] = kmeans.labels_
    st.dataframe(df_cluster)

# -------------------------------------------------------------
# MENU 5: TAMBAH DATA (AUTO RETRAIN)
# -------------------------------------------------------------
elif menu == "Tambah Data (Auto Retrain)":
    st.header("➕ Tambah Data Mahasiswa Baru")
    st.caption("Data yang ditambahkan akan tersimpan ke MongoDB dan otomatis memperbarui model machine learning.")
    
    with st.form("form_tambah_data"):
        jam = st.number_input("Jam Belajar", min_value=0.0, max_value=24.0, value=5.0)
        kehadiran = st.number_input("Kehadiran (%)", min_value=0, max_value=100, value=80)
        nilai = st.number_input("Nilai Ujian", min_value=0.0, max_value=100.0, value=70.0)
        status = st.selectbox("Status Kelulusan", ["Lulus", "Tidak Lulus"])
        
        submitted = st.form_submit_button("Simpan Data")
        if submitted:
            new_record = {
                "jam_belajar": jam,
                "kehadiran_persen": kehadiran,
                "nilai_ujian": nilai,
                "status_kelulusan": status
            }
            coll_data.insert_one(new_record)
            st.success("✅ Data berhasil disimpan ke MongoDB! Model otomatis ter-retrain.")
            st.rerun()

# -------------------------------------------------------------
# MENU 6: KOMENTAR PUBLIK
# -------------------------------------------------------------
elif menu == "Komentar Publik":
    st.header("💬 Form Komentar & Masukan Publik")
    
    with st.form("form_komentar"):
        nama = st.text_input("Nama")
        komentar = st.text_area("Komentar / Saran")
        submit_kategori = st.form_submit_button("Kirim Komentar")
        
        if submit_kategori and nama and komentar:
            coll_comments.insert_one({"nama": nama, "komentar": komentar})
            st.success("Komentar Anda berhasil dikirim!")
            st.rerun()
            
    st.subheader("Daftar Komentar:")
    list_komentar = list(coll_comments.find({}, {"_id": 0}))
    if list_komentar:
        for item in list_komentar:
            st.info(f"**{item['nama']}**: {item['komentar']}")
    else:
        st.write("Belum ada komentar.")
