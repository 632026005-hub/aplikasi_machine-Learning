
import pandas as pd
import streamlit as st
from pymongo import MongoClient

# 1. Koneksi ke MongoDB Atlas
client = MongoClient(
    st.secrets["MONGO_URI"],
    serverSelectionTimeoutMS=10000
)

# 2. Pilih database dan collection
db = client["db_machine_learning"]
coll = db["mahasiswa"]

# 3. Periksa koneksi
client.admin.command("ping")
print("MongoDB berhasil terhubung!")

# 4. Baca dataset CSV
df = pd.read_csv("dataset.csv")

# 5. Periksa dataset
print("Jumlah data:", len(df))
print("Kolom:", df.columns.tolist())

# 6. Masukkan data ke MongoDB
records = df.to_dict(orient="records")

if records:
    coll.delete_many({})
    coll.insert_many(records)

print(f"Berhasil memasukkan {len(records)} data mahasiswa!")
