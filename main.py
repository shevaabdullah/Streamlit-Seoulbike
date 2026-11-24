import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Konfigurasi halaman agar lebih luas (Wide Layout)
st.set_page_config(page_title="Seoul Bike AI Dashboard", page_icon="🚲", layout="wide")

# ==========================================
# MEMUAT DATA & MELATIH MODEL AI (CACHED)
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv('SeoulBikeData.csv', encoding='latin1')

@st.cache_resource
def train_model(data):
    df_ml = data[['Hour', 'Temperature(°C)', 'Wind speed (m/s)', 'Rainfall(mm)', 'Seasons', 'Rented Bike Count']].copy()
    df_ml = pd.get_dummies(df_ml, columns=['Seasons'])
    
    X = df_ml.drop('Rented Bike Count', axis=1)
    y = df_ml['Rented Bike Count']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    skor_r2 = model.score(X, y)
    return model, X.columns, skor_r2

df = load_data()
model, feature_cols, skor_r2 = train_model(df)

def prediksi_sepeda(jam, suhu, angin, hujan_mm, musim):
    input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
    input_data['Hour'] = jam
    input_data['Temperature(°C)'] = suhu
    input_data['Wind speed (m/s)'] = angin
    input_data['Rainfall(mm)'] = hujan_mm
    
    kolom_musim = f"Seasons_{musim}"
    if kolom_musim in input_data.columns:
        input_data[kolom_musim] = 1
        
    hasil = model.predict(input_data)[0]
    return max(0, int(round(hasil)))

# ==========================================
# MENU NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("🚲 Menu Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Simulasi Prediksi", "Tentang"])
st.sidebar.info(f"Akurasi Model Latih ($R^2$): **{skor_r2*100:.1f}%**")

st.title('Sistem Prediksi Seoul Bike Sharing')
st.write(f"Anda sedang berada di halaman: **{menu}**")
st.divider()

# ==========================================
# HALAMAN 1: DASHBOARD (EDA, FILTER, KPI & GRAFIK)
# ==========================================
if menu == "Dashboard":
    st.header('Dashboard Analisis Sepeda Seoul')
    st.subheader('1. Eksplorasi Data (EDA)')
    st.caption('Data direkam dari stasiun penyewaan sepeda publik di Seoul selama 1 tahun.')

    st.markdown("Variabel **Suhu Udara** memiliki korelasi yang sangat :blue[tinggi] terhadap jumlah sewa.")
    st.write("Transformasi logaritmik yang kita gunakan untuk target (Rented Bike Count):")
    st.latex(r''' y' = \log(1 + y) ''')

    st.divider()

    # FITUR BARU 1: Filter Musim Dinamis untuk KPI & Data
    st.subheader("Filter & Ringkasan Operasional Dataset")
    pilihan_musim_dash = st.selectbox(
        "Filter Data Berdasarkan Musim:",
        ["Semua Musim", "Spring", "Summer", "Autumn", "Winter"]
    )

    if pilihan_musim_dash == "Semua Musim":
        df_filtered = df
    else:
        df_filtered = df[df['Seasons'] == pilihan_musim_dash]

    # Menghitung KPI nyata dari data yang difilter
    rata_sewa = int(df_filtered['Rented Bike Count'].mean())
    rata_all = int(df['Rented Bike Count'].mean())
    selisih_sewa = rata_sewa - rata_all

    rata_suhu = df_filtered['Temperature(°C)'].mean()
    rata_hujan = df_filtered['Rainfall(mm)'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Rata-rata Sewa / Jam",
        value=f"{rata_sewa:,} unit",
        delta=f"{selisih_sewa:+,} vs rata-rata tahunan"
    )
    col2.metric(
        label="Suhu Rata-rata",
        value=f"{rata_suhu:.1f} °C",
        delta=f"{pilihan_musim_dash}"
    )
    col3.metric(
        label="Rata-rata Curah Hujan",
        value=f"{rata_hujan:.2f} mm",
        delta="Cerah" if rata_hujan < 0.2 else "Cenderung Basah",
        delta_color="normal"
    )

    st.subheader(f'Tinjauan Dataset ({pilihan_musim_dash} - 5 Baris Pertama)')
    st.dataframe(df_filtered.head(), use_container_width=True)

    # FITUR BARU 2: Tombol Download Data CSV
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Unduh Data ({pilihan_musim_dash}) dalam Format CSV",
        data=csv_data,
        file_name=f"SeoulBike_{pilihan_musim_dash}.csv",
        mime="text/csv"
    )

    st.divider()

    # Visualisasi Utama
    hourly_trend = df_filtered.groupby('Hour')['Rented Bike Count'].mean()
    st.subheader(f'Pola Permintaan Sepeda per Jam ({pilihan_musim_dash})')
    st.line_chart(hourly_trend)

    season_trend = df.groupby('Seasons')['Rented Bike Count'].mean()
    st.subheader('Perbandingan Rata-rata Penyewaan Antar Musim')
    st.bar_chart(season_trend)

    st.divider()

    col_kiri, col_kanan = st.columns(2)
    with col_kiri:
        st.info("Grafik Cuaca: Rata-rata Suhu per Jam (°C)")
        st.line_chart(df_filtered.groupby('Hour')['Temperature(°C)'].mean())
    with col_kanan:
        st.success("Grafik Penyewaan: Rata-rata Sewa per Jam")
        st.bar_chart(df_filtered.groupby('Hour')['Rented Bike Count'].mean())

    st.divider()

    tab1, tab2 = st.tabs(["Visualisasi Suhu", "Visualisasi Kelembapan"])
    with tab1:
        st.write("Grafik hubungan antara Suhu Udara (°C) dan rata-rata penyewaan sepeda:")
        st.line_chart(df_filtered.groupby(df_filtered['Temperature(°C)'].round())['Rented Bike Count'].mean())
    with tab2:
        st.write("Grafik hubungan antara Kelembapan (%) dan rata-rata penyewaan sepeda:")
        st.area_chart(df_filtered.groupby('Humidity(%)')['Rented Bike Count'].mean())

    with st.expander("Lihat Detail Penjelasan Dataset"):
        st.write(f"Dataset aktif saat ini menampilkan **{len(df_filtered):,} baris data** operasional per jam dengan **{len(df.columns)} variabel** utama.")

# ==========================================
# HALAMAN 2: SIMULASI PREDIKSI AI
# ==========================================
elif menu == "Simulasi Prediksi":
    st.subheader('Panel Simulasi Input Prediksi & Proyeksi 24 Jam')
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        musim = st.selectbox("Pilih Kondisi Musim", ("Spring", "Summer", "Autumn", "Winter"))
        hujan = st.radio("Apakah sedang hujan?", ("Tidak Hujan", "Hujan"))
    with col_in2:
        suhu = st.slider("Atur Suhu Udara (°C)", min_value=-20.0, max_value=40.0, value=25.0, step=0.5)
        angin = st.number_input("Kecepatan Angin (m/s)", min_value=0.0, max_value=10.0, value=1.5)

    st.write(f"Parameter aktif: **Musim {musim}**, **{hujan}**, **Suhu {suhu}°C**, **Angin {angin} m/s**")

    if st.button("Hitung Prediksi & Proyeksi Harian"):
        curah_hujan = 2.5 if hujan == "Hujan" else 0.0
        estimasi = prediksi_sepeda(jam=18, suhu=suhu, angin=angin, hujan_mm=curah_hujan, musim=musim)
        
        st.success(f"Hasil Simulasi Jam Sibuk (18:00): Estimasi permintaan sekitar **{estimasi:,} unit sepeda**.")
        
        # FITUR BARU 3: Status Rekomendasi Operasional Otomatis
        if estimasi >= 1200:
            st.warning("⚠️ **Status Siaga Puncak:** Permintaan sangat tinggi! Disarankan memobilisasi seluruh armada cadangan ke stasiun komuter.")
        elif estimasi >= 500:
            st.info("ℹ️ **Status Operasional Normal:** Kebutuhan armada berada pada tingkat menengah dan stabil.")
        else:
            st.error("❄️ **Status Permintaan Rendah:** Cuaca/musim kurang mendukung, cukup siapkan armada minimum untuk efisiensi perawatan.")

        # FITUR BARU 4: Grafik Proyeksi Prediksi Sepanjang 24 Jam
        proyeksi_24jam = [
            prediksi_sepeda(jam=j, suhu=suhu, angin=angin, hujan_mm=curah_hujan, musim=musim)
            for j in range(24)
        ]
        df_proyeksi = pd.DataFrame({
            "Jam": range(24),
            "Estimasi Permintaan (Unit)": proyeksi_24jam
        }).set_index("Jam")
        
        st.write(f"**Proyeksi Kebutuhan Sepeda Sepanjang 24 Jam ({musim}, {hujan}, {suhu}°C):**")
        st.area_chart(df_proyeksi)

    st.divider()

    # Formulir Operasional Spesifik
    st.header("Formulir Operasional Sepeda")
    with st.form("form_prediksi"):
        jam = st.number_input("Jam Operasional (0-23)", min_value=0, max_value=23, value=8)
        suhu_form = st.slider("Suhu Udara (°C)", -10.0, 40.0, 20.0)
        musim_form = st.selectbox("Musim Operasional", ("Summer", "Spring", "Autumn", "Winter"))
        submit = st.form_submit_button("Jalankan Prediksi AI")
        
    if submit:
        rekomendasi = prediksi_sepeda(jam=jam, suhu=suhu_form, angin=1.5, hujan_mm=0.0, musim=musim_form)
        st.success(f"Sistem merekomendasikan penyiapan **{rekomendasi:,} armada sepeda** pada pukul **{jam}:00** di musim **{musim_form}**.")

    st.divider()

    # FITUR BARU 5: Transparansi Model (Feature Importance)
    with st.expander("📊 Lihat Faktor Paling Berpengaruh Menurut Model AI (Feature Importance)"):
        st.write("Grafik di bawah menunjukkan seberapa besar bobot setiap variabel dalam menentukan hasil prediksi Random Forest:")
        fi_df = pd.DataFrame({
            "Variabel": feature_cols,
            "Tingkat Pengaruh": model.feature_importances_
        }).sort_values(by="Tingkat Pengaruh", ascending=False).set_index("Variabel")
        st.bar_chart(fi_df)

# ==========================================
# HALAMAN 3: TENTANG PROYEK & KELOMPOK
# ==========================================
elif menu == "Tentang":
    st.header("Tentang Aplikasi")
    st.write("Aplikasi ini merupakan implementasi algoritma Machine Learning (**Random Forest Regressor**) berbasis web menggunakan framework **Streamlit** untuk memprediksi permintaan penyewaan sepeda publik di Seoul (*Seoul Bike Sharing Demand*).")
    
    st.subheader("Anggota Kelompok 4")
    st.markdown("""
    * **Sheva Abdullah Huda** (241011400640)
    * **Michael Alfandigo** (241011400647)
    * **Firman Riyadi** (241011400649)
    
    **Program Studi Teknik Informatika — Fakultas Ilmu Komputer**  
    **Universitas Pamulang (2026)**
    """)