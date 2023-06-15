# import libary
import streamlit as st

# import norm
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import streamlit.components.v1 as components

# pige title
st.set_page_config(
    page_title="Forecasting Data Saham PT Ultrajaya Milk Industry & Trading Company",
    page_icon="https://abisgajian.id/images/thumbnail/ini-dia-daftar-saham-kategori-blue-chip-di-bursa-saham-indonesia.jpg",
)

# 0 = Anda Tidak Depresi
# 1 = Anda Depresi

# hide menu
hide_streamlit_style = """

<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# insialisasi web
st.markdown(
    "<h1 style='text-align: center; color: black; margin:0 ; padding:0;'>Forecasting Data Saham PT INDOFOOD (Time Series Data)</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">',
    unsafe_allow_html=True,
)
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown(
    ' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="https://github.com/Yayanmnh/kuispro.git" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> GitHub</button></a><a href="https://alifnurfathurrahmanprasodjo.github.io/DATAMINING/project_pendat.html?highlight=project" target="_blank"><button  style="border-radius: 12px;position: relative; top:50%;"><i style="color: orange" class="fa fa-book"></i> Jupyter Book</button></a></div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Data", "Preprocessing Data", "Model", "Implementasi"]
)
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write("Deskripsi Aplikasi :")
        st.markdown(
            "<p style='text-align: justify;'>Data PT Ultrajaya Milk Industry & Trading Company Tbk (ULTJ.JK) adalah data harga saham yang ditampilkan dalam format Jakarta Delayed Price dan dalam mata uang Rupiah (IDR). Informasi ini adalah data yang berasal dari platform atau sumber yang memberikan data pasar keuangan secara real-time atau dengan penundaan tertentu. Data diambil dari link https://finance.yahoo.com/quote/ULTJ.JK/history?p=ULTJ.JK</p>",
            unsafe_allow_html=True,
        )
        st.write("Tipe Data :")
        st.markdown(
            "<p style='text-align: justify;'>Tipe data yang diberikan adalah data harga saham PT Ultrajaya Milk Industry & Trading Company Tbk di Bursa Efek Indonesia (BEI). Biasanya, data harga saham terdiri dari beberapa kolom seperti tanggal perdagangan (Date), harga pembukaan (Open), harga penutupan (Close), harga tertinggi (High), harga terendah (Low), volume perdagangan (Volume), dan mungkin ada kolom tambahan terkait indikator atau informasi lainnya",
            unsafe_allow_html=True,
        )
        st.write("Deskripsi Data :")
        st.write(
            "<p style='text-align: justify;'>Data ini memberikan informasi tentang pergerakan harga saham PT Ultrajaya Milk Industry & Trading Company Tbk di pasar saham Jakarta. Data tersebut menggambarkan harga saham pada waktu tertentu, dan seringkali berisi historis harga saham yang dapat digunakan untuk analisis teknis, evaluasi kinerja saham, dan pengambilan keputusan investasi.",
            unsafe_allow_html=True,
        )
    with col2:
        data = pd.read_csv(
            "https://raw.githubusercontent.com/Yayanmnh/kuispro/main/ULTJ.JK.csv"
        )
        data

with tab2:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/Yayanmnh/kuispro/main/ULTJ.JK.csv",
        usecols=["Date", "Open"],
    )
    df

    dataSplit = """
    data = df["Open"]
    n = len(data)
    sizeTrain = round(n * 0.8)
    data_train = pd.DataFrame(data[:sizeTrain])
    data_test = pd.DataFrame(data[sizeTrain:])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(data_train)

    # Mengaplikasikan MinMaxScaler pada data pengujian
    test_scaled = scaler.transform(data_test)
    # joblib.dump(scaler, 'modelScaler.pkl')

    train = pd.DataFrame(train_scaled, columns=["data"])
    train = train["data"]

    test = pd.DataFrame(test_scaled, columns=["data"])
    test = test["data"]
    """
    st.code(dataSplit, language="python")

    data = df["Open"]
    n = len(data)
    sizeTrain = round(n * 0.8)
    data_train = pd.DataFrame(data[:sizeTrain])
    data_test = pd.DataFrame(data[sizeTrain:])

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(data_train)

    # Mengaplikasikan MinMaxScaler pada data pengujian
    test_scaled = scaler.transform(data_test)
    # joblib.dump(scaler, 'modelScaler.pkl')

    train = pd.DataFrame(train_scaled, columns=["data"])
    train = train["data"]

    test = pd.DataFrame(test_scaled, columns=["data"])
    test = test["data"]
    test

    saveData = """
    import numpy as np
from numpy import array

def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
    # find the end of this pattern
    end_ix = i + n_steps
    # check if we are beyond the sequence
    if end_ix > len(sequence)-1:
      break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)

  return array(X), array(y)

df_X, df_Y = split_sequence(train, 3)
x = pd.DataFrame(df_X)
y = pd.DataFrame(df_Y)
dataset_train = pd.concat([x, y], axis=1)
dataset_train.columns = [f"X{i+1}" for i in range(df_X.shape[1])] + ["Y"]

X_train = dataset_train.iloc[:, :2].values
Y_train = dataset_train.iloc[:, -1].values
test_x, test_y = split_sequence(test, 3)
x = pd.DataFrame(test_x)
y = pd.DataFrame(test_y)
dataset_test = pd.concat([x, y], axis=1)
dataset_test.columns = [f"X{i+1}" for i in range(test_x.shape[1])] + ["Y"]
# dataset_test.to_csv('data-test.csv', index=False)
X_test = dataset_test.iloc[:, :2].values
Y_test = dataset_test.iloc[:, -1].values

dataset_train
    """
    st.code(saveData, language="python")

    import numpy as np
    from numpy import array

    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence) - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)

        return array(X), array(y)

    df_X, df_Y = split_sequence(train, 3)
    x = pd.DataFrame(df_X)
    y = pd.DataFrame(df_Y)
    dataset_train = pd.concat([x, y], axis=1)
    dataset_train.columns = [f"X{i+1}" for i in range(df_X.shape[1])] + ["Y"]

    X_train = dataset_train.iloc[:, :2].values
    Y_train = dataset_train.iloc[:, -1].values
    test_x, test_y = split_sequence(test, 3)
    x = pd.DataFrame(test_x)
    y = pd.DataFrame(test_y)
    dataset_test = pd.concat([x, y], axis=1)
    dataset_test.columns = [f"X{i+1}" for i in range(test_x.shape[1])] + ["Y"]
    # dataset_test.to_csv('data-test.csv', index=False)
    X_test = dataset_test.iloc[:, :2].values
    Y_test = dataset_test.iloc[:, -1].values

    dataset_test

with tab3:
    pilihanModel = st.radio(
        "Pilih model yang ingin ditampilkan :",
        ("Decission Tree", "LinearRegression", "Multilayer perceptron"),
    )

    if pilihanModel == "Decission Tree":
        model = """
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_absolute_percentage_error
        model = DecisionTreeRegressor()
        model.fit(X_train, Y_train)

        y_pred=model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        """
        st.code(model, language="python")

        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_absolute_percentage_error

        model = DecisionTreeRegressor()
        model.fit(X_train, Y_train)

        y_pred = model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        st.write("MAPE = ", error)

    elif pilihanModel == "LinearRegression":
        model1 = """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_percentage_error
        model = LinearRegression()
        model.fit(X_train, Y_train)

        y_pred=model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        """
        st.code(model1, language="python")

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_percentage_error

        model = LinearRegression()
        model.fit(X_train, Y_train)

        y_pred = model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        st.write("MAPE = ", error)

    else:
        model1 = """
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_absolute_percentage_error
        model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
        model.fit(X_train, Y_train)

        y_pred=model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        """
        st.code(model1, language="python")

        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_absolute_percentage_error

        model = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            random_state=42,
        )
        model.fit(X_train, Y_train)

        y_pred = model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        st.write("MAPE = ", error)
with tab4:
    pilihModel = st.radio(
        "Pilih model yang ingin dipakai :",
        ("1. K-NN (K-Nearest Neighbor)", "2. K-Means"),
    )

    if pilihModel == "1. K-NN (K-Nearest Neighbor)":
        st.write("Anda memilih untuk memakai model K-NN (K-Nearest Neighbor)")
        kolom = st.columns((2, 0.5, 2.7))

        form = kolom[1].button("Form")
        about = kolom[2].button("About")

        # form page
        if form == False and about == False or form == True and about == False:
            st.markdown(
                "<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Forecasting Data Saham PT INDOFOOD (Time Series Data)</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                nama = st.text_input("Masukkan Nama", placeholder="Nama")
            with col2:
                Age = st.number_input("Masukkan Umur", max_value=100)
            sex = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
            col3, col4, col5 = st.columns(3)
            with col3:
                Number_children = st.number_input(
                    "Jumlah anak yang dimiliki", min_value=0, max_value=999999999999
                )
            with col4:
                education_level = st.number_input(
                    "Level edukasi", min_value=0, max_value=999999999999
                )
            with col5:
                total_members = st.number_input(
                    "Jumlah anggota keluarga", min_value=0, max_value=999999999999
                )
            incoming_salary = st.selectbox(
                "Apakah memiliki pendapatan?", ("Ya", "Tidak")
            )
            #    Centering Butoon
            columns = st.columns((2, 0.6, 2))
            sumbit = columns[1].button("Submit")
            if (
                sumbit
                and nama != ""
                and sex != ""
                and Number_children != 0
                and education_level != 0
                and Age != 0
                and total_members != 0
                and incoming_salary != ""
            ):
                # cek jenis kelamin
                # 0 = laki-laki
                # 1 = perempuan
                if sex == "Laki-laki":
                    sex = 0
                else:
                    sex = 1

                # cek memiliki pendapatan
                # 0 = Ya
                # 1 = Tidak
                if incoming_salary == "Ya":
                    incoming_salary = 0
                else:
                    incoming_salary = 1
                # normalisasi data
                data = norm.normalisasi(
                    [
                        sex,
                        Age,
                        Number_children,
                        education_level,
                        total_members,
                        incoming_salary,
                    ]
                )
                # prediksi data
                prediksi = norm.knn(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        st.write(prediksi)
                        time.sleep(1)
                        st.success("Hasil Prediksi : " + nama + ", anda tidak depresi")
                    else:
                        st.write(prediksi)
                        time.sleep(1)
                        st.warning("Hasil Prediksi : " + nama + ", anda depresi")

        # about page
        if about == True and form == False:
            st.markdown(
                "<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tentang Sistem ini</h1>",
                unsafe_allow_html=True,
            )
            st.write(" ")
            st.write(
                "Sistem Predeksi Depresi adalah sebuah sistem yang bertujuan untuk memprediksi apakah seseorang dalam keadaan depresi atau tidak. Sistem ini dibuat menggunakan bahasa pemrograman python dan library streamlit."
            )
            st.markdown(
                "<p  color: white;>Pada sistem ini menggunakan model KNN ( <i>K-nearest neighbors algorithm</i> ) dengan parameter <b>K = 6</b> . Dataset yang digunakan memiliki <b>8 fitur</b> termasuk kelas. Alasan menggunakan model KNN dengan parameter k = 6 adalah karena memiliki akurasi yang terbesar dari model lainnya pada dataset ini, sehingga diputuskan untuk menggunakan model tersebut.</p>",
                unsafe_allow_html=True,
            )
            st.info(
                "Pada data input level edukasi, Level 0 = tidak beredukasi||Level 1-6 = SD kelas 1-6||Level 7-9 = SMP kelas 7-9||Level 10-12 = SMA kelas 10-12||Level 13-14 = S1||Level 15-16 = S2||Level 17-19 = S3",
                icon="ℹ️",
            )
            st.caption("Alifnur Fathurrahman Prasodjo - 200411100150")
    else:
        st.write("Anda memilih untuk memakai model K-Means")
        kolom = st.columns((2, 0.5, 2.7))

        form = kolom[1].button("Form")
        about = kolom[2].button("About")

        # form page
        if form == False and about == False or form == True and about == False:
            st.markdown(
                "<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Prediksi Depresi</h1>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                nama = st.text_input("Masukkan Nama", placeholder="Nama")
            with col2:
                Age = st.number_input("Masukkan Umur", max_value=100)
            sex = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))
            col3, col4, col5 = st.columns(3)
            with col3:
                Number_children = st.number_input(
                    "Jumlah anak yang dimiliki", min_value=0, max_value=999999999999
                )
            with col4:
                education_level = st.number_input(
                    "Level edukasi", min_value=0, max_value=999999999999
                )
            with col5:
                total_members = st.number_input(
                    "Jumlah anggota keluarga", min_value=0, max_value=999999999999
                )
            incoming_salary = st.selectbox(
                "Apakah memiliki pendapatan?", ("Ya", "Tidak")
            )
            #    Centering Butoon
            columns = st.columns((2, 0.6, 2))
            sumbit = columns[1].button("Submit")
            if (
                sumbit
                and nama != ""
                and sex != ""
                and Number_children != 0
                and education_level != 0
                and Age != 0
                and total_members != 0
                and incoming_salary != ""
            ):
                # cek jenis kelamin
                # 0 = laki-laki
                # 1 = perempuan
                if sex == "Laki-laki":
                    sex = 0
                else:
                    sex = 1

                # cek memiliki pendapatan
                # 0 = Ya
                # 1 = Tidak
                if incoming_salary == "Ya":
                    incoming_salary = 0
                else:
                    incoming_salary = 1
                # normalisasi data
                data = norm.normalisasi(
                    [
                        sex,
                        Age,
                        Number_children,
                        education_level,
                        total_members,
                        incoming_salary,
                    ]
                )
                # prediksi data
                prediksi = norm.kmeans(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        st.write(prediksi)
                        time.sleep(1)
                        st.success("Hasil Prediksi : " + nama + ", anda tidak depresi")
                    else:
                        st.write(prediksi)
                        time.sleep(1)
                        st.warning("Hasil Prediksi : " + nama + ", anda depresi")

        # about page
        if about == True and form == False:
            st.markdown(
                "<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tentang Sistem ini</h1>",
                unsafe_allow_html=True,
            )
            st.write(" ")
            st.write(
                "Sistem Predeksi Depresi adalah sebuah sistem yang bertujuan untuk memprediksi apakah seseorang dalam keadaan depresi atau tidak. Sistem ini dibuat menggunakan bahasa pemrograman python dan library streamlit."
            )
            st.markdown(
                "<p  color: white;>Pada sistem ini menggunakan model KNN ( <i>K-nearest neighbors algorithm</i> ) dengan parameter <b>K = 6</b> . Dataset yang digunakan memiliki <b>8 fitur</b> termasuk kelas. Alasan menggunakan model KNN dengan parameter k = 6 adalah karena memiliki akurasi yang terbesar dari model lainnya pada dataset ini, sehingga diputuskan untuk menggunakan model tersebut.</p>",
                unsafe_allow_html=True,
            )
            st.info(
                "Pada data input level edukasi, Level 0 = tidak beredukasi||Level 1-6 = SD kelas 1-6||Level 7-9 = SMP kelas 7-9||Level 10-12 = SMA kelas 10-12||Level 13-14 = S1||Level 15-16 = S2||Level 17-19 = S3",
                icon="ℹ️",
            )
            st.caption("Alifnur Fathurrahman Prasodjo - 200411100150")
