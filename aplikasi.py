import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime
import calendar
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


# pige title
st.set_page_config(
    page_title="Forecasting Data Saham PT Ultrajaya Milk Industry & Trading Company",
    page_icon="https://abisgajian.id/images/thumbnail/ini-dia-daftar-saham-kategori-blue-chip-di-bursa-saham-indonesia.jpg",
)


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
    "<h1 style='text-align: center; color: black; margin:0 ; padding:0;'>Forecasting Data Saham PT Ultrajaya Milk Industry & Trading Company</h1>",
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

    # import numpy as np
    # from numpy import array

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

    df_X, df_Y = split_sequence(train, 2)
    x = pd.DataFrame(df_X)
    y = pd.DataFrame(df_Y)
    dataset_train = pd.concat([x, y], axis=1)
    dataset_train.columns = [f"X{i+1}" for i in range(df_X.shape[1])] + ["Y"]
    X_train = dataset_train.iloc[:, :2].values
    Y_train = dataset_train.iloc[:, -1].values
    test_x, test_y = split_sequence(test, 2)
    x = pd.DataFrame(test_x)
    y = pd.DataFrame(test_y)
    dataset_test = pd.concat([x, y], axis=1)
    dataset_test.columns = [f"X{i+1}" for i in range(test_x.shape[1])] + ["Y"]
    dataset_test.to_csv("data-test.csv", index=False)
    X_test = dataset_test.iloc[:, :2].values
    Y_test = dataset_test.iloc[:, -1].values

    dataset_train


with tab3:
    pilihanModel = st.radio(
        "Pilih model yang ingin ditampilkan :",
        ("Decission Tree", "LinearRegression", "Multilayer perceptron"),
    )

    if pilihanModel == "Decission Tree":
        model = """from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_absolute_percentage_error
        model = DecisionTreeRegressor()
        model.fit(X_train, Y_train)

        y_pred=model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        """
        st.code(model, language="python")

        # from sklearn.tree import DecisionTreeRegressor
        # from sklearn.metrics import mean_absolute_percentage_error

        model1 = DecisionTreeRegressor()
        model1.fit(X_train, Y_train)

        y_pred = model1.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        st.write("MAPE = ", error)

    elif pilihanModel == "LinearRegression":
        model1 = """from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_percentage_error
        model = LinearRegression()
        model.fit(X_train, Y_train)

        y_pred=model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        """
        st.code(model1, language="python")

        # from sklearn.linear_model import LinearRegression
        # from sklearn.metrics import mean_absolute_percentage_error

        model2 = LinearRegression()
        model2.fit(X_train, Y_train)

        y_pred = model2.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        st.write("MAPE = ", error)

    else:
        model1 = """from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_absolute_percentage_error
        model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
        model.fit(X_train, Y_train)

        y_pred=model.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        """
        st.code(model1, language="python")

        model3 = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            random_state=42,
        )
        model3.fit(X_train, Y_train)

        y_pred = model3.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        st.write("MAPE = ", error)
with tab4:
    # from sklearn.linear_model import LinearRegression
    # from sklearn.metrics import mean_absolute_percentage_error

    model2 = LinearRegression()
    model2.fit(X_train, Y_train)

    y_pred = model2.predict(X_test)
    if st.button("Process Code"):

        def ramal(dataset_test, tanggal):
            lr = model2
            tanggal = tanggal[0].split("-")
            tahun = tanggal[0]
            bulan = tanggal[1]
            hari = tanggal[2]
            tahun = int(tahun)
            bulan = int(bulan)
            hari = int(hari)
            jumlah_hari = calendar.monthrange(tahun, bulan)[1]

            last = dataset_test.tail(1)
            fitur = last.values
            n_fit = len(fitur[0])
            fiturs = np.zeros((n_fit))
            fitur = fitur[:, 1:n_fit]
            y_pred = lr.predict(fitur)
            new_fit = np.array(fitur[0])
            new_fit = np.append(new_fit, y_pred)
            fiturs[:] = new_fit
            hari += 1
            if hari > jumlah_hari:
                bulan += 1
                hari = 1
            if bulan > 12:
                tahun += 1
                bulan = 1

            tanggal = str(tahun) + "-" + f"{bulan:02d}" + "-" + f"{hari:02d}"

            # Mengonversi string ke objek datetime
            tanggal_cek = datetime.strptime(tanggal, "%Y-%m-%d")
            nama_hari = tanggal_cek.strftime("%A")
            # Mendapatkan nama hari dari objek datetime
            if nama_hari == "Saturday":
                hari += 2
                tanggal = str(tahun) + "-" + f"{bulan:02d}" + "-" + f"{hari:02d}"
            elif nama_hari == "Sunday":
                hari += 1
                tanggal = str(tahun) + "-" + f"{bulan:02d}" + "-" + f"{hari:02d}"
            return y_pred, tanggal

        tanggal_terakhir = df["Date"].tail(1).values
        pred, tanggal = ramal(dataset_test, tanggal_terakhir)
        reshaped_data = pred.reshape(-1, 1)
        original_data = scaler.inverse_transform(reshaped_data)
        pred = original_data.flatten()
        df_pred = pd.DataFrame({"Date": tanggal, "Open": pred})
        lum = df.tail(2)
        st.write(lum)
        st.write("Hasil Prediksi:")
        st.write(df_pred)
