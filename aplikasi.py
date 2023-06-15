# import libary 
import streamlit as st
import norm
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import streamlit.components.v1 as components

# pige title
st.set_page_config(
    page_title="Forecasting Data Saham PT INDOFOOD (Time Series Data)",
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
st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Forecasting Data Saham PT INDOFOOD (Time Series Data)</h1>", unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">', unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown(' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="https://github.com/AlifnurFathurrahmanPrasodjo/Project-Penambangan-Data.git" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> GitHub</button></a><a href="https://alifnurfathurrahmanprasodjo.github.io/DATAMINING/project_pendat.html?highlight=project" target="_blank"><button  style="border-radius: 12px;position: relative; top:50%;"><i style="color: orange" class="fa fa-book"></i> Jupyter Book</button></a></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Preprocessing Data", "Model", "Implementasi"])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write("Deskripsi Aplikasi :")
        st.markdown("<p style='text-align: justify;'>Aplikasi Peramalan Data Saham PT INDOFOOD adalah perangkat lunak yang dirancang untuk memberikan kemampuan peramalan dan analisis untuk data saham PT INDOFOOD, sebuah perusahaan fiktif. Aplikasi ini bertujuan untuk membantu investor, pedagang, dan analis keuangan dalam membuat keputusan dengan memprediksi harga dan tren saham di masa depan.</p>", unsafe_allow_html=True)
        st.write("Sumber data :")
        st.markdown("<p style='text-align: justify;'>Aplikasi mengambil data stok historis PT INDOFOOD dari sumber terpercaya https://finance.yahoo.com/quote/INDF.JK/profile?p=INDF.JK. Ini mengambil data seperti harga saham harian, volume perdagangan, dan metrik keuangan relevan lainnya.</p>", unsafe_allow_html=True)
        st.write("Deskripsi Data :")
        st.write("1. Date: Tanggal entri data pasar saham.\n 2. Open: Harga pembukaan saham pada hari itu.\n 3. High: Harga tertinggi yang dicapai saham selama hari perdagangan.\n 4. Low: Harga terendah yang dicapai oleh saham selama hari perdagangan.\n 5. Close: Harga penutupan saham pada hari itu.\n 6. Adj Close: Harga penutupan saham yang disesuaikan, yang memperhitungkan tindakan korporasi apa pun (seperti dividen atau pemecahan saham) yang dapat memengaruhi harga saham.\n 7. Volume: Volume perdagangan, yaitu jumlah total saham yang diperdagangkan pada hari itu.")
    with col2:
        data = pd.read_csv('https://raw.githubusercontent.com/AlifnurFathurrahmanPrasodjo/dataFolder/main/dataMining/INDF.JK.csv')
        data

with tab2:
    data = pd.read_csv('https://raw.githubusercontent.com/AlifnurFathurrahmanPrasodjo/Project-Penambangan-Data/main/b_depressed.csv')
    data.fillna(0,inplace=True)

    fd1 = data.drop(data.columns[0:2],axis=1)
    fd2 = fd1.drop(fd1.columns[2],axis=1)
    fd3 = fd2.drop(fd2.columns[5:10],axis=1)
    fd4 = fd3.drop(fd3.columns[6:14],axis=1)
    fd4

    y = fd4['depressed'].values
    x = fd4.drop(fd4.columns[6],axis=1)

    dataSplit='''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)
    X_train.shape + X_test.shape
    '''
    st.code(dataSplit, language='python')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=1)
    X_train.shape + X_test.shape

    saveData='''
    from pathlib import Path  
    filepath = Path('/content/drive/MyDrive/DATA MINING/TUGAS/model/data_depresi.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    x.to_csv(filepath) 
    '''
    st.code(saveData, language='python')    

    prepro='''
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y

    y_class = data['depressed']
    y = y_class.values.tolist()
    '''    
    st.code(prepro, language='python')

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    y
    y_class = data['depressed']
    y = y_class.values.tolist()
    
    MinMax='''
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(x)
    nama_fitur = x.columns.copy()
    scaled_fitur = pd.DataFrame(scaled,columns=nama_fitur)
    scaled_fitur
    ''' 
    st.code(MinMax, language='python')

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(x)
    nama_fitur = x.columns.copy()
    scaled_fitur = pd.DataFrame(scaled,columns=nama_fitur)
    scaled_fitur

    dataSplit2='''
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(scaled_fitur, y, test_size=0.2, random_state=1)
    X_train.shape + X_test.shape
    '''
    st.code(dataSplit2, language='python')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test=train_test_split(scaled_fitur, y, test_size=0.2, random_state=1)
    X_train.shape + X_test.shape

    savenormalisasi='''
    import joblib
    filename = '/content/drive/MyDrive/DATA MINING/TUGAS/model/norm.sav'
    joblib.dump(scaler, filename) 
    '''
    st.code(savenormalisasi, language='python')

with tab3:
    pilihanModel = st.radio(
    "Pilih model yang ingin ditampilkan :",
    ('K-NN (K-Nearest Neighbor)', 'K-Means'))

    if pilihanModel == 'K-NN (K-Nearest Neighbor)':
        model='''
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics
        #Try running from k=1 through 30 and record testing accuracy
        k_range = range(1,31)
        scores = {}
        scores_list = []
        for k in k_range:
                # install model
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train,y_train)
                # save model
                filenameKNN = '/content/drive/MyDrive/DATA MINING/TUGAS/model/modelKNN'+str(k)+'.pkl'
                joblib.dump(knn,filenameKNN)
                y_pred=knn.predict(X_test)
                scores[k] = accuracy_score(y_test,y_pred)
                scores_list.append(accuracy_score(y_test,y_pred))
        scores
        '''
        st.code(model, language='python')

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics
        #Try running from k=1 through 30 and record testing accuracy
        k_range = range(1,31)
        scores = {}
        scores_list = []
        for k in k_range:
                # install model
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train,y_train)
                # save model
                y_pred=knn.predict(X_test)
                scores[k] = accuracy_score(y_test,y_pred)
                scores_list.append(accuracy_score(y_test,y_pred))
        scores

        skor='''
        scores_list.index(max(scores_list))+1 , max(scores_list)
        '''
        st.code(skor, language='python')

        scores_list.index(max(scores_list))+1 , max(scores_list)

        akurasi='''
        knn = KNeighborsClassifier(n_neighbors=scores_list.index(max(scores_list))+1)
        knn.fit(X_train,y_train)
        y_pred_knn =knn.predict(X_test)
        # y_pred_knn
        cm = confusion_matrix(y_test,y_pred_knn)
        precision = round(precision_score(y_test,y_pred_knn, average="macro")*100,2)
        acc = round(accuracy_score(y_test,y_pred_knn)*100,2)
        recall = round(recall_score(y_test,y_pred_knn, average="macro")*100,2)
        f1score = round(f1_score(y_test, y_pred_knn, average="macro")*100,2)
        '''
        st.code(akurasi, language='python')

        knn = KNeighborsClassifier(n_neighbors=scores_list.index(max(scores_list))+1)
        knn.fit(X_train,y_train)
        y_pred_knn =knn.predict(X_test)
        # y_pred_knn
        cm = confusion_matrix(y_test,y_pred_knn)
        precision = round(precision_score(y_test,y_pred_knn, average="macro")*100,2)
        acc = round(accuracy_score(y_test,y_pred_knn)*100,2)
        recall = round(recall_score(y_test,y_pred_knn, average="macro")*100,2)
        f1score = round(f1_score(y_test, y_pred_knn, average="macro")*100,2)

        st.write('Konfusi Matrix')
        cm
        st.write('precision:')
        precision
        st.write('recall:')
        recall
        st.write('fscore:' )
        f1score
        st.write('accuracy:')
        acc
    else:
        model1='''
        from sklearn.cluster import KMeans

        # #Try running from n=1 through 30 and record testing accuracy
        n_range = range(1,31)
        akurasi = {}
        akurasi_score = []
        for k in n_range:
                # install model
                kmeans = KMeans(n_clusters=k,random_state=0)
                kmeans.fit(X_train,y_train)
                # save model
                filenameKMeans = '/content/drive/MyDrive/DATA MINING/TUGAS/model/modelKmeans'+str(k)+'.pkl'
                joblib.dump(kmeans,filenameKMeans)
                y_pred=kmeans.predict(X_test)
                akurasi[k] = accuracy_score(y_test,y_pred)
                akurasi_score.append(accuracy_score(y_test,y_pred))
        akurasi_score
        '''
        st.code(model1, language='python')

        from sklearn.cluster import KMeans

        # #Try running from n=1 through 30 and record testing accuracy
        n_range = range(1,31)
        akurasi = {}
        akurasi_score = []
        for k in n_range:
                # install model
                kmeans = KMeans(n_clusters=k,random_state=0)
                kmeans.fit(X_train,y_train)
                # save model
                y_pred=kmeans.predict(X_test)
                akurasi[k] = accuracy_score(y_test,y_pred)
                akurasi_score.append(accuracy_score(y_test,y_pred))
        akurasi_score

        skor1='''
        akurasi_score.index(max(akurasi_score))+1 , max(akurasi_score)
        '''
        st.code(skor1, language='python')

        akurasi_score.index(max(akurasi_score))+1 , max(akurasi_score)

        akurasi1='''
        KMeans = KNeighborsClassifier(n_neighbors=akurasi_score.index(max(akurasi_score))+1)
        KMeans.fit(X_train,y_train)
        y_pred_KMeans =KMeans.predict(X_test)
        # y_pred_KMeans
        cm = confusion_matrix(y_test,y_pred_KMeans)
        precision = round(precision_score(y_test,y_pred_KMeans, average="macro")*100,2)
        acc = round(accuracy_score(y_test,y_pred_KMeans)*100,2)
        recall = round(recall_score(y_test,y_pred_KMeans, average="macro")*100,2)
        f1score = round(f1_score(y_test, y_pred_KMeans, average="macro")*100,2)
        '''
        st.code(akurasi1, language='python')

        KMeans = KNeighborsClassifier(n_neighbors=akurasi_score.index(max(akurasi_score))+1)
        KMeans.fit(X_train,y_train)
        y_pred_KMeans =KMeans.predict(X_test)
        # y_pred_KMeans
        cm = confusion_matrix(y_test,y_pred_KMeans)
        precision = round(precision_score(y_test,y_pred_KMeans, average="macro")*100,2)
        acc = round(accuracy_score(y_test,y_pred_KMeans)*100,2)
        recall = round(recall_score(y_test,y_pred_KMeans, average="macro")*100,2)
        f1score = round(f1_score(y_test, y_pred_KMeans, average="macro")*100,2)

        st.write('Konfusi Matrix')
        cm
        st.write('precision:')
        precision
        st.write('recall:')
        recall
        st.write('fscore:' )
        f1score
        st.write('accuracy:')
        acc
with tab4:
    pilihModel = st.radio(
    "Pilih model yang ingin dipakai :",
    ('1. K-NN (K-Nearest Neighbor)', '2. K-Means'))

    if pilihModel == '1. K-NN (K-Nearest Neighbor)':
        st.write('Anda memilih untuk memakai model K-NN (K-Nearest Neighbor)')
        kolom = st.columns((2 , 0.5, 2.7))   

        form = kolom[1].button('Form')
        about = kolom[2].button('About')

        # form page
        if form==False and about==False or form==True and about==False:
            st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Forecasting Data Saham PT INDOFOOD (Time Series Data)</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                nama = st.text_input("Masukkan Nama",placeholder='Nama')
            with col2:
                Age = st.number_input("Masukkan Umur",max_value=100)
            sex = st.selectbox("Jenis Kelamin",('Laki-laki','Perempuan'))
            col3,col4,col5 = st.columns(3)
            with col3:
                Number_children = st.number_input("Jumlah anak yang dimiliki",min_value=0,max_value=999999999999)
            with col4:
                education_level = st.number_input("Level edukasi",min_value=0,max_value=999999999999)
            with col5:
                total_members = st.number_input("Jumlah anggota keluarga",min_value=0,max_value=999999999999)
            incoming_salary = st.selectbox("Apakah memiliki pendapatan?",('Ya','Tidak'))
            #    Centering Butoon 
            columns = st.columns((2, 0.6, 2))
            sumbit = columns[1].button("Submit")
            if sumbit and nama != '' and sex != '' and Number_children != 0 and education_level != 0 and Age != 0 and total_members != 0 and incoming_salary != '':
                # cek jenis kelamin
                #0 = laki-laki
                #1 = perempuan
                if sex == 'Laki-laki':
                    sex = 0
                else:
                    sex = 1
                
                # cek memiliki pendapatan
                #0 = Ya
                #1 = Tidak
                if incoming_salary == 'Ya':
                    incoming_salary = 0
                else:
                    incoming_salary = 1
                # normalisasi data
                data = norm.normalisasi([sex,Age,Number_children,education_level,total_members,incoming_salary])
                # prediksi data
                prediksi = norm.knn(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        st.write(prediksi)
                        time.sleep(1)
                        st.success("Hasil Prediksi : "+nama+", anda tidak depresi")
                    else:
                        st.write(prediksi)
                        time.sleep(1)
                        st.warning("Hasil Prediksi : "+nama+", anda depresi")

        # about page
        if about==True and form==False:
            st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tentang Sistem ini</h1>", unsafe_allow_html=True)
            st.write(' ')
            st.write('Sistem Predeksi Depresi adalah sebuah sistem yang bertujuan untuk memprediksi apakah seseorang dalam keadaan depresi atau tidak. Sistem ini dibuat menggunakan bahasa pemrograman python dan library streamlit.')
            st.markdown("<p  color: white;>Pada sistem ini menggunakan model KNN ( <i>K-nearest neighbors algorithm</i> ) dengan parameter <b>K = 6</b> . Dataset yang digunakan memiliki <b>8 fitur</b> termasuk kelas. Alasan menggunakan model KNN dengan parameter k = 6 adalah karena memiliki akurasi yang terbesar dari model lainnya pada dataset ini, sehingga diputuskan untuk menggunakan model tersebut.</p>", unsafe_allow_html=True)
            st.info('Pada data input level edukasi, Level 0 = tidak beredukasi||Level 1-6 = SD kelas 1-6||Level 7-9 = SMP kelas 7-9||Level 10-12 = SMA kelas 10-12||Level 13-14 = S1||Level 15-16 = S2||Level 17-19 = S3', icon="ℹ️")
            st.caption('Alifnur Fathurrahman Prasodjo - 200411100150')
    else :
        st.write('Anda memilih untuk memakai model K-Means')
        kolom = st.columns((2 , 0.5, 2.7))   

        form = kolom[1].button('Form')
        about = kolom[2].button('About')

        # form page
        if form==False and about==False or form==True and about==False:
            st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Prediksi Depresi</h1>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: white;'>Harap Diisi Semua Kolom</p>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                nama = st.text_input("Masukkan Nama",placeholder='Nama')
            with col2:
                Age = st.number_input("Masukkan Umur",max_value=100)
            sex = st.selectbox("Jenis Kelamin",('Laki-laki','Perempuan'))
            col3,col4,col5 = st.columns(3)
            with col3:
                Number_children = st.number_input("Jumlah anak yang dimiliki",min_value=0,max_value=999999999999)
            with col4:
                education_level = st.number_input("Level edukasi",min_value=0,max_value=999999999999)
            with col5:
                total_members = st.number_input("Jumlah anggota keluarga",min_value=0,max_value=999999999999)
            incoming_salary = st.selectbox("Apakah memiliki pendapatan?",('Ya','Tidak'))
            #    Centering Butoon 
            columns = st.columns((2, 0.6, 2))
            sumbit = columns[1].button("Submit")
            if sumbit and nama != '' and sex != '' and Number_children != 0 and education_level != 0 and Age != 0 and total_members != 0 and incoming_salary != '':
                # cek jenis kelamin
                #0 = laki-laki
                #1 = perempuan
                if sex == 'Laki-laki':
                    sex = 0
                else:
                    sex = 1
                
                # cek memiliki pendapatan
                #0 = Ya
                #1 = Tidak
                if incoming_salary == 'Ya':
                    incoming_salary = 0
                else:
                    incoming_salary = 1
                # normalisasi data
                data = norm.normalisasi([sex,Age,Number_children,education_level,total_members,incoming_salary])
                # prediksi data
                prediksi = norm.kmeans(data)
                # cek prediksi
                with st.spinner("Tunggu Sebentar Masih Proses..."):
                    if prediksi[-1] == 0:
                        st.write(prediksi)
                        time.sleep(1)
                        st.success("Hasil Prediksi : "+nama+", anda tidak depresi")
                    else:
                        st.write(prediksi)
                        time.sleep(1)
                        st.warning("Hasil Prediksi : "+nama+", anda depresi")

        # about page
        if about==True and form==False:
            st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Tentang Sistem ini</h1>", unsafe_allow_html=True)
            st.write(' ')
            st.write('Sistem Predeksi Depresi adalah sebuah sistem yang bertujuan untuk memprediksi apakah seseorang dalam keadaan depresi atau tidak. Sistem ini dibuat menggunakan bahasa pemrograman python dan library streamlit.')
            st.markdown("<p  color: white;>Pada sistem ini menggunakan model KNN ( <i>K-nearest neighbors algorithm</i> ) dengan parameter <b>K = 6</b> . Dataset yang digunakan memiliki <b>8 fitur</b> termasuk kelas. Alasan menggunakan model KNN dengan parameter k = 6 adalah karena memiliki akurasi yang terbesar dari model lainnya pada dataset ini, sehingga diputuskan untuk menggunakan model tersebut.</p>", unsafe_allow_html=True)
            st.info('Pada data input level edukasi, Level 0 = tidak beredukasi||Level 1-6 = SD kelas 1-6||Level 7-9 = SMP kelas 7-9||Level 10-12 = SMA kelas 10-12||Level 13-14 = S1||Level 15-16 = S2||Level 17-19 = S3', icon="ℹ️")
            st.caption('Alifnur Fathurrahman Prasodjo - 200411100150')