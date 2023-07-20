from module.cnn import *
#paggil module cnn
#library for flask web
import io, os
import logging
import time
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, abort, session,flash
from flask import render_template_string, stream_with_context
import pymysql.cursors


###=>>FLASK ROOT SETTING
project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, 'templates')
static_path = os.path.join(project_root, 'static')
#lakukan setting project

app = Flask(__name__, template_folder=template_path, static_folder=static_path)
#lakukan inisiasi web flask
app.secret_key = 'ini kunci rahasia'
###=>>END FLASK ROOT SETTING


cursor=conn=None

def openDb():
    # use global variable, can be used by all read and write functions and modules here
    global conn, cursor

    while True:
        # connect to database
        try:
            conn = pymysql.connect(host="localhost", user="root", password ='' ,db="voice_classification", cursorclass=pymysql.cursors.DictCursor, autocommit=True)
            #print(conn)
            # once connected can break out of this function            
            cursor = conn.cursor();
            break
        except Exception as e:
            print("Sorry - there is a problem connecting to the database...", e)

@app.route('/read')
def hello_world():
  if not conn:
      openDb()
  
  with conn:
    try:
      #cursor = conn.cursor();
      sql = "SELECT * FROM `admin`"
      cursor.execute(sql)
      result = cursor.fetchone()
      print(result)
    except pymysql.IntegrityError as e:
        print(e)
    except pymysql.InternalError as e:
        print(e)      

    except Exception as e:
        print("Error - please try again", e)
  # Read a single record
  return 'Hello world!'

def closeDb():
  global conn, cursor
  cursor.close()
  conn.close()
  #memutuskan koneksi ke database server
###=>>ENDKONEKSI DATABASE

#No caching at all for API endpoints.
@app.after_request
def add_header(response):
  # response.cache_control.no_store = True
  response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
  response.headers['Pragma'] = 'no-cache'
  response.headers['Expires'] = '-1'
  return response
  #untuk menghapus cache pada server flask secara automatis

@app.errorhandler(404)
def page_not_found(e):
  return render_template('404.html'), 404
  #untuk menampilakn halaman 404 ketika url tidak ditemukan

@app.route('/login', methods=['GET', 'POST'])
def login():
  if not conn:
      openDb()
  msg = ''
  # Output message if something goes wrong...
  if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
  # Check if "username" and "password" POST requests exist (user submitted form)
    username = request.form['username']
    password = request.form['password']
    # Create variables for easy access
    #sql = "SELECT * FROM `users` WHERE `username` = %s AND `password` = %s"
    #cursor.execute(sql, (username, password,))
    cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (username, password,))
    # Check if account exists using MySQL
    account = cursor.fetchone()
    # Fetch one record and return result
    if account:
    # If account exists in accounts table in out database
      session['loggedin'] = True
      session['id'] = account['id_admin']
      session['username'] = account['username']
      # Create session data, we can access this data in other routes
      #return redirect(url_for('indexDs'))
      return redirect(url_for('indexHome'))
      # Redirect to home page
    else:
      msg = 'Username atau password salah!'
      # Account doesnt exist or username/password incorrect
  return render_template('login.html', msg=msg)
  # Show the login form with message (if any)

@app.route('/logout')
def logout():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  session.pop('loggedin', None)
  session.pop('id', None)
  session.pop('username', None)
  # Remove session data, this will log the user out
  return redirect(url_for('login'))
  # Redirect to login page

#fungsi view index() untuk menampilkan data dari database
@app.route('/log')
def log():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  rowLog = []
  sql = "SELECT * FROM log_identifikasi"
  cursor.execute(sql)
  results = cursor.fetchall()
  #menjalkan query untuk mengambil sema ditabel log_identifikasi
  for data in results:
    rowLog.append(data)
    #menapung data pada list rowLog
  return render_template('log/index.html', rowLog=rowLog)
  #tampilkan halaman index.html untuk page log

#####################
# KELAS             #
#####################

#fungsi view index() untuk menampilkan kelas
@app.route('/kelas')
def indexKelas():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  rowKelas = []
  sql = "SELECT * FROM kelas"
  cursor.execute(sql)
  results = cursor.fetchall()
  print(results)
  #jalanakan query untuk mengambil semua data di tabel kelas
  for data in results:
    rowKelas.append(data)
    #tampung data pada list rowKelas
  return render_template('kelas/index.html', rowKelas=rowKelas)
  #tampilkan halaman index di page kelas

#fungsi untuk menghapus data kelas
@app.route('/kelas/hapus/<id>', methods=['GET','POST'])
def hapusKelas(id):
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  cursor.execute('SELECT * FROM kelas WHERE id_kelas=%s', (id,))
  row = cursor.fetchone()
  kelas = row['nama_kelas']
  #menjalankan query untuk mengambil nama kelas berdasarkan id
  shutil.rmtree(f"static/voice/{row['nama_kelas']}")   
  #remove folder kelas
  cursor.execute('DELETE FROM dataset WHERE label=%s', (row['id_kelas'],))
  conn.commit()
  #menghapus data di tabel dataset berdasarkan nama kelasnya
  cursor.execute('DELETE FROM kelas WHERE id_kelas=%s', (id,))
  conn.commit()
  #hapus nama kelas berdasarkan id kelas yang dipilih
  flash('Kelas {} berhasil dihapus!'.format(kelas), 'success')
  return redirect(url_for('indexKelas'))
  #kembalikan ke halaman index pada page kelas

#fungsi view edit() untuk form edit kelas
@app.route('/kelas/edit/<id>', methods=['GET','POST'])
def editKelas(id):
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  cursor.execute('SELECT * FROM kelas WHERE id_kelas=%s', (id))
  data = cursor.fetchone()
  print(data)
  #jalankan query untuk mengambil kelas berdasarkan id yang dipilih
  if request.method == 'POST':
    #jika terdapat kiriman data dari form
    old_id = request.form['id']
    kelasDipilih = request.form['kelasDipilih']
    #ambil data dari form tersebut

    kelas = request.form['ubahKelas'].capitalize()
    cursor.execute('SELECT * FROM kelas WHERE nama_kelas=%s', (kelas))
    #jalankan query untuk mengambil kelas berdasarkan nama kelas yang dipilih

    if cursor.fetchone() != None:
      #cek jika data sudah ada
      return render_template('kelas/edit.html', msg='Nama Kelas Sudah Ada!', data=data)
      #maka kembalikan kehalaman edit kelas dengan pesan nama kelas sudah ada
    else:
      #atau
      sql = "UPDATE kelas SET nama_kelas=%s WHERE id_kelas=%s"
      val = (kelas, old_id)
      cursor.execute(sql, val)
      conn.commit()
      #jalankan query untuk mengubah data ditambel kelas berdasarkan id
      os.rename(f"static/voice/{kelasDipilih}", f"static/voice/{kelas}")
      #melakukan chage pada nama file
    flash('Kelas {} berhasil diperbarui!'.format(kelas), 'success')
    return redirect(url_for('indexKelas'))
    #kembalikan ke halaman index di page kelas
  else:
    return render_template('kelas/edit.html', data=data, msg='')
    #tampilkan halaman form edit

#fungsi view tambah() untuk membuat form tambah
@app.route('/kelas/tambah', methods=['GET','POST']) 
def tambahKelas():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  if request.method == 'POST':
    #jika terdapat kiriman data dari form
    kelas = request.form['inputKelas'].capitalize()
    #ambil data tersebut
    cursor.execute('SELECT * FROM kelas WHERE nama_kelas=%s', (kelas))
    #jalankan query untuk mengambil data kelas berdasarkan nama kelas yang dipilih

    if cursor.fetchone() != None:
      #jika kelas ada
      return render_template('kelas/tambah.html', msg='Nama Kelas Sudah Ada!')
      #maka tampilkan halaman form tambah kelas dengan pesan nama kelas sudah ada
    else :
      #selain itu
      sql = "INSERT INTO kelas (nama_kelas) VALUES (%s)"
      val = (kelas)
      cursor.execute(sql, val)
      conn.commit()
      #jalankan query untuk mengambil menambahkan data baru pada tabel kelas
      if not os.path.exists(f"static/voice/{kelas}"):
        #jika folder kelas tersebut belum ada
        os.mkdir(f"static/voice/{kelas}")
        #maka buat folder dengan nama kelas tersebut
    flash('Kelas {} berhasil ditambahkan!'.format(kelas), 'success')
    return redirect(url_for('indexKelas'))
    #kembalikan ke halaman index page kelas
  else:
    return render_template('kelas/tambah.html', msg='')
    #tampilkan halaman form tambah kelas

####################
#END KELAS         #
####################

#fungsi view index() untuk menampilkan data dari database
@app.route('/ds')
def indexDs():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  rowDataset = []
  sql = "SELECT ds.id_dataset, nama_kelas, nama_file from kelas kls, dataset ds WHERE kls.id_kelas=ds.label"
  cursor.execute(sql)
  results = cursor.fetchall()
  #jalankan query untuk megambil data dari dua tabel secara bersamaan berdasarkan nama kelas yang sama
  for data in results:
    rowDataset.append(data)
    #tampung data pada list rowDataset

  return render_template('ds/index.html', rowDataset=rowDataset)
  #tampilkan halaman index untuk page dataset

@app.route('/home')
def indexHome():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  return render_template('home.html')
  #tampilkan halaman home login

###get semua record kelas dataset
def getKelas():
  rowKelas = []
  cursor.execute("select * from kelas")
  results = cursor.fetchall()
  #jalakan query untuk mengambil semua data kelas
  for row in results:
    rowKelas.append(row)
    #tampung data pada list rowKelas
  return rowKelas
  #kembalikan data tersebut

#fungsi view tambah() untuk membuat form tambah dataset
@app.route('/ds/tambah', methods=['GET','POST'])
def tambahDs():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404
  
  if request.method == 'POST':
    #jika terdapat kiriman data dari form
    kelas = request.form['listKelas'].split('::')
    voices = request.files.getlist("fileDataset[]")
    #tampung data yang dikrimkan ke varibel masing2
    jumlahFile = len(voices)

    for fileDataset in voices:
      #lakukan perulangan karena upload voice bisa multiple
      nm_voice = fileDataset.filename
      #ambil nama file dari voice yang diupload
      
      sql = "INSERT INTO dataset (label, nama_file) VALUES (%s, %s)"
      val = (kelas[0], nm_voice)
      cursor.execute(sql, val)
      conn.commit()
      #jalankan query untuk menambhkan dataset baru

      if fileDataset.filename != '':
      #jika file memiliki nama
        if os.path.isdir(f'static/voice/{kelas[1]}'):
          #jika folder untuk  kelas tersebut ada
          path = 'static/voice/%s/%s' % (kelas[1], nm_voice)
          fileDataset.save(path)
          #maka simpan voice pada folder tersebut    

    pesan = f"{jumlahFile} Dataset berhasil ditambahkan!"
    flash(pesan, 'success')
    return redirect(url_for('indexDs'))
    #kembalikan ke halaman index dataset
  else:
    return render_template('ds/tambah.html', rowKelas=getKelas())
    #tampilkan halaman tambah dataset

#fungsi view edit() untuk form edit dataset
@app.route('/ds/edit/<id>', methods=['GET','POST'])
def editDs(id):
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  cursor.execute('SELECT ds.id_dataset, kls.id_kelas, nama_kelas, nama_file from kelas kls, dataset ds WHERE kls.id_kelas=ds.label AND ds.id_dataset=%s', (id))
  data = cursor.fetchone()
  #jalankan query untuk mengambil data dari tabel kelas dan dataset
  if request.method == 'POST':
    #cek jika terdapat kiriman data dari form
    old_id = request.form['id']
    kelasDipilih = request.form['kelasDipilih']
    oldDataset = request.form['oldDataset']
    kelas = request.form['listKelas'].split('::')
    newDataset = request.files['newDataset']
    nm_voice = newDataset.filename or request.form['oldDataset']
    #ambil dan tampung data tersebut
    sql = "UPDATE dataset SET label=%s, nama_file=%s WHERE id_dataset=%s"
    val = (kelas[0], nm_voice, old_id)
    cursor.execute(sql, val)
    conn.commit()
    #jalankan query untuk mengubah dataset 
    
    if newDataset.filename == '' and kelasDipilih != kelas[1]:
      #cek jika nama kelas berubah
      shutil.move(f"static/voice/{kelasDipilih}/{oldDataset}", f"static/voice/{kelas[1]}/{oldDataset}")
      #pindah file ke folder kelas baru
  
    elif newDataset.filename != '':
      #cek jika file memiliki nama
      path = 'static/voice/%s/%s' % (kelasDipilih, oldDataset)
      if os.path.exists(path):
      #cek jika file ada
        os.remove(path)
        #hapus file lama
      newDataset.save('static/voice/%s/%s' % (kelas[1], nm_voice))
      #simpan file baru

    flash('Dataset {} berhasil diperbarui!'.format(nm_voice), 'success')
    return redirect(url_for('indexDs'))
    #kembalikan ke halaman index dataset
  else:
    return render_template('ds/edit.html', data=data, rowKelas=getKelas())
    #tampilkan halaman edit dataset

#fungsi untuk menghapus dataset
@app.route('/ds/hapus/<id>', methods=['GET','POST'])
def hapusDs(id):
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  cursor.execute('SELECT nama_kelas, nama_file from kelas kls, dataset ds WHERE kls.id_kelas=ds.label AND ds.id_dataset=%s', (id,))
  row = cursor.fetchone()
  dataset = row['nama_file']
  os.remove(f"static/voice/{row['nama_kelas']}/{row['nama_file']}")
  cursor.execute('DELETE FROM dataset WHERE id_dataset=%s', (id,))
  conn.commit()

  flash('Dataset {} berhasil dihapus!'.format(dataset), 'success')
  return redirect(url_for('indexDs'))

###>>PAGE PLAYING AUDIO 
@app.route('/play/<table>/<id>', methods=['GET','POST'])
def playAudio(table, id):
  if table == 'ds':
    #cek jika tabel yg dipilih dataset
    cursor.execute('SELECT ds.id_dataset, nama_kelas, nama_file from kelas kls, dataset ds WHERE kls.id_kelas=ds.label AND ds.id_dataset=%s', (id,))
    row = cursor.fetchone()
    #jalankan query untuk mengambil data pada dari tabel kelas dan dataset
    kelas = row['nama_kelas']
    audioName = row['nama_file']
    #ambil nama kelas dan nama audio
    fileAudio = 'voice/%s/%s' % (row['nama_kelas'], row['nama_file'])
    #cari nama file    
  elif table == 'li':
    #jika tabel log identifikasi
    cursor.execute('SELECT * FROM log_identifikasi WHERE id_log=%s', (id,))
    row = cursor.fetchone()
    #jalankan query untuk mengambil data pada tabel log identifikasi berdasarkan id
    kelas = row['hasil_id']
    audioName = row['nama_file_log']
    #ambil nama file
    fileAudio = 'voice_upload/%s' % (row['nama_file_log'])
    #cari di folder voice upload
  data = [kelas, audioName, fileAudio]
  #satukan file ke dalam list
  return render_template('playaudio.html', data=data)
  #tampilkan halaman play audio

@app.route("/trainmodel")
def trainmodel():
  global training_process
  if not conn:
      openDb()
  try:
    cursor.execute('SELECT * FROM hyperparam where id_hyperparam=1')
    row = cursor.fetchone()
    #jalakan query untuk mengampil nilai parameter
    num_epochs = row['epoch']
    num_batch_size = row['batch_size']
    #ambil nilai epoch dan batch size

    if training_process == True:
      #jika pelatihan sedang berjalan
      return f"[INFO] Please Wait.. <br> [INFO] Training parameter {num_epochs} epoch and {num_batch_size} batch size <br> [INFO] Model Training in Progress..."
      #tampilkan informasi training

    def generate():
      global training_process
      training_process = True

      print("training start")
      yield "<p> <a href='/training'><< back to training page</a> </p>\n"
      #buat link kembali page training
      model = my_model()
      #inisiasi model
      f = open("temp/log_train.txt", "w")
      #buka file log train

      stream = io.StringIO()
      model.summary(print_fn=lambda x:stream.write(x + '<br>'))
      summary_string = stream.getvalue()
      stream.close()
      #ambil summary model
      yield summary_string + '\n'
      #tampilkan ke browser
      f.write(f'<ul><li>{summary_string}</li>')
      #tulikan juga ke file log

      yield "[INFO] Preparing... <br>\n"
      f.write('<li>[INFO] Preparing...</li>')
      create_metadata()
      x_train, y_train, x_test, y_test = data_train_test()
      #jalankan feature extraction dan lakukan splitting data

      yield "[INFO] Training process running... <br>\n"
      f.write('<li>[INFO] Training process running...</li>')
      time.sleep(0.8)
      yield f"[INFO] Parameter set with {num_epochs} epoch and {num_batch_size} batch size <br>\n"
      f.write(f'<li>[INFO] Parameter set with {num_epochs} epoch and {num_batch_size} batch size</li>')
      time.sleep(0.6)
      yield "[INFO] Please wait. don't refresh browser until the finished... <br>\n"
      f.write('<li>[INFO] Please wait. dont refresh browser until the finished...</li>')
      f.write('<li>[INFO] Training process finish...</li></ul>')
      f.close()

      hist = model.fit(x_train, y_train,
      batch_size=num_batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
          callbacks=[callbacks],
          verbose=1)
          #jalakan proses pelatihan mode
      model.save(nama_model)
      #simpan model jika pelatihan selesa
      save_chart_loss_acc(hist)
      #buat grafik pelatihan
      acc, loss = model.evaluate(x_test,y_test, verbose=0)
      #lakukan evaluasi

      #yield "<li>Accuracy : {:.2f} Loss : {:.2f} </li>\n".format(acc, loss) 
      # f.write('<li>Accuracy : {:.2f} Loss : {:.2f} </li>'.format(acc, loss))
      yield "[INFO] Training process finish... <a href='/training'>back to admin</a> \n"
      training_process = False
      print("training selesai")
      print(nama_model)
  except Exception as ex:
    app.logger.error(f'{ex}')
    return '0'
    #tampilan error pada log

  return app.response_class(generate())
  #berikan respon pada browser

@app.route('/training', methods=['GET', 'POST'])
def training():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404

  train_model = False
  cursor.execute('SELECT * FROM hyperparam where id_hyperparam=1')
  rowData = cursor.fetchone()
  #ambil nilai hyper parameter

  if request.method == 'POST':
    #cek kiriman data dari form
    if 'train_model' in list(request.form):
      sql = "UPDATE hyperparam SET epoch=%s, batch_size=%s WHERE id_hyperparam=1"
      val = (request.form['epoch'], request.form['batch_size'])
      cursor.execute(sql, val)
      conn.commit()
      #update nilai hyper paramer pada tabel
      train_model = False
      return redirect(url_for('trainmodel'))
      #kembalikan ke halaman train nidek

  f = open("temp/log_train.txt", "r")
  #buka file log train
  return render_template('training.html', train_model=train_model, training_process=training_process, row=rowData, log=f.read())
  #tampilkan halaman training

@app.route("/stoptrain")
def stoptrain():
  global training_process
  training_process=False
  #hentikan proses training
  return redirect(url_for('training'))
  #kembalikan ke halaman training

@app.route('/testing', methods=['GET', 'POST'])
def testing():
  if 'loggedin' not in session:
    return render_template('404.html')
  #jika admin belum login munculkan 404
  
  fileTesting = "temp/test.pkl"
  #ambil file testing
  if request.method == 'POST' and os.path.exists(fileTesting):
    test = pickle.load(open(fileTesting, "rb"))
    model = load_model('models/audio_model.h5')
    #panggi model
    y_pred = model.predict(test["x_test"])
    # Argmax will classify the classes which has the highest probability 
    y_predc=y_pred.argmax(axis=1)
    y_testc=test["y_test"].argmax(axis=1)
    #Plot confusion matrix
    cm = confusion_matrix(y_testc, y_predc)
    matrix = pd.DataFrame(data=cm, columns=class_nama, index=class_nama)
    plt.figure()
    sns.heatmap(data=matrix, vmin=-1, vmax=1, annot=True, cmap=plt.cm.Greens_r)
    #buat confusion matrix
    cmChart = 'static/grafik/confusion_matrix.png'
    if os.path.exists(cmChart):
      os.remove(cmChart)
      #hapus confusion matrik lama
      plt.savefig(cmChart)
      #buat gambar confusion baru

    # Plot classification report
    report=classification_report(y_testc, y_predc, target_names=class_nama, output_dict=True)
    #buat report klasifikasi
    reportFile = open("temp/report.pkl", "wb")
    pickle.dump(report, reportFile)
    #simpan dalam bentuk file pikle
    reportFile.close()
  else:
    report = pickle.load(open("temp/report.pkl", "rb"))
    #buka file report 

  return render_template('testing.html', report=report)
  #tampilkan halaman testing

def prediksi(f):
    global nama_model
    global num_columns
    CNNmodel = load_model(nama_model)
    # panggil trained model
    test_features = extract_features(f)
    # lakukan ekstraksi fitur
    test_features = np.asarray(test_features)
    # ubah ke array
    test_features = test_features.reshape(num_channels, num_rows, num_columns, 1)
    # ubah dimensi fitur
    prediction = CNNmodel.predict(test_features)
    # lakukan prediksi
    class_idx = np.argmax(prediction)
    class_prob = prediction[0][class_idx]
    class_percent = "{:.2%}".format(class_prob)
    percent_notrecog = "{:.2%}".format(1 - class_prob)
    class_name = class_nama[class_idx]

    if class_prob > 0.5:
        return class_name, class_percent
    else:
        return "Not Recognized", percent_notrecog

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global OUTPUT
    global PROB

    if not conn:
        openDb()

    #OUTPUT = 'Not Recognized'
    f = request.files['audio_data']
    temp_file = secure_filename(f.filename)
    # ambil nama file
    f.save(f'static/voice_upload/{temp_file}')
    # simpan file audio
    OUTPUT, PROB = prediksi(f'static/voice_upload/{temp_file}')
    # lakukan prediksi
    #if OUTPUT != 'Not Recognized':
        #print('savedata')
    sql = "INSERT INTO log_identifikasi (nama_file_log, hasil_id, probabilitas, tanggal) VALUES (%s, %s, %s, %s)"
    val = (temp_file, OUTPUT, PROB, datetime.now())
    cursor.execute(sql, val)
    conn.commit()
        # simpan hasil prediksi ke tabel log identifikasi
    return '0'

OUTPUT = 'Not Recognized'
PROB = '0'
@app.route('/')
def main():
    OUTPUT = 'Not Recognized'
    PROB = '0'
    return render_template('index.html', result=0)
    # tampilan halaman index identifikasi

@app.route('/result')
def result():
    return render_template('index.html', result=1, output=OUTPUT, probabilitas=PROB)
    # tampilkan halaman hasil identifikasi

if __name__ == '__main__':
  # Start the Flask server in a new thread
  openDb()
  #jalakan koneksi database
  logging.basicConfig(filename='static/error.log',level=logging.DEBUG)
  #lakukan logging
  app.run(host='0.0.0.0',port=8000, debug=True)
  #jalankan server web flask