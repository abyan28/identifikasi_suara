###****************************###
#  DEEP CNN LEARNING UTILITIES   #
##################################

###library form deep learning
import os
import numpy as np
import pandas as pd
import librosa
import pickle
import csv  
import tensorflow as tf
import random
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from werkzeug.utils import secure_filename
import pymysql.cursors

##Deep Learning Properties
tf.compat.v1.disable_eager_execution()
max_pad_len = 7223
dataset = 'static/voice'
features = []
header = ['file', 'feature','label']
num_rows = 40
num_columns = 7223
num_channels = 1
num_epochs = 100
num_batch_size = 32
nama_model="models/audio_model.h5"
class_nama=os.listdir(dataset)
class_nama.sort()
num_labels=len(class_nama)
#class_nama=['Abyan', 'Ilmi', 'Itsbat','Lana','Ulya']
#predict_threshold=0.75
accuracy_threshold = 1.0000
val_accuracy_threshold = 1.0000
training_process = False

### Model Fitting Callback
class myCallback(tf.keras.callbacks.Callback):
  def on_train_end(self, logs=None):
    global training_process
    training_process = False
    
  def on_epoch_end(self, epochs, logs={}) :
    global training_process
    global accuracy_threshold
    if((logs.get('accuracy') is not None and logs.get('accuracy') >= accuracy_threshold) 
      and (logs.get('val_accuracy') is not None and logs.get('val_accuracy') >= val_accuracy_threshold)) :
        print('\nSudah mencapai akurasi 100%, maka training dihentikan!')
        self.model.stop_training = True
        training_process = False

      
callbacks = myCallback()


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

# function to extract feature from audio
def extract_features(file_name):   
  try:
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    print(sample_rate)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    print(mfccs.shape)
    pad_width = max_pad_len - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')      
  except Exception as e:
    print("Error encountered while parsing file: ", file_name)
    return None 
  return mfccs

'''def feature_extraction(file_path):
  try:
    # load the audio file
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # extract features from the audio
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
  except Exception as e:
    print("Error encountered while parsing file: ", file_path)
    return None 
  return mfcc'''

## function to extract all audio
def create_metadata():
  global features
  features = []
  if not conn:
      openDb()

  with open("metadata.csv", 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    noc=0
    for i in class_nama:
      cursor.execute('SELECT nama_kelas, nama_file from kelas kls, dataset ds WHERE kls.id_kelas=ds.label AND kls.nama_kelas=%s', (i))
      results = cursor.fetchall()
      for row in results:
        dr=dataset+"/"+row['nama_kelas']+"/"+row['nama_file']
        data = extract_features(dr)
        #data = feature_extraction(dr)
        features.append([dr,data, noc])
        datacsv=[dr,data, noc]
        writer.writerow(datacsv)
      noc+=1

def data_train_test():
  global features
  global num_labels
  # Convert into a Panda dataframe 
  featuresdf = pd.DataFrame(features, columns=['file','feature','label'])
  featuresdf = shuffle(featuresdf)
  print('Finished feature extraction from ', len(featuresdf), 'files')

  # Convert features and corresponding classification labels into numpy arrays
  X = np.array(featuresdf.feature.tolist())
  y = np.array(featuresdf.label.tolist())
  print(X.shape)

  # Encode the classification labels
  le = LabelEncoder()
  yy = to_categorical(le.fit_transform(y)) 
  # # split the dataset 
  
  x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.3, random_state = 42)
  print(x_test.shape)

  x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
  x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)
  num_labels = yy.shape[1]
  data = {
    "x_test" : x_test,
    "y_test" : y_test
  }
  test = open("temp/test.pkl", "wb")
  pickle.dump(data, test)
  test.close()

  return x_train, y_train, x_test, y_test

def my_model():
  global num_labels
  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))

  model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))

  model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))

  model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Dropout(0.2))
  model.add(GlobalAveragePooling2D())

  model.add(Dense(num_labels, activation='softmax'))

  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

  return model

def save_chart_loss_acc(history):
  training_process=False
  plt.figure(figsize=(10, 4))
  plt.subplot(1, 2, 1)
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.plot(history.history['accuracy'], color='g')
  plt.plot(history.history['val_accuracy'], color='m')
  plt.legend(['train', 'val'], loc='upper left')

  plt.subplot(1, 2, 2)
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.plot(history.history['loss'], color='r')
  plt.plot(history.history['val_loss'], color='m')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig('static/grafik/train_chart.png')

###**********************###
#    END DEEP LEARNING     #
############################ 