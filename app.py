from keras.models import load_model
from flask import Flask, render_template,request
from random import shuffle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)
APP_ROOT=os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

RESIM_BOYUTU=100
TEST_KLASORU='images'

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    
    if not os.path.isdir(target):
        os.mkdir(target)
        
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        
        destination = "/".join([target, filename])
        print("Save it to:", destination)
        file.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("upload.html")

@app.route("/tahmin", methods=["GET"])
def tahmin():
    testing_data = []
    str_label=[]
    for img in (os.listdir(TEST_KLASORU)):
        path = os.path.join(TEST_KLASORU,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (RESIM_BOYUTU,RESIM_BOYUTU))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    tahmin_verisi=np.load('test_data.npy', allow_pickle=True)
    
    for no, veri in enumerate(tahmin_verisi[:]):
     
     resim_no = veri[1]
     resim_verisi = veri[0]
     
     orig = resim_verisi
     veri = resim_verisi.reshape(-1,RESIM_BOYUTU, RESIM_BOYUTU, 1)
     ag_cikisi = model.predict([veri])[0]
     
     if np.argmax(ag_cikisi) == 0:
         str_label = 'çamaşır makinesi'
     elif np.argmax(ag_cikisi) == 1:
         str_label = 'bulaşık makinesi'
     elif np.argmax(ag_cikisi) == 2:
         str_label = 'Buzdolabı'
     else:
         str_label = 'tanımsız'
    camasir=round((ag_cikisi[0]*100),2)
    bulasik=round((ag_cikisi[1]*100),2)
    buzdolabi=round((ag_cikisi[2]*100),2)
    
    print('camasir makinesi %',camasir)
    print('bulasik makinesi %',bulasik)
    print('buzdolabi %',buzdolabi)
    return render_template("complete.html",total=str_label)

def get_model():
    global model
    model=load_model('testt')
    print('loaded model')

if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	get_model()
	app.run(debug = True, threaded = False)
