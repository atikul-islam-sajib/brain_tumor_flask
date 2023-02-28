
from flask import Flask, render_template, request, redirect, flash, url_for
import cv2
import urllib.request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.utils as image
import keras

UPLOAD_FOLDER = '/config/workspace/static'

application = Flask(__name__)
app = application

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return "No file"
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return "File"
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            
            model = keras.models.load_model('/config/workspace/brain_tumor.h5', compile=False)


            Image = image.load_img('static/'+filename, target_size = (150, 150))
            Image_array_ = np.asarray(Image)
            Image_array_ = image.img_to_array(Image)
            Image_array_numpy = (Image_array_/255)
            test_data = np.expand_dims(Image_array_numpy, axis = 0)
            predicted_ =  model.predict(test_data)
            predicted_ = np.argmax(predicted_, axis = 1)
            return render_template('index.html', result = predicted_[0], img = 'static/'+filename)

if __name__=="__main__":
    application.run(host="0.0.0.0", port = 5002, debug=True)
