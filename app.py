from flask import Flask, render_template, url_for , request , send_from_directory
import numpy as np
import math
import tensorflow as tf
import pickle

import os
import sys
import glob
from werkzeug.utils import secure_filename

#keras_modules
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

app = Flask(__name__)


MODEL_PATH1 = 'model1_vgg19.h5'
#Malaria

model1 = load_model(MODEL_PATH1)


#Heart
model2 = pickle.load(open('Linear_regression_Heart_model.pkl','rb'))


#Diabetes
model3 = pickle.load(open('random_forest.pkl','rb'))

UPLOAD_FOLDER = 'uploads'

#Cancer
model4 = pickle.load(open('Random_Forest_UsingGridSearch.pkl','rb'))

def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))

    x = image.img_to_array(img)
    x=np.expand_dims(x,axis=0) 

    x = x* 1.0/255

    

    preds = np.argmax(model.predict(x),axis=1) 

    if preds==1:
        preds = "Not Infected"
    else:
        preds="Infected"

    return preds

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)



@app.route('/')
def index():
    return render_template('try.html')



@ app.route('/Diabetes')
def Diabetes():
    title = 'Diabetes'
    return render_template('Diabetes.html', title=title)


@ app.route('/Heart')
def Heart():
    title = 'Heart'
    return render_template('Heart.html', title=title)



@ app.route('/Malaria')
def Malaria():
    title = 'Malaria'
    return render_template('Malaria.html', title=title)



@ app.route('/Cancer')
def Cancer():
    title = 'Cancer'
    return render_template('Cancer.html', title=title)




#Diabetes
@app.route("/predict1", methods=['POST'])
def predict1():
    if request.method == 'POST':
        Pregnancies= int(request.form['Pregnancies'])
        Glucose = int(request.form['Glucose'])
        BloodPressure = int(request.form['BloodPressure'])
        SkinThickness = int(request.form['SkinThickness'])
        Insulin = int(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = int(request.form['Age'])
        prediction = model3.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        disease = 'Diabetes'
        return render_template('after.html',data=prediction,disease = disease)

#Heart
@app.route("/predict2", methods=['POST'])
def predict2():
    if request.method == 'POST':
        Age= int(request.form['Age'])
        Sex = int(request.form['Sex'])
        CP = int(request.form['CP'])
        Trestbps = int(request.form['Trestbps'])
        Trestbps = math.log(Trestbps)
        Chol = int(request.form['Chol'])
        Chol = math.log(Chol)
        Restecg = float(request.form['Restecg'])
        Thalach = float(request.form['Thalach'])
        Exang = int(request.form['Exang'])
        Oldpeak = int(request.form['Oldpeak'])
        Slope = int(request.form['Slope'])
        Thal = int(request.form['Thal'])
        prediction = model2.predict([[Age,Sex,CP,Trestbps,Chol,Restecg,Thalach,Exang,Oldpeak,Slope,Thal]])
        disease = 'Heart Disease'
        return render_template('after.html',data=prediction,disease = disease)

#Cancer
@app.route("/predict4", methods=['POST'])
def predict4():
    if request.method == 'POST':
        Feature_1 = float(request.form['Feature 1'])
        Feature_2 = float(request.form['Feature 2'])
        Feature_3 = float(request.form['Feature 3'])
        Feature_4 = float(request.form['Feature 4'])
        Feature_5 = float(request.form['Feature 5'])
        Feature_6 = float(request.form['Feature 6'])
        Feature_7 = float(request.form['Feature 7'])
        Feature_8 = float(request.form['Feature 8'])
        Feature_9 = float(request.form['Feature 9'])
        Feature_10 = float(request.form['Feature 10'])
        prediction = model4.predict([[Feature_1,Feature_2,Feature_3,Feature_4,Feature_5,Feature_6,Feature_7,Feature_8,Feature_9,Feature_10]])
        disease = 'Cancer'
        return render_template('after.html',data=prediction,disease=disease)





#Malaria
@app.route('/predict3',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path,model1)
        return render_template("pred.html",data=preds,image_file_name=f.filename)
    return None









if __name__ == "__main__":
    app.run(debug=True