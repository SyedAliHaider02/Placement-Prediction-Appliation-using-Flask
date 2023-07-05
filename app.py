from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open('placement.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def man():

 return render_template('front.html')



@app.route('/predict', methods=['GET', 'POST'])
def home():
    data1 = request.form['id']
    data2 = request.form['gender']
    data3 = request.form['marks1']
    data4 = request.form['boards1']
    data5 = request.form['marks2']
    data6 = request.form['boards2']
    data7 = request.form['strm']
    data8 = request.form['deg_p']
    data9 = request.form['deg_s']
    data10 = request.form['wrx']
    data11 = request.form['amcat']
    data12 = request.form['sp']
    data13 = request.form['mba_p']
    data14 = request.form['sal']
   
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14]])
    pred = model.predict(arr)
    return render_template('home.html', data=pred,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7,data8=data8,data9=data9,data10=data10,data11=data11,data12=data12,data13=data13,data14=data14)


if __name__=="__main__":
    app.run(debug=True)