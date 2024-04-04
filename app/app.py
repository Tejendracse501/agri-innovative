# Your Flask app routes
from flask import Flask, render_template, url_for, request
import numpy as np
import requests
import pickle
from utils import config
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
    return render_template('home.html')
#====================================================DASHBOARD=================================================================================
#Backend code for Dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

#===================================================SERVICES==================================================================================
#Backend code for Services page
@app.route('/services')
def services():
    return render_template('services.html')
#-----------------------------------------------------------SUITABLE MARKET---------------------------------------------------------------------------------

@app.route('/suitable_market')
def suitable_market():
    return render_template('suitable_market.html')
#---------------------------------------------------------------CROP RECOMMENDATION-----------------------------------------------------------------------------
#Loading Crop Recommendation Models
crop_recommendation_model_path = 'models/crop_recommendation2.pkl'
minmax_scaler_path = 'models/minmax_scaler.pkl'
@app.route('/crop_recommendation')
def crop_recommendation():
    

    return render_template('crop_recommendation.html')


#-----------------------------------------------------------------DISEASE IDENTIFICATION---------------------------------------------------------------------------
@app.route('/disease_identification')
def  disease_identification():
    return render_template('disease_identification.html')

#------------------------------------------------------------------FETILIZER RECOMMENDATION--------------------------------------------------------------------------
@app.route('/fertilizer_recommendation')
def fertilizer_recommendation():
    return render_template('fertilizer_recommendation.html')
#---------------------------------------------------------------------CHECK WEATHER-----------------------------------------------------------------------
@app.route('/check_weather')
def check_weather():
    return render_template('check_weather.html')

#==============================================================NEWS=======================================================================
#Backend code for News page
@app.route('/news')
def news():
    return render_template('news.html')

#==============================================================CONNECT=======================================================================
#Backend code for Connect page
@app.route('/connect')
def connect():
    return render_template('connect.html')

#==============================================================LOGIN SIGNUP=======================================================================

#Backend code for Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


#=====================================================================================================================================
#Backend code for Help page
@app.route('/help')
def help():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(debug=True)
