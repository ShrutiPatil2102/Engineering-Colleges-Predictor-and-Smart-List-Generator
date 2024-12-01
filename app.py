from distutils.log import debug
from fileinput import filename
import os
import io
from os import environ
from flask import *
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from flask_cors import CORS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.pipeline import make_pipeline
# Example using NLTK for data cleaning
#from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
app.secret_key = "abc"

@app.route('/')  
def main():
    #return render_template("signin.html")
    Respon=make_response("hii")
    return Respon

@app.route('/signin')  
def signin():  
    Respon=make_response("hii")
    return Respon

@app.route('/success', methods = ['POST','GET'])  
def success():
    Alldata="Successfully"
    if request.method == 'POST':
        Percen = request.form['Twelfth']
        CETSCORE = request.form['CETSCORE']
        CRate = request.form['Rating']

        # Step 1: Load the dataset
        data = pd.read_csv("merged_data.csv")

        group_cid_label= data.groupby(['Collegeid','College Name'])
        B1=group_cid_label.first()
        Collegeindex = dict(B1.index)
        print('College Name and ID -')
        print(Collegeindex)

        # Step 2: Preprocess the data
        # Remove unnecessary columns
        data = data[['Collegeid','Percentage', 'CETSCORE', 'Review Rating']]

        # Handle missing values (if any)
        data.dropna(inplace=True)

        # Step 3: Define admission criteria
        # Define threshold values for percentage and review rating
        percentage_threshold = int(Percen)  # Example: 70%
        review_rating_threshold = int(CRate)  # Example: 4 stars out of 5
        CETSCORE_threshold = int(CETSCORE)

        # Create the admission column based on the criteria
        #data['Admission'] = (data['Percentage'] <= percentage_threshold) & (data['CETSCORE'] <= CETSCORE_threshold) & (data['Review Rating'] >= review_rating_threshold)
        #data['Admission'] = (data['Percentage'] <= percentage_threshold) & (data['CETSCORE'] <= CETSCORE_threshold)
        #data['Admission'] = (data['Percentage'] >= percentage_threshold-5) & (data['Percentage'] <= percentage_threshold+5) &(data['CETSCORE'] >= CETSCORE_threshold-5) & (data['CETSCORE'] <= CETSCORE_threshold+5)
        data['Admission'] =  (data['Percentage'] <= percentage_threshold+10) & (data['CETSCORE'] <= CETSCORE_threshold+2)

        print(len(data['Admission']))
        # Step 4: Split the dataset into features and target variable
        X = data[['CETSCORE', 'Review Rating']]
        y = data['Admission']

        # Step 5: Train a machine learning model
        X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Step 6: Evaluate model accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", accuracy)

        # Step 7: Generate confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Step 6: Predict college admissions for a list of colleges
        def predict_admissions(college_list):
            predictions = []
            for college in college_list:
                CETSCORE = college['CETSCORE']
                review_rating = college['Review Rating']
                admission_prediction = model.predict([[CETSCORE,review_rating]])
                if admission_prediction==True and college['Review Rating']>=review_rating_threshold:
                    #print(college['Review Rating'])
                    predictions.append((college['Collegeid'],college['Review Rating'],admission_prediction[0]))
            print(predictions)
            return predictions

        # Step 2: Create college_list from data
        unique_colleges = data.drop_duplicates(subset=['Collegeid'])[['Collegeid', 'CETSCORE', 'Review Rating']]

        college_list = []
        for index, row in unique_colleges.iterrows():
            college_data = {
                'Collegeid': row['Collegeid'],
                'CETSCORE': int(CETSCORE),
                'Review Rating': row['Review Rating']
            }
            college_list.append(college_data)

        Alldata="<table><tr><th>No.</th><th></th><th>College</th><th>Rating</th></tr>"
        TRcount=0
        admission_predictions = predict_admissions(college_list)
        for college,ReviewRating,admission in admission_predictions[0:20]:
            TRcount=TRcount+1
            #print(str(college)+"--"+Collegeindex[int(college)]+"--"+str(ReviewRating)+"--"+str(admission))
            #Alldata=Alldata+"<tr><td>"+str(TRcount)+"</td><td> <input type='checkbox' name='items1[]' value='"+str(college)+"' checked></td><td>"+str(college)+"</td><td>"+Collegeindex[int(college)]+"</td><td>"+str(ReviewRating)+"</td></tr>"
            Alldata=Alldata+"<tr><td>"+str(TRcount)+"</td><td> <input type='checkbox' name='items1[]' value='"+str(int(college))+"' checked></td><td>"+Collegeindex[int(college)]+"</td><td>"+str(ReviewRating)+"</td></tr>"


        Alldata=Alldata+"</table>"

    Respon=make_response(Alldata)
    return Respon

@app.route('/Mainpage', methods=['GET'])  
def Mainpage():
    Respon=make_response("")
    #return Respon
    #return render_template("Mainpage.html")
    return Respon
            
@app.route('/shutdown')
def shutdown():
    sys.exit()
    os.exit(0)
    return
   
if __name__ == '__main__':
   HOST = environ.get('SERVER_HOST', '0.0.0.0')
   #HOST = environ.get('SERVER_HOST', 'localhost')
   try:
      PORT = int(environ.get('SERVER_PORT', '5555'))
   except ValueError:
      PORT = 5555
   app.run(HOST, PORT)
   #app.run(debug=True)
