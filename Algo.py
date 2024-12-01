import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Step 1: Load the dataset
data = pd.read_csv("merged_data.csv")

group_cid_label= data.groupby(['Collegeid','College Name'])
B1=group_cid_label.first()
Collegeindex = dict(B1.index)
print('College Name and ID -')
print(Collegeindex)

# Step 2: Preprocess the data
# Remove unnecessary columns
data = data[['Collegeid', 'Percentage', 'Review Rating']]

# Handle missing values (if any)
data.dropna(inplace=True)

# Step 3: Define admission criteria
# Define threshold values for percentage and review rating
percentage_threshold = 65  # Example: 70%
review_rating_threshold = 1  # Example: 4 stars out of 5

# Create the admission column based on the criteria
data['Admission'] = (data['Percentage'] <= percentage_threshold) & (data['Review Rating'] >= review_rating_threshold)

# Step 4: Split the dataset into features and target variable
X = data[['Percentage', 'Review Rating']]
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
        percentage = college['Percentage']
        review_rating = college['Review Rating']
        admission_prediction = model.predict([[percentage, review_rating]])
        if admission_prediction==True:
            predictions.append((college['Collegeid'],college['Review Rating'],admission_prediction[0]))
    return predictions
'''
# Example usage:
college_list = [
    {'College Name': 'ABC College', 'Percentage': 80, 'Review Rating': 4.5},
    {'College Name': 'XYZ College', 'Percentage': 75, 'Review Rating': 3.8},
    # Add more colleges to the list as needed
]
'''

# Step 2: Create college_list from data
unique_colleges = data.drop_duplicates(subset=['Collegeid'])[['Collegeid', 'Percentage', 'Review Rating']]

college_list = []
for index, row in unique_colleges.iterrows():
    college_data = {
        'Collegeid': row['Collegeid'],
        'Percentage': 65,
        'Review Rating': row['Review Rating']
    }
    college_list.append(college_data)

Alldata="<table><tr><th>ID</th><th>College ID</th><th>College</th><th>Rating</th></tr>"
TRcount=0
admission_predictions = predict_admissions(college_list)
for college,ReviewRating,admission in admission_predictions[0:30]:
    TRcount=TRcount+1
    #print(str(college)+"--"+Collegeindex[int(college)]+"--"+str(ReviewRating)+"--"+str(admission))
    Alldata=Alldata+"<tr><td>"+str(TRcount)+"</td><td>"+str(college)+"</td><td>"+Collegeindex[int(college)]+"</td><td>"+str(ReviewRating)+"</td></tr>"

Alldata=Alldata+"</table>"

print(Alldata)
