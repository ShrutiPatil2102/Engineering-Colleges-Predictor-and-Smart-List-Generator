import pandas as pd
import mysql.connector


# Step 1: Load the dataset
data = pd.read_csv("merged_data.csv")
data = data[['Collegeid', 'College Name', 'Review Rating']]
group_cid_label= data.groupby(['Collegeid','College Name','Review Rating'])
B1=group_cid_label.first()


#Delete FROM `collegelist`

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="studentadmission"
    )

    cursor = conn.cursor()

    # Step 3: Insert grouped data into MySQL database table
    for index, row in B1.iterrows():
        college_id, college_name, review_rating = index
        sql = "INSERT INTO collegelist (ID,CollegeName,Rating) VALUES (%s, %s, %s)"
        val = (college_id, college_name, review_rating)
        cursor.execute(sql, val)
    
    # Commit changes and close connection
    conn.commit()
    print("Data inserted successfully!")

except mysql.connector.Error as err:
    print("Error:", err)

finally:
    if conn:
        conn.close()
