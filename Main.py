import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file
file1_path = 'College.csv'
df1 = pd.read_csv(file1_path)


print(df1.head(5))

print(df1.info())

print(df1.describe())

print(df1.dtypes)

df1['Review Rating'] = df1['Review Rating'].apply(lambda x: pd.to_numeric(str(x).split(' ', 1)[0], errors='coerce'))

# Sort DataFrame by 'ID' for better visualization
df1.sort_values(by='ID', inplace=True)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(df1['ID'], df1['Review Rating'], color='blue')
plt.xlabel('College ID')
plt.ylabel('Review Rating')
plt.title('College ID vs Review Rating')
plt.show()


# Read the second CSV file
file2_path = 'Student.csv'  # Replace with the path to your second file
df2 = pd.read_csv(file2_path)

# Merge the two DataFrames on the common column 'CollegeID'
merged_df = pd.merge(df1, df2, left_on='ID', right_on='Collegeid')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)
