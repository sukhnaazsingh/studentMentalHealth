import pandas as pd
from sqlalchemy import create_engine

# Database connection details
DB_NAME = "student_mental_health"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create database engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Query to load data from the table
query = "SELECT * FROM student_mental_health"

# Load data into pandas DataFrame
df = pd.read_sql(query, engine)

# Display the first few rows of the DataFrame
print(df.head())

# You can now perform further analysis on the DataFrame
# For example, let's get some basic statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# # Example: Count the number of students with depression
# depression_count = df['Do_you_have_Depression?'].value_counts()
# print(depression_count)

df = df.dropna(how='any',axis=0)
print(df.isnull().sum())

plt.figure(figsize=(10,10))
plt.hist(df['Age'],color='b')
plt.title("Age distribution");
