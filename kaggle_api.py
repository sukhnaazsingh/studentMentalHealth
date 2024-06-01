import os
import json
import pandas as pd
from sqlalchemy import create_engine

# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

# API credentials for Kaggle
with open('kaggle.json') as f:
    data = json.load(f)

os.environ['KAGGLE_USERNAME'] = data['username']
os.environ['KAGGLE_KEY'] = data['key']

from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize API
api = KaggleApi()
api.authenticate()

# Download file
api.dataset_download_file('shariful07/student-mental-health', 'Student Mental health.csv')

# Read data to pandas data frame
df = pd.read_csv('Student%20Mental%20health.csv', sep=',')
print(df.head())

# Preprocess column names to replace spaces with underscores
#df.columns = [col.replace(' ', '_') for col in df.columns]
print(df.head())

# Database connection details
DB_NAME = "student_mental_health"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create database engine
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Insert data into the table
df.to_sql('student_mental_health', engine, if_exists='replace', index=False)

print("Data has been successfully inserted into the database.")
