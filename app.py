from flask import Flask, render_template, redirect, url_for, flash
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
import io
import base64
import os
import json

# Use a non-GUI backend for Matplotlib
plt.switch_backend('Agg')

app = Flask(__name__, template_folder='.')
app.secret_key = 'supersecretkey'  # Necessary for flashing messages

# Database connection details
DB_NAME = "student_mental_health"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"


def load_data_from_kaggle():
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

    # Database connection details
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Insert data into the table
    df.to_sql('student_mental_health', engine, if_exists='replace', index=False)

    print("Data has been successfully inserted into the database.")


@app.route('/reload-data')
def reload_data():
    load_data_from_kaggle()
    flash('Data has been successfully reloaded!', 'success')
    return redirect(url_for('index'))


@app.route('/')
def index():
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

    # Get some basic statistics
    print(df.describe())

    # Check for missing values
    print(df.isnull().sum())

    # Drop rows with any missing values
    df = df.dropna(how='any', axis=0)
    print(df.isnull().sum())

    df.rename(columns={'Choose your gender': 'gender'}, inplace=True)

    df['Your current year of Study'] = df['Your current year of Study'].str.lower().str.capitalize()

    df['CGPA_gender'] = df['What is your CGPA?'].apply(
        lambda x: (float(x.split('-')[0].strip()) + float(x.split('-')[1].strip())) / 2)

    # Change the type of the column to float
    df['CGPA_gender'] = df['CGPA_gender'].astype(float)

    total_students = len(df)
    anxiety_count = len(df[df['Do you have Anxiety?'] == 'Yes'])
    depression_count = len(df[df['Do you have Depression?'] == 'Yes'])
    both_count = len(df[(df['Do you have Anxiety?'] == 'Yes') & (df['Do you have Depression?'] == 'Yes')])

    anxiety_count_percentage = (anxiety_count / total_students) * 100
    depression_count_percentage = (depression_count / total_students) * 100
    both_count_percentage = (both_count / total_students) * 100

    years_of_study = df['Your current year of Study'].unique()
    genders = df['gender'].unique()
    avg_cgpa = []

    for year in years_of_study:
        year_data = {'year': year}
        for gender in genders:
            gender_year_df = df[(df['Your current year of Study'] == year) & (df['gender'] == gender)]

            if gender_year_df.empty:
                print(f"No data for {gender} students in year {year}")
                continue

            avg_cgpa_value = gender_year_df['CGPA_gender'].mean()
            year_data[f'{gender.lower()}_avg'] = float("{:.2f}".format(avg_cgpa_value))
            print(f"Average CGPA for {gender} students: {avg_cgpa_value:.2f}")

        if 'female_avg' not in year_data:
            year_data['female_avg'] = None
        if 'male_avg' not in year_data:
            year_data['male_avg'] = None

        avg_cgpa.append(year_data)

    # Create age distribution plot with axis labels
    plt.figure(figsize=(10, 10))
    plt.hist(df['Age'], color='b')
    plt.title("Age distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    age_distribution_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create gender distribution plot
    plt.figure(figsize=(10, 10))
    plt.title("Gender distribution")
    plt.pie(df.gender.value_counts(), explode=(0.025, 0.025), labels=df.gender.value_counts().index,
            colors=['skyblue', 'navajowhite'], autopct='%1.1f%%', startangle=180)
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    gender_distribution_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create study per year distribution plot
    plt.figure(figsize=(10, 10))
    sns.countplot(data=df, x='Your current year of Study', hue='gender')
    plt.title("Students studying in particular year")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    study_per_year_distribution_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create anxiety by depression plot
    plt.figure(figsize=(10, 10))
    sns.countplot(data=df, x='Do you have Anxiety?', hue='Do you have Depression?')
    plt.title("Anxiety by Depression")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_by_depression_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create anxiety by gender plot
    plt.figure(figsize=(10, 10))
    sns.set_theme(style="darkgrid")
    sns.countplot(y="Do you have Anxiety?", hue="gender", data=df)
    plt.title("Anxiety by Gender")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_by_gender_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create anxiety by study year plot
    plt.figure(figsize=(10, 10))
    sns.set_theme(style="darkgrid")
    sns.countplot(x="Do you have Anxiety?", hue="Your current year of Study", data=df)
    plt.title("Anxiety by study year")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_by_study_year_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create depression by study year plot
    plt.figure(figsize=(10, 10))
    sns.set_theme(style="darkgrid")
    sns.countplot(x="Do you have Depression?", hue="Your current year of Study", data=df)
    plt.title("Depression by study year")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    depression_by_study_year_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Filtering and converting to a list
    filtered_df = df[(df['What is your course?'] == 'Engineering') & (df['Do you have Anxiety?'] == 'Yes') & (
            df['Do you have Depression?'] == 'Yes')]
    filtered_entries = filtered_df.values.tolist()

    # P-Value Analysis
    # Calculate the correlation and p-value
    correlation, p_value = pearsonr(df['Age'], df['CGPA_gender'])
    correlation = float("{:.2f}".format(correlation))
    p_value = float("{:.2f}".format(p_value))

    # Print the correlation and p-value
    print("Correlation:", correlation)
    print("P-value:", p_value)

    # Statistical analysis: Chi-square test for independence between Anxiety and Depression
    contingency_table = pd.crosstab(df['Do you have Anxiety?'], df['Do you have Depression?'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2 = float("{:.2f}".format(chi2))
    p = float("{:.2f}".format(p))
    dof = float("{:.2f}".format(dof))

    print(f"Chi-square test results:\nChi2: {chi2}\nP-value: {p}\nDegrees of freedom: {dof}")

    return render_template('index.html',
                           total_students=total_students,
                           anxiety_count=anxiety_count, anxiety_count_percentage=anxiety_count_percentage,
                           depression_count=depression_count, depression_count_percentage=depression_count_percentage,
                           both_count=both_count, both_count_percentage=both_count_percentage,
                           avg_cgpa=avg_cgpa,
                           age_distribution_plot=age_distribution_plot,
                           gender_distribution_plot=gender_distribution_plot,
                           study_per_year_distribution_plot=study_per_year_distribution_plot,
                           anxiety_by_depression_plot=anxiety_by_depression_plot,
                           anxiety_by_gender_plot=anxiety_by_gender_plot,
                           anxiety_by_study_year_plot=anxiety_by_study_year_plot,
                           depression_by_study_year_plot=depression_by_study_year_plot,
                           correlation=correlation, p_value=p_value,
                           chi2=chi2, p_value_chi2=p, dof=dof, filtered_entries=filtered_entries)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
