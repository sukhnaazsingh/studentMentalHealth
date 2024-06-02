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
import numpy as np

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


def set_kaggle_credentials():
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        try:
            with open('kaggle.json') as f:
                data = json.load(f)
                kaggle_username = data['username']
                kaggle_key = data['key']
        except FileNotFoundError:
            raise FileNotFoundError("Kaggle credentials not found in environment variables or kaggle.json file")

    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key


def load_data_from_kaggle():
    set_kaggle_credentials()

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


def analyze_data(df):
    # Drop any rows with missing values
    df = df.dropna(how='any', axis=0)

    # Rename columns for easier access
    df.rename(columns={'Choose your gender': 'gender',
                       'What is your course?': 'course',
                       'Your current year of Study': 'year',
                       'Age': 'age',
                       'What is your CGPA?': 'cgpa',
                       'Marital status': 'marital_status',
                       'Do you have Depression?': 'depression',
                       'Do you have Anxiety?': 'anxiety',
                       'Do you have Panic attack?': 'panic_attack',
                       'Did you seek any specialist for a treatment?': 'seek_any_specialist_for_treatment'},
              inplace=True)

    # Standardize the 'year' column
    df['year'] = df['year'].str.lower().str.capitalize()

    # Calculate the average CGPA and convert to float
    df['CGPA_average'] = df['cgpa'].apply(
        lambda x: (float(x.split('-')[0].strip()) + float(x.split('-')[1].strip())) / 2)
    df['CGPA_average'] = df['CGPA_average'].astype(float)

    # Convert categorical columns to binary numeric values
    df['depression'] = df['depression'].map({'Yes': 1, 'No': 0})
    df['anxiety'] = df['anxiety'].map({'Yes': 1, 'No': 0})

    return df


def plot_academic_performance(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='CGPA_average', y='depression', data=df, logistic=True)
    plt.title('Correlation between CGPA and Depression')
    plt.xlabel('CGPA')
    plt.ylabel('Depression (Yes=1, No=0)')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    depression_vs_cgpa = base64.b64encode(img.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.regplot(x='CGPA_average', y='anxiety', data=df, logistic=True)
    plt.title('Correlation between CGPA and Anxiety')
    plt.xlabel('CGPA')
    plt.ylabel('Anxiety (Yes=1, No=0)')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_vs_cgpa = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return depression_vs_cgpa, anxiety_vs_cgpa


def plot_gender_differences(df):
    plt.figure(figsize=(10, 6))
    sns.lmplot(x='CGPA_average', y='depression', hue='gender', data=df, logistic=True)
    plt.title('Correlation between CGPA and Depression by Gender')
    plt.xlabel('CGPA')
    plt.ylabel('Depression (Yes=1, No=0)')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    gender_depression_vs_cgpa = base64.b64encode(img.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lmplot(x='CGPA_average', y='anxiety', hue='gender', data=df, logistic=True)
    plt.title('Correlation between CGPA and Anxiety by Gender')
    plt.xlabel('CGPA')
    plt.ylabel('Anxiety (Yes=1, No=0)')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    gender_anxiety_vs_cgpa = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return gender_depression_vs_cgpa, gender_anxiety_vs_cgpa


def plot_cgpa_groups(df):
    high_cgpa = df[df['CGPA_average'] >= df['CGPA_average'].median()]
    low_cgpa = df[df['CGPA_average'] < df['CGPA_average'].median()]

    high_cgpa_counts = high_cgpa['depression'].value_counts(normalize=True)
    low_cgpa_counts = low_cgpa['depression'].value_counts(normalize=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=high_cgpa_counts.index, y=high_cgpa_counts.values)
    plt.title('Depression Prevalence in High CGPA Students')
    plt.xlabel('Do you have Depression?')
    plt.ylabel('Proportion')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    high_cgpa_depression = base64.b64encode(img.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=low_cgpa_counts.index, y=low_cgpa_counts.values)
    plt.title('Depression Prevalence in Low CGPA Students')
    plt.xlabel('Do you have Depression?')
    plt.ylabel('Proportion')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    low_cgpa_depression = base64.b64encode(img.getvalue()).decode()
    plt.close()

    high_cgpa_counts = high_cgpa['anxiety'].value_counts(normalize=True)
    low_cgpa_counts = low_cgpa['anxiety'].value_counts(normalize=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=high_cgpa_counts.index, y=high_cgpa_counts.values)
    plt.title('Anxiety Prevalence in High CGPA Students')
    plt.xlabel('Do you have Anxiety?')
    plt.ylabel('Proportion')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    high_cgpa_anxiety = base64.b64encode(img.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=low_cgpa_counts.index, y=low_cgpa_counts.values)
    plt.title('Anxiety Prevalence in Low CGPA Students')
    plt.xlabel('Do you have Anxiety?')
    plt.ylabel('Proportion')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    low_cgpa_anxiety = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return high_cgpa_depression, low_cgpa_depression, high_cgpa_anxiety, low_cgpa_anxiety


@app.route('/')
def index():
    # Create database engine
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Query to load data from the table
    query = "SELECT * FROM student_mental_health"
    df = pd.read_sql(query, engine)

    # Perform data analysis
    df = analyze_data(df)

    # Generate plots
    depression_vs_cgpa, anxiety_vs_cgpa = plot_academic_performance(df)
    gender_depression_vs_cgpa, gender_anxiety_vs_cgpa = plot_gender_differences(df)
    high_cgpa_depression, low_cgpa_depression, high_cgpa_anxiety, low_cgpa_anxiety = plot_cgpa_groups(df)

    # Other statistics
    total_students = len(df)
    anxiety_count = len(df[df['anxiety'] == 1])
    depression_count = len(df[df['depression'] == 1])
    both_count = len(df[(df['anxiety'] == 1) & (df['depression'] == 1)])
    anxiety_count_percentage = (anxiety_count / total_students) * 100
    depression_count_percentage = (depression_count / total_students) * 100
    both_count_percentage = (both_count / total_students) * 100

    years_of_study = df['year'].unique()
    genders = df['gender'].unique()
    avg_cgpa = []

    for year in years_of_study:
        year_data = {'year': year}
        for gender in genders:
            gender_year_df = df[(df['year'] == year) & (df['gender'] == gender)]

            if gender_year_df.empty:
                print(f"No data for {gender} students in year {year}")
                continue

            avg_cgpa_value = gender_year_df['CGPA_average'].mean()
            year_data[f'{gender.lower()}_avg'] = float("{:.2f}".format(avg_cgpa_value))
            print(f"Average CGPA for {gender} students: {avg_cgpa_value:.2f}")

        if 'female_avg' not in year_data:
            year_data['female_avg'] = None
        if 'male_avg' not in year_data:
            year_data['male_avg'] = None

        avg_cgpa.append(year_data)

    # Create age distribution plot with axis labels
    plt.figure(figsize=(10, 10))
    plt.hist(df['age'], color='b')
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
    sns.countplot(data=df, x='year', hue='gender')
    plt.title("Students studying in particular year")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    study_per_year_distribution_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create anxiety by depression plot
    plt.figure(figsize=(10, 10))
    sns.countplot(data=df, x='anxiety', hue='depression')
    plt.title("Anxiety by Depression")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_by_depression_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create anxiety by gender plot
    plt.figure(figsize=(10, 10))
    sns.set_theme(style="darkgrid")
    sns.countplot(y="anxiety", hue="gender", data=df)
    plt.title("Anxiety by Gender")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_by_gender_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create anxiety by study year plot
    plt.figure(figsize=(10, 10))
    sns.set_theme(style="darkgrid")
    sns.countplot(x="anxiety", hue="year", data=df)
    plt.title("Anxiety by study year")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    anxiety_by_study_year_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Create depression by study year plot
    plt.figure(figsize=(10, 10))
    sns.set_theme(style="darkgrid")
    sns.countplot(x="depression", hue="year", data=df)
    plt.title("Depression by study year")
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    depression_by_study_year_plot = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # Filtering and converting to a list
    filtered_df = df[(df['course'] == 'Engineering') & (df['anxiety'] == 1) & (
            df['depression'] == 1)]
    filtered_entries = filtered_df.values.tolist()

    # P-Value Analysis
    # Calculate the correlation and p-value
    correlation, p_value = pearsonr(df['age'], df['CGPA_average'])
    correlation = float("{:.2f}".format(correlation))
    p_value = float("{:.2f}".format(p_value))

    # Print the correlation and p-value
    print("Correlation:", correlation)
    print("P-value:", p_value)

    # Statistical analysis: Chi-square test for independence between Anxiety and Depression
    contingency_table = pd.crosstab(df['anxiety'], df['depression'])
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
                           chi2=chi2, p_value_chi2=p, dof=dof,
                           depression_vs_cgpa=depression_vs_cgpa,
                           anxiety_vs_cgpa=anxiety_vs_cgpa,
                           gender_depression_vs_cgpa=gender_depression_vs_cgpa,
                           gender_anxiety_vs_cgpa=gender_anxiety_vs_cgpa,
                           high_cgpa_depression=high_cgpa_depression,
                           low_cgpa_depression=low_cgpa_depression,
                           high_cgpa_anxiety=high_cgpa_anxiety,
                           low_cgpa_anxiety=low_cgpa_anxiety,
                           filtered_entries=filtered_entries)


if __name__ == '__main__':
    app.run(port=8000, debug=True)
