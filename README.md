# 📊 Student Mental Health Analysis Flask Application

Welcome to the Student Mental Health Analysis Flask Application! This application provides insights into student mental health data. Below you'll find all the instructions to get this application up and running locally or via Docker. 

## 🛠️ Features
- 📈 Visualize mental health data of students
- 📊 Perform various data analyses
- 🌐 Web interface for easy interaction

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Flask
- Docker (if running in a Docker container)
- Kaggle account with `kaggle.json` API key

### Database Setup
1. Change Database credentials
    ```bash
    python3 kaggle_api.py
    ```
### Local Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/mental-health-app.git
    cd mental-health-app
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up Kaggle API credentials**:
    Place your `kaggle.json` file in the root directory of the project.

4. **Run the application**:
    ```bash
    python3 app.py
    ```
    The application will be available at `http://localhost:8000`.

### Docker Setup

1. **Build the Docker image**:
    ```bash
    docker build -t students-mental-health-app .
    ```

2. **Run the Docker container**:
    ```bash
    docker run -p 8000:8000 -d \
      -e DB_USER=<DB_USER> \
      -e DB_PASSWORD=<DB_PASSWORD> \
      -e DB_HOST=<DB_HOST> \
      -e DB_PORT=<DB_PORT> \
      -e KAGGLE_USERNAME=<KAGGLE_USERNAME> \
      -e KAGGLE_KEY=<KAGGLE_KEY> \
      students-mental-health-app
    ```
    The application will be available at `http://localhost:8000`.

## 🌐 Live Demo

You can check out the live demo of the application here: [Student Mental Health Analysis](https://studentmentalhealth.onrender.com/home)

## 📂 Project Structure
```
mental-health-app/
│
├── app.py              # Main application file
├── kaggle_api.py       # For Data initialisation on Database
├── analyse.ipynb       # Jupyter Notebook for playing around with Data
├── requirements.txt    # Python dependencies
├── index.html          # Index HTML File for visualization
├── Dockerfile          # Dockerfile for Deployments
├── kaggle.json         # Kaggle API credentials (should be replaced/added by the user)
└── README.md           # This readme file
```

## 🤝 Contributing

Contributions are welcome! Please create a pull request with detailed information on the changes.

---

Feel free to reach out if you have any questions or need further assistance! Happy coding! 👨‍💻👩‍💻
