
# Context-Aware Chat Application

This repository contains a Streamlit application that uses OpenAI's GPT-4 model to provide context-aware chat functionality. The application allows users to select data from preloaded CSV files, newly uploaded CSV files, or a connected PostgreSQL database to provide context for the chat model.

## Setup Instructions

### Clone the Repository

```sh
git clone https://github.com/jf2fa/sapp.git
cd sapp
```

### Create a Virtual Environment and Install Requirements

1. **Create a virtual environment:**
   ```sh
   python -m venv env
   ```

2. **Activate the virtual environment:**
   - On Windows:
     ```sh
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source env/bin/activate
     ```

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

### Create and Populate the PostgreSQL Database

1. **Ensure PostgreSQL is installed and running on your system.**

2. **Run the database setup script:**
   ```sh
   python setup_database.py
   ```

### Build and Run the Docker Image

1. **Build the Docker image:**
   ```sh
   docker build -t context-app .
   ```

2. **Run the Docker container:**
   ```sh
   docker run -p 8501:8501 context-app
   ```

### Access the Application

Open your web browser and go to `http://localhost:8501` to access the Streamlit application.

## Project Structure

```
project-root/
│
├── app.py
├── auth_module.py
├── chat_module.py
├── data_management.py
├── setup_database.py
├── logoapp.png
├── synthetic_survey_responses_1.csv
├── synthetic_survey_responses_2.csv
├── synthetic_survey_responses_3.csv
├── synthetic_survey_responses.csv
├── requirements.txt
└── Dockerfile
```

## Adding New CSV Files

To add new CSV files for context, upload them through the "Data Management" tab in the application. The selected files will be used as context for the chat model.

## Connecting to a PostgreSQL Database

You can connect to a PostgreSQL database by entering the connection string and a SQL query in the "Data Management" tab. The results of the query can be added as context for the chat model.

## Updating OpenAI API Credentials

You can update the OpenAI API credentials in the "Model Configuration" tab of the application.

## License

This project is licensed under the MIT License.
