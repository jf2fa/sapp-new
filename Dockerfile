# Use the official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory contents into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run"]

# Pass the name of the Streamlit app as an argument to the entrypoint
CMD ["app.py"]
