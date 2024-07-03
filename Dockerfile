# Use the official Python image from the Docker Hub
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install wheel
RUN pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
