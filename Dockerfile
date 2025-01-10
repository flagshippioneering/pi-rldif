# Use an official Python runtime as a parent image, specifying version 3.11.5
FROM python:3.11.5-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy only the requirements file first to install dependencies
COPY env/requirements.txt ./

# Install system packages required for building PyTorch extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && apt-get clean

# Install PyTorch first
RUN pip install torch==2.0.0

# Install other required Python packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application using the module path
CMD ["python", "-m", "run.run"]