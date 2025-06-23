# Use an official Python image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port Flask will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
