# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your project files
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies using uv
RUN pip install --no-cache-dir uvicorn fastapi pandas scikit-learn seaborn matplotlib

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

