# Start from the base image
FROM agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /app

# Copy project files
COPY main.py ./
COPY pipeline_v1.bin ./

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir fastapi uvicorn scikit-learn==1.6.1

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
