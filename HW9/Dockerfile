FROM agrigorev/model-2025-hairstyle:v1

# Install runtime deps
RUN pip install --no-cache-dir pillow numpy onnxruntime

# Copy handler
COPY lambda_function.py .

# Default command for AWS Lambda base images expects the runtime to receive
# invocations; the CMD below is a placeholder (actual AWS Lambda behavior differs).
CMD ["lambda_function.lambda_handler"]