FROM python:3.10-slim

# Set up a safe working directory inside the container
WORKDIR /code

# Copy and install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of your backend scripts and joblib weights
COPY . .

# Hugging Face Spaces strictly requires containers to expose port 7860
EXPOSE 7860

# Launch the FastAPI app via Uvicorn on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]