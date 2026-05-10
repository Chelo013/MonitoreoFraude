FROM python:3.9-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all the model files and api backend
COPY . /code

# HuggingFace Spaces runs on port 7860 by default
CMD ["uvicorn", "api_paysim_backend:app", "--host", "0.0.0.0", "--port", "7860"]
