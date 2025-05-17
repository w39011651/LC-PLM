#FOR T4 GPU
FROM 2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY . /app

RUN pip install --no-build-isolation mamba-ssm

RUN pip install transformers==4.51.3

RUN pip install --no-chche-dir -r requirements.txt