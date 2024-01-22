FROM python:3.10-slim

COPY . /

RUN pip install joblib 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN  python3 train_meta.py