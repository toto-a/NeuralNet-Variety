FROM jupyter/scipy-notebook

RUN pip install joblib 

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY siren.py ./siren.py
COPY train_siren.py ./train_siren.py

RUN python3 train_siren.py