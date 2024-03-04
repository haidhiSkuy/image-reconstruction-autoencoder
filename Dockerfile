FROM python:3.11 

WORKDIR /app 

COPY . /app

ENV PIP_ROOT_USER_ACTION=ignore

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt 

ENTRYPOINT ["python", "train.py"]