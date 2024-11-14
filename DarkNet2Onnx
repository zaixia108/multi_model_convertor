FROM mydocker.zaixia108.com/circleci/python:3.6-bullseye-node-browsers-legacy
LABEL authors="zaixia108"

ENTRYPOINT ["top", "-b"]

COPY pytorch-YOLOv4 /root/pytorch-YOLOv4

COPY 50k.cfg /root/pytorch-YOLOv4/cfg/50k.cfg
COPY 50k.weights /root/pytorch-YOLOv4/weights/50k.weights
COPY 50k.names /root/pytorch-YOLOv4/data/50k.names

WORKDIR /root/pytorch-YOLOv4

RUN sudo chmod 777 /root/pytorch-YOLOv4

RUN sudo pip install virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN sudo python -m venv venv

RUN sudo venv/bin/pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN sudo venv/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN sudo venv/bin/pip install fastapi uvicorn python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple

ENTRYPOINT ["sudo", "bash", "run.sh"]