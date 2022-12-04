import multiprocessing
import requests
import numpy as np
import time
from flask import Flask,request
import pickle


app=Flask(__name__)
RES_MAP = [360, 600, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
models=dict()

@app.route("/task/<cameranum>/<tasknum>",methods=["POST"])
def pipeline(cameranum,tasknum):
    frame=pickle.loads(request.data)
    print(frame.shape)
    model=models[(cameranum,tasknum)]
    # dummy
    step=int(model[5:])
    for i in range(step,5):
        frame=np.ones(frame.shape)*frame
    ans=np.mean(frame)
    return pickle.dumps(ans)


@app.route("/config",methods=["POST"])
def config():
    js=request.get_json()
    confs=js["config"]
    tasks=js["task"]
    # print(confs)
    for cam in tasks:
        # print(cam)
        # print(type(confs))
        # print(confs[cam])
        conf=confs[cam]["config"]
        if len(conf)==7:
            cc=conf[4:7]
        elif len(conf)==3:
            cc=conf
        else:
            cc="aaa"
        n_task=len(confs[cam]["place"])
        for i in range(n_task):
            if confs[cam]["place"][i]!=0:
                assert cc!="aa"
                models[(cam,confs[cam]["name"][i])]="dummy"+str(cc)
    return "ok"