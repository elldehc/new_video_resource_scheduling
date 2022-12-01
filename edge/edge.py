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
cloud_addr={"addr":"127.0.0.1","port":6000}

@app.route("/task/<cameranum>/<tasknum>",methods=["POST"])
def pipeline(cameranum,tasknum):
    frame=pickle.loads(request.data)
    print(frame.shape)
    model,step=models[(cameranum,tasknum)]
    # dummy
    for i in range(step):
        frame=np.ones(frame.shape)*frame
    if step<5:
        data=pickle.dumps(frame)
        r=requests.post("http://{}:{}/task/{}/{}".format(cloud_addr["addr"],cloud_addr["port"],cameranum,tasknum),data=data)
        return r.content
    else:
        ans=np.mean(frame)
        return pickle.dumps(ans)


@app.route("/config",methods=["POST"])
def config():
    js=request.get_json()
    confs=js["config"]
    tasks=js["task"]
    for cam in tasks:
        conf=confs[cam]["config"]
        if len(conf)==7:
            ec=conf[1:3]
            cc=conf[4:7]
        elif len(conf)==3:
            ec="aa"
            cc=conf
        else:
            ec=conf
            cc="aaa"
        n_task=len(confs[cam]["place"])
        for i in range(n_task):
            if confs[cam]["place"][i]==0:
                models[(cam,confs[cam]["name"][i])]=("dummy"+str(ec),5)
            else:
                models[(cam,confs[cam]["name"][i])]=("dummy"+str(cc),int(cc[2]))
        
    return "ok"
                
