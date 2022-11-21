import multiprocessing
import requests
import numpy as np
import time
from flask import Flask,request
import pickle


app=Flask(__name__)
RES_MAP = [360, 600, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
pipelines=[]

def pipeline(config,cameranum,tasknum):
    if len(config)==3:
        resolution=RES_MAP[int(config[0])]
        fps=FPS_MAP[int(config[1])]
        step=int(config[2])
    else:
        resolution=RES_MAP[int(config[0])]
        fps=FPS_MAP[int(config[1])]
        step=5
    
    for _ in range(10):
        # decodes video
        # dummy
        frame=np.random.random((fps,3,resolution,resolution//3*4))

        # process
        tttt=time.time()
        
        # dummy
        for i in range(step):
            frame=np.random.random((fps,3,resolution,resolution//3*4))*frame
        if step<5:
            ans=pickle.loads(requests.post(f"http://127.0.0.1:6000/task/{cameranum}/{tasknum}",data=pickle.dumps(frame)).content)
        else:
            ans=np.mean(frame)
        # print(ans.shape)
        tttt=time.time()-tttt
        print("time=",tttt)

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
                assert ec!="aa"
                p=multiprocessing.Process(target=pipeline,args=(ec,cam,confs[cam]["name"][i]))
                p.start()
                pipelines.append(p)
            else:
                assert cc!="aa"
                p=multiprocessing.Process(target=pipeline,args=(cc,cam,confs[cam]["name"][i]))
                p.start()
                pipelines.append(p)
    return "ok"
                
