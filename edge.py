import multiprocessing
import requests
import numpy as np
import time
from flask import Flask,request
import pickle
import ffmpeg
from pathlib import Path
import cv2
import ComputationResource


app=Flask(__name__)
RES_MAP = [360, 600, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
models=dict()
cloud_addr={"addr":"127.0.0.1","port":6000}

@app.route("/task/<cameranum>/<tasknum>",methods=["POST"])
def pipeline(cameranum,tasknum):
    print(len(request.data),request.data[:50])
    open(f"{cameranum}_{tasknum}.mp4","wb").write(request.data)
    frame,err=ffmpeg.input(f"{cameranum}_{tasknum}.mp4",format="mp4").output("pipe:",format="rawvideo",pix_fmt="rgb24").overwrite_output().run(input=request.data,capture_stdout=True)
    probe = ffmpeg.probe(f"{cameranum}_{tasknum}.mp4")
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    frame=np.frombuffer(frame,np.uint8).reshape([-1,height, width, 3])
    print(frame.shape)
    model,step=models[(cameranum,tasknum)]
    # dummy
    for i in range(step):
        frame=np.ones(frame.shape)*frame
    if step<5:
        frame_flattened=np.reshape(frame,[frame.shape[0]*frame.shape[1],frame.shape[2]*frame.shape[3]])
        mx=np.max(np.abs(frame_flattened))
        f=np.math.ceil(np.math.log2(mx/128))
        frame_flattened=(frame_flattened/(1<<f)).astype(np.int8)
        cv2.imwrite(f"{cameranum}_{tasknum}.png",frame_flattened)
        data=open(f"{cameranum}_{tasknum}.png","rb").read()
        data=pickle.dumps({"shape":frame.shape,"factor":f,"data":data})
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
                
@app.route("/get_computation_resource", methods=["POST"])
def get_computation_resource():
    return str(ComputationResource.query_computation_resource())
