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

@app.route("/task/<cameranum>/<tasknum>",methods=["POST"])
def pipeline(cameranum,tasknum):
    model=models[(cameranum,tasknum)]
    step=int(model[5:])
    if step==0:
        open(f"{cameranum}_{tasknum}.mp4","wb").write(request.data)
        frame,err=ffmpeg.input(f"{cameranum}_{tasknum}.mp4",format="mp4").output("pipe:",format="rawvideo",pix_fmt="rgb24").overwrite_output().run(input=request.data,capture_stdout=True)
        probe = ffmpeg.probe(f"{cameranum}_{tasknum}.mp4")
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        frame=np.frombuffer(frame,np.uint8).reshape([-1,height, width, 3])
        print(frame.shape)
    else:
        data=pickle.loads(request.data)
        shape=data["shape"]
        raw_frame=data["data"]
        f=data["factor"]
        frame=cv2.imdecode(np.frombuffer(raw_frame,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        frame=(np.reshape(frame,shape)*(1<<f)).astype(np.float32)
    print(frame.shape)
    # dummy
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


@app.route("/get_computation_resource", methods=["POST"])
def get_computation_resource():
    return str(ComputationResource.query_computation_resource())