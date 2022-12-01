import multiprocessing
import requests
import numpy as np
import time
from flask import Flask,request
import pickle
import argparse
import CreateCameraApp
import json


def read_json(filename):
    file = open(filename, 'r')
    string_camera_info = file.read()
    camera_info = json.loads(string_camera_info)
    file.close()
    # print(camera_info)
    return camera_info


def pipeline(edge_addr,edge_resolution,edge_fps,cloud_addr,cloud_resolution,cloud_fps,step,task_place,camera_num):
    tttt=time.time()
    frames_edge=np.random.randint(0,256,(edge_fps,3,edge_resolution,edge_resolution//3*4),np.uint8)
    frames_cloud=np.random.randint(0,256,(cloud_fps,3,cloud_resolution,cloud_resolution//3*4),np.uint8)
    for task,pos in task_place.items():
        print(task,pos)
        if pos==0:
            data=pickle.dumps(frames_edge)
            r=requests.post("http://{}:{}/task/{}/{}".format(edge_addr["addr"],edge_addr["port"],camera_num,task),data=data)
            ans=pickle.loads(r.content)
        elif pos==1 and step!=0:
            data=pickle.dumps(frames_cloud)
            r=requests.post("http://{}:{}/task/{}/{}".format(edge_addr["addr"],edge_addr["port"],camera_num,task),data=data)
            ans=pickle.loads(r.content)
        else:
            data=pickle.dumps(frames_cloud)
            r=requests.post("http://{}:{}/task/{}/{}".format(cloud_addr["addr"],cloud_addr["port"],camera_num,task),data=data)
            ans=pickle.loads(r.content)
    print("time=",time.time()-tttt)




RES_MAP = [360, 600, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("camera_num")
    args=parser.parse_args()
    camera_num=args.camera_num
    init_camera_json = 'camera_app-{m,8,4}.json'
    region_num = 2
    camera_info = read_json(init_camera_json)
    task=camera_info[camera_num]
    scheduler_addr={"addr":"127.0.0.1","port":7000}
    r=requests.post("http://{}:{}/task_register".format(scheduler_addr["addr"],scheduler_addr["port"]),json={"camera":camera_num,"task":task})
    configs=r.json()
    edge_addr=configs["edge"]
    cloud_addr=configs["cloud"]
    config=configs["config"]
    if len(config["config"])==3:
        cloud_resolution=RES_MAP[int(config["config"][0])]
        cloud_fps=FPS_MAP[int(config["config"][1])]
        edge_resolution=0
        edge_fps=0
        step=int(config["config"][2])
    elif len(config["config"])==2:
        edge_resolution=RES_MAP[int(config["config"][0])]
        edge_fps=FPS_MAP[int(config["config"][1])]
        cloud_resolution=0
        cloud_fps=0
        step=-1
    else:
        edge_resolution=RES_MAP[int(config["config"][1])]
        edge_fps=FPS_MAP[int(config["config"][2])]
        cloud_resolution=RES_MAP[int(config["config"][4])]
        cloud_fps=FPS_MAP[int(config["config"][5])]
        step=int(config["config"][6])
    task_place=dict()
    for i in range(len(config["name"])):
        task_place[config["name"][i]]=config["place"][i]
    print(edge_resolution,edge_fps,cloud_resolution,cloud_fps)
    for _ in range(2):
        multiprocessing.Process(target=pipeline,args=(edge_addr,edge_resolution,edge_fps,cloud_addr,cloud_resolution,cloud_fps,step,task_place,camera_num)).start()
        time.sleep(1)
    
