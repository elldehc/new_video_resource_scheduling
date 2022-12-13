import multiprocessing
import requests
import numpy as np
import time
from flask import Flask,request
import pickle
import ffmpeg
from pathlib import Path
import cv2
from model import Model
from dataset.voc2007 import VOC2007
from backbone.resnet50 import ResNet50
from backbone.resnet101 import ResNet101,resnet101front
from config.eval_config import EvalConfig as Config
import torch
import ComputationResource


app=Flask(__name__)
RES_MAP = [360, 600, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
RESNET50_CHECKPOINT="model-22500.pth"
RESNET101_CHECKPOINT="model-180000.pth"
models=dict()
cloud_addr={"addr":"127.0.0.1","port":6000}

@app.route("/task/<cameranum>/<tasknum>",methods=["POST"])
def pipeline(cameranum,tasknum):
    with torch.no_grad():
        # print(len(request.data),request.data[:50])
        open(f"{cameranum}_{tasknum}.mp4","wb").write(request.data)
        frame,err=ffmpeg.input(f"{cameranum}_{tasknum}.mp4",format="mp4").output("pipe:",format="rawvideo",pix_fmt="rgb24").overwrite_output().run(input=request.data,quiet=True)
        probe = ffmpeg.probe(f"{cameranum}_{tasknum}.mp4")
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        frame=np.frombuffer(frame,np.uint8).reshape([-1,height, width, 3]).astype(np.float32)
        frame=torch.from_numpy(np.transpose(frame,(0,3,1,2)))
        print(frame.shape,frame.requires_grad)
        image_shape=frame.shape
        model,step=models[(cameranum,tasknum)]
        if step<=5:
            frame=model.eval().forward(frame.cuda())
            print(frame.requires_grad)
            frame=frame.cpu().numpy()
            frame_flattened=np.reshape(frame,[frame.shape[0]*frame.shape[1],frame.shape[2]*frame.shape[3]])
            mx=np.max(np.abs(frame_flattened))
            f=np.math.ceil(np.math.log2(mx/128))
            frame_flattened=(frame_flattened/(1<<f)).astype(np.uint8)
            retval,data=cv2.imencode(".png",frame_flattened)
            data=pickle.dumps({"shape":frame.shape,"image_shape":image_shape,"factor":f,"data":data})
            r=requests.post("http://{}:{}/task/{}/{}".format(cloud_addr["addr"],cloud_addr["port"],cameranum,tasknum),data=data)
            del frame
            del frame_flattened
            return r.content
        else:
            detection_bboxes, detection_classes, detection_probs, _ =model.eval().forward(frame.cuda())
            del frame
            return pickle.dumps({"detection_bboxes":detection_bboxes.cpu().numpy(),"detection_classes":detection_classes.cpu().numpy(),"detection_probs":detection_probs.cpu().numpy()})

@app.route("/config",methods=["POST"])
def config():
    with torch.no_grad():
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
                    # model=fasterrcnn_resnet50_fpn(True,False)
                    model=Model(ResNet50(pretrained=True),VOC2007.num_classes(), pooler_mode=Config.POOLER_MODE,
                        anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                        rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
                    model.load(RESNET50_CHECKPOINT)
                    model.eval()
                    models[(cam,confs[cam]["name"][i])]=(model,6)
                else:
                    model=resnet101front(int(cc[2]),True).cuda()
                    model.eval()
                    model.load_state_dict(torch.load(RESNET101_CHECKPOINT),strict=False)
                    models[(cam,confs[cam]["name"][i])]=(model,int(cc[2]))
            
    return "ok"
                
@app.route("/get_computation_resource", methods=["POST"])
def get_computation_resource():
    return str(ComputationResource.query_computation_resource())
