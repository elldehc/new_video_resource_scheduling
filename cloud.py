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
from dataset.coco2017 import COCO2017
from backbone.resnet50 import ResNet50
from backbone.resnet101 import ResNet101,ResNet101Back
from config.eval_config import EvalConfig as Config
import torch


app=Flask(__name__)
RES_MAP = [360, 600, 720, 900, 1080]
FPS_MAP = [2, 3, 5, 10, 15]
RESNET50_CHECKPOINT="model-22500.pth"
RESNET101_CHECKPOINT="model-180000.pth"
models=dict()

@app.route("/task/<cameranum>/<tasknum>",methods=["POST"])
def pipeline(cameranum,tasknum):
    with torch.no_grad():
        model,step=models[(cameranum,tasknum)]
        if step==0:
            open(f"{cameranum}_{tasknum}.mp4","wb").write(request.data)
            frame,err=ffmpeg.input(f"{cameranum}_{tasknum}.mp4",format="mp4").output("pipe:",format="rawvideo",pix_fmt="rgb24").overwrite_output().run(input=request.data,capture_stdout=True)
            probe = ffmpeg.probe(f"{cameranum}_{tasknum}.mp4")
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            frame=np.frombuffer(frame,np.uint8).reshape([-1,height, width, 3]).astype(np.float32)
            frame=torch.from_numpy(np.transpose(frame,(0,3,1,2)))
            print(frame.shape)
            detection_bboxes, detection_classes, detection_probs, _ =model.eval().forward(frame.cuda())
            del frame
            return pickle.dumps({"detection_bboxes":detection_bboxes.cpu().numpy(),"detection_classes":detection_classes.cpu().numpy(),"detection_probs":detection_probs.cpu().numpy()})
        else:
            data=pickle.loads(request.data)
            shape=data["shape"]
            image_shape=data["image_shape"]
            raw_frame=data["data"]
            f=data["factor"]
            frame=cv2.imdecode(np.frombuffer(raw_frame,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            frame=(np.reshape(frame,shape)*(1<<f)).astype(np.float32)
            frame=torch.from_numpy(frame)
            print(frame.shape)
            detection_bboxes, detection_classes, detection_probs, _ =model.eval().forward(frame.cuda(),image_batch_shape=image_shape)
            del frame
            return pickle.dumps({"detection_bboxes":detection_bboxes.cpu().numpy(),"detection_classes":detection_classes.cpu().numpy(),"detection_probs":detection_probs.cpu().numpy()})


@app.route("/config",methods=["POST"])
def config():
    with torch.no_grad():
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
                    step=int(cc[2])
                    if step==0:
                        model=Model(ResNet101(pretrained=True),COCO2017.num_classes(), pooler_mode=Config.POOLER_MODE,
                            anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=[64,128,256,512],
                            rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
                        model.load(RESNET101_CHECKPOINT)
                        model.eval()
                    else:
                        model=Model(ResNet101Back(step,pretrained=True),COCO2017.num_classes(), pooler_mode=Config.POOLER_MODE,
                            anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=[64,128,256,512],
                            rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
                        model.load(RESNET101_CHECKPOINT)
                        model.eval()
                    models[(cam,confs[cam]["name"][i])]=(model,step)
    return "ok"