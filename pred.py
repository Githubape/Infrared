import os
import os.path as osp
import time
import subprocess
import torch

from torchvision import transforms

import numpy as np
import cv2
import yaml

from helpers.utils import Metrics, parse_args, pred_to_rgb

import argparse
from PIL import Image

def gettemp():
    pass

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UAV')

    parser.add_argument('--source', type=str, default='test.MP4',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--sourceout', type=str, default='rtmp://218.192.100.219/live/livestream5',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--show', type=str, default=False,
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--dataset', type=str, default='aero',
                        help='file/dir/URL/glob/screen/0(webcam)')

    parser.add_argument('--best_model_path', type=str, default='/home/beidou/A-Project/AI/segment/UAVSeg/savedmodels/aero/deeplabv3plus+mobilenetv2_78.pth',
                        help='file/dir/URL/glob/screen/0(webcam)')
    # 0. Config file?
    parser.add_argument('--config-file', default='configs/config.yaml', help='Path to configuration file')

    # 1. Data Loading
    parser.add_argument('--bands', default=0, type=int)
    parser.add_argument('--use_augs', action='store_false', help='Use data augmentations?')
    parser.add_argument('--n_classes', default=12, type=int, help='Number of classes')
    parser.add_argument('--ignore_index', default=0, type=int, help='ignore index')

    # 2. Network selections
    # Which network?
    parser.add_argument('--network_arch', default='deeplabv3+', help='Network architecture?')

    # # Save weights post network config
    # parser.add_argument('--best_model_path', default=None, help='Path to save Network weights')

    # Use GPU or not
    parser.add_argument('--use_GPU', action='store_true', help='use GPUs?')

    # Hyperparameters
    parser.add_argument('--batch-size', default=4, type=int, help='Number of images sampled per minibatch?')
    parser.add_argument('--init_weights', default='kaiming', help="Choose from: 'normal', 'xavier', 'kaiming'")

    # Pretrained representation present?
    parser.add_argument('--pretrained_weights', default=None, help='Path to pretrained weights for network')

    args = parse_args(parser)
    print(args)


    rtmpout = args.sourceout #"rtmp://218.192.100.219/live/livestream1"
    isshow = args.show
    command = ['ffmpeg',
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', '960*720',  # 根据输入视频尺寸填写
                    '-r', '25',
                    '-i', '-',
                    '-c:v', 'h264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-flvflags', 'no_duration_filesize',
                    '-f', 'flv',
                    rtmpout]
    # 创建、管理子进程
    pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    rtmp_str = args.source  # "rtmp://218.192.100.219/live/livestream3"
    # 通过cv2中的类获取视频流操作对象cap
    cap = cv2.VideoCapture(rtmp_str)
    # 调用cv2方法获取cap的视频帧（帧：每秒多少张图片）
    # fps = self.cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    # 获取cap视频流的每帧大小
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    print(size)
    # # 定义编码格式mpge-4
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    # # 定义视频文件输入对象
    # outVideo = cv2.VideoWriter('saveDir1.avi', fourcc, fps, size)


    ret, image = cap.read()
    while(image is None):
        ret, image = cap.read()
        print("Nome")
    #image=cv2.resize(image,(600,500))
    # cv2.imshow(image,"o")
    #cv2.waitKey(20)
    num=1
    while (cap.isOpened()):
        if(ret):
            # if ret == True:
            #outVideo.write(image)
            print(num)
            num=num+1
            if(isshow):
                cv2.imshow("0",image)
                # cv2.imwrite("/home/beidou/1.JPG",image)
                cv2.waitKey(1)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            r = cv2.split(image)[2]

            if(isshow):
                cv2.imshow("015", img)
                cv2.waitKey(1)
                cv2.imshow("r", r)
                cv2.waitKey(1)
            ret, thresh = cv2.threshold(img, 235, 250, cv2.THRESH_BINARY_INV)  #[7-45]夜间(200-210) 白热 下午（240，250）
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            fimage = image.copy()
            print("contournums:"+str(len(contours)))
            draw=0
            for c in contours:
                #print("lenofc"+str(len(c)))
                if len(c) < 20:  # 去除外轮廓  #夜间 [5-55]白热 下午[20,60]
                    continue
                if len(c)>60:
                    continue
                draw=draw+1
                if draw>50:
                    continue
                # 找到边界坐标
                # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红
                # 找面积最小的矩形
                rect = cv2.minAreaRect(c)  # ((cx, cy), (bw, bh), angle), 浮点数
                # 得到最小矩形的坐标
                box = cv2.boxPoints(rect)  # 最小外接矩形的四个点坐标,浮点数
                # 标准化坐标到整数
                box = np.int0(box)  # 四个点坐标转为整数
                # 画出边界

                cv2.drawContours(fimage, [box], 0, (0, 255, 0), 1)  # 以轮廓线形式画出最小外接矩形，绿色
                cenx=int((box[0][0]+box[3][0])/2)
                ceny=int((box[0][1]+box[3][1])/2)

                print(len(c))
                if(cenx>960 or ceny>720):
                    continue
                print(cenx)
                print(ceny)
                centem=int(img[ceny-1][cenx-1]/255*38-6)
                print(centem)
                cv2.putText(fimage, str(centem), (cenx,ceny), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
                if(isshow):
                    cv2.imshow("02", fimage)
                    cv2.waitKey(1)




            print("draw"+str(draw))
            if(isshow==False):
                pipe.stdin.write(fimage.tobytes())


            # cv2.imshow('video', label_pred)
            # cv2.imshow('video2', imgadd1)

            # cv2.waitKey(int(1000 / int(fps)))  # 延迟
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     outVideo.release()
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     break
            cap.grab()  # .read() = .grab() followed by .retrieve()

            success, im = cap.retrieve()
            if success:
                image = im
                print("get")
            else:
                print("loss")
                # self.imgs[i] = np.zeros_like(self.imgs[i])
                # cap.open(stream)  # re-open stream if signal was lost
        print(ret)
