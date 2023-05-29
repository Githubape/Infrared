import os
import os.path as osp
import time
import subprocess


import numpy as np
import cv2


from helpers.utils import parse_args

import argparse
from PIL import Image

width=0
height=0
def dealimg(img):
    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 从RGB色彩空间转换到HSV色彩空间
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    # H、S、V范围一：
    lower1 = np.array([0, 43, 46])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)  # mask1 为二值图像
    res1 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask1)

    # H、S、V范围二：
    lower2 = np.array([156, 43, 46])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    res2 = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask2)

    # 将两个二值图像结果 相加
    mask3 = mask1 + mask2

    fimage = img.copy()
    #############################################################################################################################
    # ret, thresh = cv2.threshold(mask3, 254, 256, cv2.THRESH_BINARY_INV)  # [7-45]夜间(200-210) 白热 下午（240，250）
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #draw = 0
    #print("countours:"+str(len(contours)))
    # for c in contours:
    #     if len(c) < 80:  # 去除外轮廓  #夜间 [5-55]白热 下午[20,60]
    #         continue
    #     if len(c) > 160:
    #         continue
    #     draw = draw + 1
    #     if draw > 50:
    #         continue
    #
    #     # 找到边界坐标
    #     # x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
    #     # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红
    #     # 找面积最小的矩形
    #     rect = cv2.minAreaRect(c)  # ((cx, cy), (bw, bh), angle), 浮点数
    #     # 得到最小矩形的坐标
    #     box = cv2.boxPoints(rect)  # 最小外接矩形的四个点坐标,浮点数
    #     # 标准化坐标到整数
    #     box = np.int0(box)  # 四个点坐标转为整数
    #     # 画出边界
    #     ############################################
    #     #cv2.drawContours(fimage, [box], 0, (0, 255, 0), 1)  # 以轮廓线形式画出最小外接矩形，绿色
    #     #cenx = int((box[0][0] + box[3][0]) / 2)
    #     #ceny = int((box[0][1] + box[3][1]) / 2)
    #     ############################################
    #
    #
    #     #print(len(c))
    #     # if (cenx > 960 or ceny > 720):
    #     #     continue
    #     # print(cenx)
    #     # print(ceny)
    #     centem = 35
    #     #print(centem)
    #     #cv2.putText(fimage, str(centem), (cenx, ceny), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (0, 255, 0), 2)
    #     if (isshow):
    #         cv2.imshow("02", fimage)
    #         cv2.waitKey(1)
    #print("draw"+str(draw))
    #############################################################################################################################
    # 结果显示

    mask3=cv2.cvtColor(mask3, cv2.COLOR_GRAY2RGB)
    img2=cv2.hconcat([mask3,img])
    img2=cv2.resize(img2,(width,height))
    #cv2.namedWindow("img2", cv2.WINDOW_FREERATIO);
    # cv2.imshow("mask3", mask3)
    # cv2.imshow("img", img)
    #cv2.imshow("img2", img2)
    # cv2.imshow("Mask1", mask1)
    # cv2.imshow("res1", res1)
    # cv2.imshow("Mask2", mask2)
    # cv2.imshow("res2", res2)
    # cv2.imshow("grid_RGB", grid_RGB[:, :, ::-1])  # imshow()函数传入的变量也要为b g r通道顺序
    #cv2.waitKey(1)
    return img2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UAV')

    parser.add_argument('--source', type=str, default='test.MP4',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--sourceout', type=str, default='rtmp://218.192.100.219/live/livestream5',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--show', type=str, default=True,
                        help='showimg')
    parser.add_argument('--pipe', type=str, default=False,
                        help='pipe stream')

    # 0. Config file?
    parser.add_argument('--config-file', default='configs/config.yaml', help='Path to configuration file')

    args = parse_args(parser)
    print(args)


    rtmpout = args.sourceout #"rtmp://218.192.100.219/live/livestream1"
    isshow = args.show

    if(isshow==False):
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

    num=1
    while (cap.isOpened()):
        if(ret):
            # if ret == True:
            #outVideo.write(image)
            print("number"+str(num))
            num=num+1
            if(isshow):
                pass
                #cv2.imshow("0",image)
                # cv2.imwrite("/home/beidou/1.JPG",image)
                #cv2.waitKey(1)
            #img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fimage=dealimg(image)

            if(isshow):
                cv2.imshow("out",fimage)
                cv2.waitKey(1)



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
                #print("get")
            else:
                print("loss")
                # self.imgs[i] = np.zeros_like(self.imgs[i])
                # cap.open(stream)  # re-open stream if signal was lost
        #print(ret)
