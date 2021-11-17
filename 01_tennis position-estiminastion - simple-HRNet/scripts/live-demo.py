# python scripts/live-demo.py --save_video --filename sample.mp4

import os
os.environ["PATH"] += r"D:\ffmpeg-N-103601-gb35653d4c4-win64-gpl\bin;"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np
import os
import json


sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations
from math import atan2, degrees

from inference import inference_video


def angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else (deg1 - deg2)


def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video, 
         video_format,video_framerate, device):
    
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
        nof_frames = video.get(cv2.CAP_PROP_FRAME_COUNT) #add
        
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()
    
    ##add##
    file_name = os.path.splitext(os.path.basename(filename))[0]

        
    if use_tiny_yolo:
         yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
         yolo_model_def="./models/detectors/yolo/config/yolov3.cfg"
         yolo_class_path="./models/detectors/yolo/data/coco.names"
         yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights"
    
    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes= not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0
    
    index = 0 #add
    
    key_name = ['nose', 
                'left_eye', 'right_eye', 
                'left_ear', 'right_ear', 
                'left_shoulder', 'right_shoulder', 
                'left_elbow', 'right_elbow', 
                'left_wrist', 'right_wrist', 
                'left_hip', 'right_hip', 
                'left_knee', 'right_knee', 
                'left_ankle', 'right_ankle']
    
    skelton=[[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]]
    
    left_elbow_angle_li = []
    left_shoulder_angle_li = []
    left_hip_body_angle_li = []
    left_hip_leg_angle_li = []
    left_knee_angle_li = []
    right_elbow_angle_li = []
    right_shoulder_angle_li = []
    right_hip_body_angle_li = []
    right_hip_leg_angle_li = []
    right_knee_angle_li = []
    body_center_li=[]

    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)

        if not disable_tracking:
            boxes, pts = pts

        if not disable_tracking:
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids
            
        

        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for _, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)

        fps = 1. / (time.time() - t)
        print('\rframerate: %f fps' % fps, end='')
        
        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter(file_name+'_keypoint.mp4', fourcc, video_framerate, (frame.shape[1], frame.shape[0]))
            video_writer.write(frame)
            
            
        if len(left_elbow_angle_li) == 0:
            for li in [left_elbow_angle_li, left_shoulder_angle_li,left_hip_body_angle_li, left_hip_leg_angle_li, left_knee_angle_li,
                       right_elbow_angle_li, right_shoulder_angle_li, right_hip_body_angle_li,right_hip_leg_angle_li, right_knee_angle_li, body_center_li]:
                for _ in range(len(pts)):
                    li.append([])
                    
        elif len(left_elbow_angle_li) < len(pts):
            for li in [left_elbow_angle_li, left_shoulder_angle_li,left_hip_body_angle_li, left_hip_leg_angle_li, left_knee_angle_li,
                       right_elbow_angle_li, right_shoulder_angle_li, right_hip_body_angle_li, right_hip_leg_angle_li, right_knee_angle_li, body_center_li]:
                for _ in range(len(pts) - len(li)):
                    li.append([])
                    
#         if len(skelton)!=len(pts):
        while len(skelton)!=len(pts):
            skelton.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]])
        
        for j, pt in enumerate(pts):
            keypoints=pt.tolist()
            for jj, sk_p in enumerate(keypoints):
                skelton[j][jj].append({'x':float(sk_p[1]), 'y':float(sk_p[0])})            
            
            left_elbow_angle = angle_between((keypoints[5][1], keypoints[5][0]), (keypoints[7][1], keypoints[7][0]), (keypoints[9][1], keypoints[9][0]))            
            left_shoulder_angle = angle_between((keypoints[7][1], keypoints[7][0]),(keypoints[6][1], keypoints[6][0]),(keypoints[5][1], keypoints[5][0]))
            left_hip_leg_angle = angle_between((keypoints[12][1], keypoints[12][0]),(keypoints[11][1], keypoints[11][0]),(keypoints[13][1], keypoints[13][0]))
            left_hip_body_angle = angle_between((keypoints[5][1], keypoints[5][0]),(keypoints[11][1], keypoints[11][0]),(keypoints[12][1], keypoints[12][0]))
            left_knee_angle = angle_between((keypoints[11][1], keypoints[11][0]),(keypoints[13][1], keypoints[13][0]),(keypoints[15][1], keypoints[15][0]))  
            
            right_elbow_angle = angle_between((keypoints[6][1], keypoints[6][0]),(keypoints[8][1], keypoints[8][0]),(keypoints[10][1], keypoints[10][0]))
            right_shoulder_angle = angle_between((keypoints[8][1], keypoints[8][0]),(keypoints[6][1], keypoints[6][0]),(keypoints[12][1], keypoints[12][0]))
            right_hip_body_angle = angle_between((keypoints[6][1], keypoints[6][0]),(keypoints[12][1], keypoints[12][0]),(keypoints[11][1], keypoints[11][0]))
            right_hip_leg_angle = angle_between((keypoints[11][1], keypoints[11][0]),(keypoints[12][1], keypoints[12][0]),(keypoints[14][1], keypoints[14][0]))
            right_knee_angle = angle_between((keypoints[12][1], keypoints[12][0]),(keypoints[14][1], keypoints[14][0]),(keypoints[16][1], keypoints[16][0]))
            body_center = {'x':(keypoints[5][1]+keypoints[6][1])/2, 'y':(keypoints[5][0]+keypoints[10][0])/2}
            
            left_elbow_angle_li[j].append(left_elbow_angle)
            left_shoulder_angle_li[j].append(left_shoulder_angle)
            left_hip_body_angle_li[j].append(left_hip_body_angle)
            left_hip_leg_angle_li[j].append(left_hip_leg_angle)
            left_knee_angle_li[j].append(left_knee_angle)
            right_elbow_angle_li[j].append(right_elbow_angle)
            right_shoulder_angle_li[j].append(right_shoulder_angle)
            right_hip_body_angle_li[j].append(right_hip_body_angle)
            right_hip_leg_angle_li[j].append(right_hip_leg_angle)
            right_knee_angle_li[j].append(right_knee_angle)
            body_center_li[j].append(body_center)
            
        fps = 1. / (time.time() - t)
        print('\rframe: % 4d / %d - framerate: %f fps ' % (index, nof_frames - 1, fps), end='')

        index += 1
    
    
    del pts
    pred_class = inference_video(np.array([filename]), filepath="./models/classifier/curing_bi_gru")
    
    dict_skelton_object=list()
    
    for sk in skelton:
        dict_skelton=list()
        for key, part in zip(key_name,sk):
            dict_skelton.append(part)
        dict_skelton_object.append(dict_skelton)    
        
    skelton_points_object=list()
    for dict_sk in dict_skelton_object:
        skelton_points =dict()
        
        for key, part_all in zip(key_name, dict_sk):
            print(key)
            skelton_points[key]=part_all
        skelton_points_object.append(skelton_points)
        
    objects = list()
    for pos, bc, le, ls, lhb, lhl, lk, re, rs, rhb, rhl, rk in zip(skelton_points_object, body_center_li,
                                  left_elbow_angle_li,left_shoulder_angle_li,left_hip_body_angle_li,left_hip_leg_angle_li, left_knee_angle_li,
                                  right_elbow_angle_li,right_shoulder_angle_li,right_hip_body_angle_li,right_hip_leg_angle_li, right_knee_angle_li):
        
        
        object_tmp = pos
        object_tmp = {'skelton': pos, 'position':bc, 
                      'left elbow angle': le, 'left shoulder angle':ls, 'left hip body angle':lhb, 'left hip leg angle':lhl, 'left knee angle':lk,
                      'right elbow angle': re, 'right shoulder angle':rs, 'right hip body angle':rhb, 'right hip leg angle':rhl, 'right knee angle':rk,}
        
        objects.append(object_tmp)
    
    with open(file_name+".json", 'w') as f:
        json_data = {'file name': filename, 'action': pred_class,  'object': objects}
        json.dump(json_data, f, indent=4)
        
        
    if save_video:
        video_writer.release()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection). Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                                     "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    
    args = parser.parse_args()
    main(**args.__dict__)
