import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.01, 'iou threshold')
flags.DEFINE_float('score', 0.1, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    ## make capture folder
    if not os.path.isdir('./tracking'):
        os.makedirs('./tracking')
    
    
    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps_ori = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps_ori, (width, height))

    frame_num = 0
    obj_num = 0
    label_dict = dict()
    track_li = [list() for i in range(10)]
    hide_frame_li = [0,0,0,0,0,0,0,0,0,0]
    track_num = 0
    hide_num=0
    same = False
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
  
        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people) -> 사람만 탐색
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
        
        ##########
        # id count
        ################
        
            #print(str(track.track_id))
            id_object = str(track.track_id)
            if not id_object in label_dict.keys():
                label_dict[id_object] = 1
            else:
                label_dict[id_object] = label_dict[id_object]+1 
            #print(label_dict)
            
            #for id_tmp in label_dict.keys():
            #print(fps_ori, int(label_dict[id_object])/fps_ori)
            if int(label_dict[id_object])/fps_ori >= 10:
                alert = True
                alert_cnt+=1
                #print('Warning!!!!!!! : {}'.format(id_object))
                
                if int(label_dict[id_object])/fps_ori > 20:
                    label_dict[id_object] = 1
            else:
                alert=False    
                alert_cnt = 0
            
            # 연속프레임으로 보이기
            if same:
                if same_cnt<=90:
                  same_cnt += 1
                else:
                  same = False
                  same_cnt = 0
                
            else:
                same_cnt=0
                same = False
                
                
            if alert_cnt==10:
                print('Warning!!!!!!! : {}'.format(id_object))
                
                img_crop = frame[int(bbox[1]) if int(bbox[1])>0 else 0:int(bbox[3]) if int(bbox[3])<len(frame) else len(frame), int(bbox[0]) if int(bbox[0])>0 else 0:int(bbox[2]) if int(bbox[2])<len(frame[0]) else len(frame[0])]
                print(len(img_crop), len(img_crop[0]))
                
                #plt.imshow(img_crop)
                #plt.show()
                obj_cnt=0
                total_cnt = 0
                for pre_img in track_li:
                    #print(pre_img)
                    #pre_img = cv2.resize(pre_img, (pre_img.shape[1], pre_img.shape[0]))
                    if pre_img == list():
                        continue
                    else:
                        total_cnt+=1
                        ###확인
                        sift = cv2.SIFT_create() 

                        # find the keypoints and descriptors with SIFT 
                    
                        kp1, des1 = sift.detectAndCompute(pre_img,None) 
                    
                        kp2, des2 = sift.detectAndCompute(img_crop,None) # FLANN parameters 
                        FLANN_INDEX_KDTREE = 0 
                        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 3) 
                        search_params = dict(checks=50) # or pass empty dictionary 
                        flann = cv2.FlannBasedMatcher(index_params,search_params) 
                        matches = flann.knnMatch(des1,des2,k=2) # Need to draw only good matches, so create a mask 
                    
                        matchesMask = [[0,0] for pp in range(len(matches))] # ratio test as per Lowe's paper 
                        for iii,(mmm,nnn) in enumerate(matches): 
                            if mmm.distance < 0.7*nnn.distance: 
                                matchesMask[iii]=[1,0]
                                draw_params = dict(matchColor = (0,255,0), 
                                                   singlePointColor = (255,0,0), 
                                                   matchesMask = matchesMask, 
                                                   flags = 0) 
                            else:
                                draw_params = dict(matchColor = (0,255,0), 
                                                   singlePointColor = (255,0,0), 
                                                   matchesMask = matchesMask, 
                                                   flags = 0) 

                        if draw_params['matchesMask'].count([1,0])>10:
                            obj_cnt+=1

                            
                if obj_cnt>int(total_cnt/2)-2:
                    print('*'*30)
                    print('Warning')
                    print('*'*30)
                    same = True

                    
                
               
                #img_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                if not img_crop==list():
                  track_li[track_num%10] = img_crop
                
                track_num+=1
                #print(track_li)

        
        
        ##############
        # 접근
        ##############
            area_frame = original_h * original_w
            area_object = (int(bbox[3])-int(bbox[1]))*(int(bbox[2])-int(bbox[0]))
            
            
            #print(area_frame, area_object, area_object/area_frame)
            if area_object/area_frame >0.7:
                close = True
            
            else:
                close = False

        
        ###########
        # 가려짐
        ###########
            print(np.mean(frame))
            
            hide_frame_tmp = [i for i in hide_frame_li if i>0]
            print(np.mean(hide_frame_tmp))
            

            if np.mean(hide_frame_tmp) - np.mean(frame)>15:
                hiding = True
                print('hide')

            else:
                hiding = False
                hide_frame_li[hide_num%10]=np.mean(frame)
                hide_num+=1
                
                
                
        # draw bbox on screen
            """
            if alert_cnt==10:
                img_crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                cv2.imwrite('./tracking/object_{}.bmp'.format(obj_num), img_crop)
                obj_num+=1
            """     
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)

            if same:
                cv2.putText(frame, class_name + "-" + str(track.track_id)+ ' - Same',(int(bbox[0]), int(bbox[1]-10)),0, 1, (255,255,0),3)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 4)
                
            elif alert:
                cv2.putText(frame, class_name + "-" + str(track.track_id)+ ' - Warning',(int(bbox[0]), int(bbox[1]-10)),0, 1, (255,0,0),3)
            
            elif close:
                cv2.putText(frame, class_name + "-" + str(track.track_id)+ ' - Close',(int(bbox[0]), int(bbox[1]+10)),0, 1, (255,0,0),3)
            
            else:
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                
            if hiding:
                cv2.putText(frame,'occluded',(int(len(frame[0])/2)-50,int(len(frame)/2)),0, 3, (255,0,255),3)
            
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
