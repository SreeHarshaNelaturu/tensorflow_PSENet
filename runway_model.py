# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from utils.utils_tool import logger, cfg
import runway
from runway.data_types import *
from nets import model
from pse import pse

def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    #ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w


    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        kernal = np.where(seg_maps[..., i]>thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time()-start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        #(y,x)
        points = np.argwhere(mask_res_resized==label_value)
        points = points[:, (1,0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes), kernals, timer


g = None
@runway.setup(options={"checkpoint" : file(is_directory=True)})
def setup(opts):
    global g

    sess = tf.Session()
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    seg_maps_pred = model.model(input_images, is_training=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    model_path = opts["checkpoint"] + "/" + "model.ckpt" 
    saver.restore(sess, model_path)    
        
    g = tf.get_default_graph()

    return { "in_ph" : input_images,
             "seg_maps_pred" : seg_maps_pred,
             "sess" : sess
            }

command_inputs = {"input_image" : image}
command_outputs = {"bboxes" : array(image_bounding_box)}


@runway.command("localize_text", inputs=command_inputs, outputs=command_outputs, description="Localize Text in Image")
def localize_text(model, inputs):    

        timer = {'net': 0, 'pse': 0}
        #im = cv2.imread("./o.jpg")[:, :, ::-1]
        im = np.array(inputs["input_image"])[:, :, ::-1]
        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = resize_image(im)
        
        h, w, _ = im_resized.shape
        start = time.time()
        with g.as_default():
            seg_maps = model["sess"].run(model["seg_maps_pred"], feed_dict={model["in_ph"]: [im_resized]})
        timer['net'] = time.time() - start
        #print

        boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)
        
        if boxes is not None:
            boxes = boxes.reshape((-1, 4, 2))
            boxes[:, :, 0] /= w
            boxes[:, :, 1] /= h
            #print(ratio_w)
            #print(ratio_h)
            h, w, _ = im.shape
            boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
            boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)

            #print(boxes.shape, boxes)

            num =0
            for i in range(len(boxes)):
                # to avoid submitting errors
                box = boxes[i]
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
               
                num += 1
        final_list = []

        for j in range(len(boxes)):
                box = boxes[j]
                top = box[3]# print(len(out_bboxes))
                bottom = box[0]
                x1 = min(box[:, 0])
                x2 = max(box[:, 0])
                y1 = min(box[:, 1])
                y2 = max(box[:, 1])
                bbox = [x1, y1, x2, y2]
                final_list.append(bbox)
            
        print(final_list) 
        return {"bboxes" : final_list}

if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "./model"})

