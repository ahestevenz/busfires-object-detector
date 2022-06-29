from typing import Tuple, Dict, List
from loguru import logger as logging
from pathlib import Path

import os
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

plt.rc('image', cmap='gray')
plt.rc('grid', linewidth=0)
plt.rc('xtick', top=False, bottom=False, labelsize='large')
plt.rc('ytick', left=False, right=False, labelsize='large')
plt.rc('axes', facecolor='F8F8F8', titlesize="large", edgecolor='white')
plt.rc('text', color='a8151a')
plt.rc('figure', facecolor='F0F0F0')
MATPLOTLIB_FONT_DIR = os.path.join(os.path.dirname(plt.__file__), "mpl-data/fonts/ttf")

def plot_metrics(history:Dict, 
                 metric_name:str, 
                 title:str, 
                 ylim:int = 5) -> plt.plot:
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history[f'val_{metric_name}'],color='green',label=f'val_{metric_name}')

def draw_bounding_box_on_image(image:tf.Tensor, 
                               ymin:int, 
                               xmin:int, 
                               ymax:int, 
                               xmax:int, 
                               color:tuple = (0, 0, 255), 
                               thickness:int = 5)->cv2.rectangle:
    image_width = image.shape[1]
    image_height = image.shape[0]
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)


def draw_bounding_boxes_on_image(image:tf.Tensor, 
                                 boxes:list, 
                                 color:list = [], 
                                 thickness:int = 5)->cv2.rectangle:
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 1], boxes[i, 0], boxes[i, 3],
                                 boxes[i, 2], color[i], thickness)


def draw_bounding_boxes_on_image_array(image, boxes, color=[], thickness=5):
    draw_bounding_boxes_on_image(image, boxes, color, thickness)
    return image

def display_digits_with_boxes(images:List, 
                              pred_bboxes:List, 
                              bboxes:List,
                              iou:list,
                              title:str,
                              dirname:str = None, 
                              bboxes_normalized:bool = False,
                              iou_threshold:float = 0.5):

    n = len(images)
    colors = {}
    colors["yellow"] = (255,255,0)
    colors["blue"] = (0,0,255)
    legend = f'{list(colors)[0]}: predicted, {list(colors)[1]}: annotated' if (len(pred_bboxes) > 0) else  f'{list(colors)[0]}: annotated'
    fig = plt.figure(figsize=(30, 8))
    plt.title(f'{title} | {legend}')
    plt.yticks([])
    plt.xticks([])
  
    for i in range(n):
      ax = fig.add_subplot(1, 10, i+1)
      bboxes_to_plot = []
      if (len(pred_bboxes) > i):
        logging.info(f'Predicted box: {i}')
        bbox = pred_bboxes[i]
        bbox = [bbox[0] * images[i].shape[1], bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1], bbox[3] * images[i].shape[0]]
        logging.info(f'## pbbox{i} : {bbox}')
        bboxes_to_plot.append(bbox)
      if (len(bboxes) > i):
        logging.info(f'Annotated box: {i}')
        bbox = bboxes[i]
        if bboxes_normalized == True:
          bbox = [bbox[0] * images[i].shape[1],bbox[1] * images[i].shape[0], bbox[2] * images[i].shape[1], bbox[3] * images[i].shape[0] ]
        logging.info(f'## abbox{i} : {bbox}')
        bboxes_to_plot.append(bbox)

      img_to_draw = draw_bounding_boxes_on_image_array(image=images[i], boxes=np.asarray(bboxes_to_plot), color=[colors["yellow"], colors["blue"]])
      plt.xticks([])
      plt.yticks([])
    
      plt.imshow(img_to_draw)
      if len(iou) > i :
        color = "green"
        if (iou[i][0] < iou_threshold):
          color = "red"
        ax.text(0.2, -0.3, f"iou: {iou[i][0]}", color=color, transform=ax.transAxes)
      name = "result" if len(pred_bboxes) > 0 else "valid_data"
    if dirname is not None:
      dirname = Path(dirname)
      dirname.mkdir(parents=False, exist_ok=True)
      plt.savefig(f'{dirname}/{name}_with_boxes.png')
