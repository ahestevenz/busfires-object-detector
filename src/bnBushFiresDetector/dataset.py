from typing import Tuple, Dict, List
from pathlib import Path
import os
import subprocess
from loguru import logger as logging

import numpy as np
import pandas as pd
import tensorflow as tf

class DataManagement:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=False, exist_ok=True)
        if not (Path(self.data_dir)/Path('Smoke')).exists():
            self.download_smoke_dataset()
        self.ds = self.load_smoke_dataset()
        self.batch_size = batch_size

    def download_smoke_dataset(self):
        cmd = f'curl -L "https://public.roboflow.com/ds/3bLgtHIm8e?key=CFnFVougtG" > {self.data_dir}/roboflow.zip; unzip -o {self.data_dir}/roboflow.zip -d {self.data_dir}/Smoke; rm {self.data_dir}/roboflow.zip'
        subprocess.run(cmd, shell=True, universal_newlines=True, check=True)

    def load_smoke_dataset(self)->Dict:
        dataset_paths = [('train','/Smoke/train/'),('valid', '/Smoke/valid/'), ('test', '/Smoke/test/')]
        ds = {}
        for dataset, dataset_path in dataset_paths:
            df = pd.read_csv(f'{self.data_dir}/{dataset_path}/_annotations.csv')
            for i,row in df.iterrows():
                image_path = dataset_path + row["filename"]
                if Path(f'{self.data_dir}/{image_path}').is_file():
                    df["filename"].to_numpy()[i] = image_path
                file_paths = df["filename"].values
                bboxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values
                ds[dataset] = tf.data.Dataset.from_tensor_slices((file_paths, bboxes))
        return ds

    def read_image(self, 
                   image_file:str)->tf.Tensor:
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image) 
        return image

    def normalise_image(self, image:tf.Tensor)->Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        shape = tf.shape(image)
        factor_x = tf.cast(shape[1], tf.float32)
        factor_y = tf.cast(shape[0], tf.float32)
        image = tf.image.resize(image, (224, 224,))
        image = image/127.5
        image -= 1  
        return image, shape

    def normalise_image_and_bboxes(self, 
                                   image:tf.Tensor, 
                                   bbox:tf.Tensor)->Tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        image, shape = self.normalise_image(image)
        factor_x = tf.cast(shape[1], tf.float32)
        factor_y = tf.cast(shape[0], tf.float32)

        bbox = tf.cast(bbox,  tf.float32)
        bbox_list = [bbox[0] / factor_x , 
                    bbox[1] / factor_y, 
                    bbox[2] / factor_x , 
                    bbox[3] / factor_y]
        return image, bbox_list 


    def normalise_image_and_bboxes_from_path(self,
                                             image_file:str, 
                                             bbox:tf.Tensor)->Tuple[tf.Tensor, tf.Tensor]:
        image = self.read_image(self.data_dir + image_file)
        image, bbox_list = self.normalise_image_and_bboxes(image, bbox)
        return image, bbox_list

    def normalise_n_flip_image_and_bboxes_from_path(self,
                                                    image_file:str, 
                                                    bbox:tf.Tensor)->Tuple[tf.Tensor,tf.Tensor]:
        image = self.read_image(self.data_dir + image_file)
        image, shape = self.normalise_image(image)
        factor_x = tf.cast(shape[1], tf.float32)
        factor_y = tf.cast(shape[0], tf.float32)

        image =  image[:,::-1,:]
        img_center_x = factor_x/2.0
        bbox = tf.cast(bbox,  tf.float32)
        box_w = abs((bbox[0] - bbox[2])/factor_x)
        bbox_list = [(bbox[0] + 2*(img_center_x - bbox[0]) - box_w)/factor_x, 
                      bbox[1] / factor_y, 
                     (bbox[2] + 2*(img_center_x - bbox[2]) + box_w)/factor_x, 
                      bbox[3] / factor_y]
        return image, bbox_list

    def original_normalise_image_and_bboxes(self,
                                            image:tf.Tensor, 
                                            bbox:tf.Tensor)->Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        original_image = image
        image, bbox_list = self.normalise_image_and_bboxes(image, bbox)
        return original_image, image, bbox_list

    def original_image_and_bboxes_from_path(self,
                                            image_file:str, 
                                            bbox:tf.Tensor)->Tuple[tf.Tensor, tf.Tensor]:
        image = self.read_image(self.data_dir + image_file)
        bbox_list = [bbox[0], 
                     bbox[1], 
                     bbox[2], 
                     bbox[3]]        
        return image, bbox_list

    def dataset_to_numpy_arrays(self,
                                dataset:tf.data.Dataset, 
                                batch_size:int = 0, 
                                N:int = 0)->Tuple[np.array,np.array]:
        take_dataset = dataset.shuffle(1024)
        if batch_size > 0:
            take_dataset = take_dataset.batch(batch_size)
        if N > 0:
            take_dataset = take_dataset.take(N)
        if tf.executing_eagerly():
            ds_images, ds_bboxes = [], []
            for images, bboxes in take_dataset:
                ds_images.append(images.numpy())
                ds_bboxes.append(bboxes.numpy())
        return (np.array(ds_images), np.array(ds_bboxes))
        
    def dataset_to_numpy_arrays_with_original_bboxes(self,
                                                     dataset:tf.data.Dataset, 
                                                     batch_size:int = 0, 
                                                     N:int = 0)->Tuple[np.array,np.array,np.array]:

        normalized_dataset = dataset.map(self.original_normalise_image_and_bboxes)
        if batch_size > 0:
            normalized_dataset = normalized_dataset.batch(batch_size)
        if N > 0:
            normalized_dataset = normalized_dataset.take(N)
        if tf.executing_eagerly():
            ds_original_images, ds_images, ds_bboxes = [], [], []    
        for original_images, images, bboxes in normalized_dataset:
            ds_images.append(images.numpy())
            ds_bboxes.append(bboxes.numpy())
            ds_original_images.append(original_images.numpy())
        return np.array(ds_original_images), np.array(ds_images), np.array(ds_bboxes)

    def get_dataset(self,
                    set:str,
                    aug_enabled:bool = False) -> Tuple[tf.data.Dataset, int]:
        dataset = self.ds[set].map(self.normalise_image_and_bboxes_from_path, num_parallel_calls=16)
        if aug_enabled == True and set == 'train':
            flipped_dataset = self.ds[set].map(self.normalise_n_flip_image_and_bboxes_from_path, num_parallel_calls=16)
            dataset = dataset.concatenate(flipped_dataset)
        len_dataset = len(dataset)
        if set == 'train':
            dataset = dataset.shuffle(512, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        if set == 'train':
            dataset = dataset.prefetch(-1) 
        return dataset, len_dataset, self.ds[set]