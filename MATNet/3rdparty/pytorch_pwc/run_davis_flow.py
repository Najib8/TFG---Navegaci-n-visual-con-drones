from enum import Enum
from time import time

import torch
import glob
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from run import estimate
import flow_vis, cv2
import csv


class FlowMethod(Enum):
    PWC = 'pwc'
    Farneback = 'farneback'
    TVL1 = 'tlv1'
    Farneback_075 = 'farneback-0.75'
    TVL1_075 = 'tlv1-0.75'
    Farneback_05 = 'farneback-0.5'
    TVL1_05 = 'tlv1-0.5'


def main():

    original_shape_cv = None
    all_methods = [FlowMethod.PWC,
                   FlowMethod.Farneback, FlowMethod.TVL1,
                   FlowMethod.Farneback_075, FlowMethod.TVL1_075,
                   FlowMethod.Farneback_05, FlowMethod.TVL1_05]

    for method in all_methods:

        if method == FlowMethod.Farneback_05 or method == FlowMethod.TVL1_05:
            resize_factor = 0.5
        elif method == FlowMethod.Farneback_075 or method == FlowMethod.TVL1_075:
            resize_factor = 0.75
        else:
            resize_factor = None

        # For DAVIS DB examples
        raw_images_folder = '/home/najib/MATNet/data/DAVIS2017/JPEGImages/480p'
        save_dir = '/home/najib/MATNet/data/DAVIS2017/davis2017-flow-' + method.value

        # For tests1.0
        # raw_images_folder = '/home/najib/Escritorio/Ing_Inf_4/TFG/tests1.0/frames'
        # save_dir = '/home/najib/Escritorio/Ing_Inf_4/TFG/tests1.0/frames-flow'

        videos = os.listdir(raw_images_folder)

        time_videos = []

        for idx, video in enumerate(videos):
            print('process {}[{}/{}]'.format(video, idx, len(videos)))
            save_dir_video = os.path.join(save_dir, video)
            if not os.path.exists(save_dir_video):
                os.makedirs(save_dir_video)

            imagefiles = sorted(glob.glob(os.path.join(raw_images_folder, video, '*.jpg')))

            time_video = {'sequence': video, 'time': 0}

            for i in range(len(imagefiles)-1):

                f1 = imagefiles[i]
                f2 = imagefiles[i+1]

                save_name = os.path.basename(f1)[:-4] + '_' + os.path.basename(f2)[:-4] + '.png'
                save_file = os.path.join(save_dir_video, save_name)
                time_video['time'] += run(f1, f2, save_file, method, resize_factor, original_shape_cv)

            time_video['time'] /= len(imagefiles)-1
            time_videos.append(time_video)

        time_video = {'sequence': 'all sequences', 'time': sum([time['time'] for time in time_videos])}
        time_videos.append(time_video)

        with open(save_dir + '/times.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=time_videos[0].keys())
            writer.writeheader()
            writer.writerows(time_videos)


def run(imagefile1, imagefile2, save_file, method, resize_factor, original_shape_cv):

    if method == FlowMethod.PWC:
        tensor_first = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile1))[:, :, ::-1].transpose(2, 0, 1)
                                         .astype(numpy.float32) * (1.0 / 255.0))
        tensor_second = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile2))[:, :, ::-1].transpose(2, 0, 1)
                                          .astype(numpy.float32) * (1.0 / 255.0))

        time_start = time()
        tensor_output = estimate(tensor_first, tensor_second).numpy().transpose(1, 2, 0)
        time_spent = time() - time_start

    elif method == FlowMethod.Farneback:
        image1 = cv2.imread(imagefile1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(imagefile2, cv2.IMREAD_GRAYSCALE)

        time_start = time()
        tensor_output = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 5, 3, 5, 1.2, 0)
        time_spent = time() - time_start

    elif method == FlowMethod.Farneback_05 or method == FlowMethod.Farneback_075:
        image1 = cv2.imread(imagefile1, cv2.IMREAD_GRAYSCALE)
        if not original_shape_cv:
            original_shape_cv = (image1.shape[1], image1.shape[0])
            reduced_shape_cv = (int(original_shape_cv[0] * resize_factor), int(original_shape_cv[1] * resize_factor))
        image2 = cv2.imread(imagefile2, cv2.IMREAD_GRAYSCALE)

        image1 = cv2.resize(image1, reduced_shape_cv)
        image2 = cv2.resize(image2, reduced_shape_cv)

        time_start = time()
        reduced_tensor_output = cv2.calcOpticalFlowFarneback(image1, image2, None, 0.5, 3, 5, 3, 5, 1.2, 0)
        time_spent = time() - time_start

        tensor_output = numpy.zeros(shape=(original_shape_cv[1], original_shape_cv[0], 2))
        tensor_output[:, :, 0] = cv2.resize(reduced_tensor_output[:, :, 0], original_shape_cv)
        tensor_output[:, :, 1] = cv2.resize(reduced_tensor_output[:, :, 0], original_shape_cv)

    elif method == FlowMethod.TVL1:
        image1 = cv2.imread(imagefile1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(imagefile2, cv2.IMREAD_GRAYSCALE)

        time_start = time()
        dual_tvl1_optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        tensor_output = dual_tvl1_optical_flow.calc(image1, image2, None)
        time_spent = time() - time_start

    elif method == FlowMethod.TVL1_05 or method == FlowMethod.TVL1_075:
        image1 = cv2.imread(imagefile1, cv2.IMREAD_GRAYSCALE)
        if not original_shape_cv:
            original_shape_cv = (image1.shape[1], image1.shape[0])
            reduced_shape_cv = (int(original_shape_cv[0]*resize_factor), int(original_shape_cv[1]*resize_factor))
        image2 = cv2.imread(imagefile2, cv2.IMREAD_GRAYSCALE)

        image1 = cv2.resize(image1, reduced_shape_cv)
        image2 = cv2.resize(image2, reduced_shape_cv)

        time_start = time()
        dual_tvl1_optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        reduced_tensor_output = dual_tvl1_optical_flow.calc(image1, image2, None)
        time_spent = time() - time_start

        tensor_output = numpy.zeros(shape=(original_shape_cv[1], original_shape_cv[0], 2))
        tensor_output[:, :, 0] = cv2.resize(reduced_tensor_output[:, :, 0], original_shape_cv)
        tensor_output[:, :, 1] = cv2.resize(reduced_tensor_output[:, :, 0], original_shape_cv)
    else:
        time_spent = 0

    flow_color = flow_vis.flow_to_color(tensor_output, convert_to_bgr=True)
    cv2.imwrite(save_file, flow_color)

    return time_spent

    # objectOutput = open(save_file, 'wb')

    # numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
    # numpy.array([ tensor_output.size(2), tensor_output.size(1) ], numpy.int32).tofile(objectOutput)
    # numpy.array(tensor_output.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

	#objectOutput.close()


if __name__ == '__main__':
    main()
