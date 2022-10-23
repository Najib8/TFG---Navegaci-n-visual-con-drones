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


def main():
    # Capture WebCam Video
    capture = cv2.VideoCapture(0)

    # Get the Previous Frame
    ret, frame1 = capture.read()
    prev_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    time_frames = []

    while True:

        # Get the Next Frame
        ret, frame2 = capture.read()
        if not ret:
            print('No frames grabbed!')
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        time_start = time()

        # Create directory and checks for the correct number of streaming sequence to use
        save_name = 'streaming_[:-4] + '_' + os.path.basename(f2)[:-4] + '.png'
        save_file = os.path.join(save_dir_video, save_name)
        run(f1, f2, save_file)

        time_end = time()

        time_videos[video].append(time_end - time_start)
        print('Time seqs:', time_videos[video])

    print(time_videos)


def run(imagefile1, imagefile2, save_file):
	tensorFirst = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile1))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
	tensorSecond = torch.FloatTensor(numpy.array(PIL.Image.open(imagefile2))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))

	tensorOutput = estimate(tensorFirst, tensorSecond)

	flow_color = flow_vis.flow_to_color(tensorOutput.numpy().transpose(1,2,0), convert_to_bgr=True)
	cv2.imwrite(save_file, flow_color)

	#objectOutput = open(save_file, 'wb')

	#numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
	#numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
	#numpy.array(tensorOutput.numpy().transpose(1, 2, 0), numpy.float32).tofile(objectOutput)

	#objectOutput.close()


if __name__ == '__main__':
    main()
