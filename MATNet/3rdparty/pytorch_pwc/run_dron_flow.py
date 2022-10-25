from enum import Enum
import time

import torch
import numpy
import PIL
import PIL.Image
from run import estimate
import flow_vis
import cv2
from djitellopy import Tello


class FlowMethod(Enum):
    PWC = 'pwc'
    Farneback = 'farneback'
    TVL1 = 'tlv1'
    Farneback_075 = 'farneback-0.75'
    TVL1_075 = 'tlv1-0.75'
    Farneback_05 = 'farneback-0.5'
    TVL1_05 = 'tlv1-0.5'


def main():
    # Create the Tello Object
    tello = Tello()

    # Initialize connection and streaming
    tello.connect()
    tello.streamon()

    # Initialize the frame reader
    frame_read = tello.get_frame_read()

    # Obtain the first frame
    f1 = frame_read.frame
    cv2.imwrite("frame1.png", f1)

    # Wait for three tenths of a second
    time.sleep(0.3)

    # Obtain the second frame
    f2 = frame_read.frame
    cv2.imwrite("frame2.png", f2)

    # Finish the streaming and disconnect
    tello.streamoff()
    tello.end()

    # Estimate the optical flow
    flow = run(f1, f2)
    cv2.imwrite("flow.png", flow)


def run(image1, image2):

    tensor_first = torch.FloatTensor(numpy.array(PIL.Image.fromarray(image1))[:, :, ::-1].transpose(2, 0, 1)
                                     .astype(numpy.float32) * (1.0 / 255.0))
    tensor_second = torch.FloatTensor(numpy.array(PIL.Image.fromarray(image2))[:, :, ::-1].transpose(2, 0, 1)
                                      .astype(numpy.float32) * (1.0 / 255.0))

    tensor_output = estimate(tensor_first, tensor_second).numpy().transpose(1, 2, 0)

    flow_color = flow_vis.flow_to_color(tensor_output, convert_to_bgr=True)

    return flow_color


if __name__ == '__main__':
    main()
