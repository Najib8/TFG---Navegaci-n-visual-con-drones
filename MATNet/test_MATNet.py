from enum import Enum

import cv2
import torch
import numpy as np
from torchvision import transforms

import os
from PIL import Image, ImageDraw
from scipy.misc import imresize

from modules.MATNet import Encoder, Decoder
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch

import time
from djitellopy import Tello
from MATNet.thirdparty.pytorch_pwc.run_dron_flow import run


class FlowMethod(Enum):
    PWC = 'pwc'
    Farneback = 'farneback'
    TVL1 = 'tlv1'
    Farneback_075 = 'farneback-0.75'
    TVL1_075 = 'tlv1-0.75'
    Farneback_05 = 'farneback-0.5'
    TVL1_05 = 'tlv1-0.5'


def flip(x, dim):
    if x.is_cuda:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long().cuda(0))
    else:
        return torch.index_select(x, dim, torch.arange(x.size(dim) - 1, -1, -1).long())


inputRes = (473, 473)
use_flip = True

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
image_transforms = transforms.Compose([to_tensor, normalize])

model_name = 'MATNet'  # specify the model name
epoch = 0  # specify the epoch number
test_result_dir = 'MATNet/output/DAVIS16'


encoder_dict, decoder_dict, enc_opt_dict, dec_opt_dict, load_args =\
    load_checkpoint_epoch(model_name, epoch, True, False)
encoder = Encoder()
decoder = Decoder()
encoder_dict, decoder_dict = check_parallel(encoder_dict, decoder_dict)
encoder.load_state_dict(encoder_dict)
decoder.load_state_dict(decoder_dict)

encoder.cuda()
decoder.cuda()

encoder.train(False)
decoder.train(False)

# -- Prepare directories to save results --

# Frames directory
save_folder_frames = os.getcwd() + '/results/frames'
if not os.path.exists(save_folder_frames):
    os.makedirs(save_folder_frames)

# Optical flows
save_folder_flows = os.getcwd() + '/results/optical_flows'
if not os.path.exists(save_folder_flows):
    os.makedirs(save_folder_flows)

# Segmentations
save_folder_video = os.getcwd() + '/results/segmentations'
if not os.path.exists(save_folder_video):
    os.makedirs(save_folder_video)

# Binarizations
save_folder_binarizations = os.getcwd() + '/results/binarizations'
if not os.path.exists(save_folder_binarizations):
    os.makedirs(save_folder_binarizations)

# Bounding boxes
save_folder_boxes = os.getcwd() + '/results/bounding-boxes'
if not os.path.exists(save_folder_boxes):
    os.makedirs(save_folder_boxes)

# -- Prepare dron flow --

# Create the Tello Object
tello = Tello()

# Initialize connection and streaming
tello.connect()
tello.streamon()

# Initialize the frame reader
frame_read = tello.get_frame_read()

# Declare the number of frames taken before the connection is closed
number_frames_taken = 1000

# Declare threshold parameter
threshold = 191

# -- Start --

# Obtain the first frame
f1 = frame_read.frame
cv2.imwrite(os.path.join(save_folder_frames, 'frame1.png'), f1)
print('Frame 1 obtained \n')

# Wait for three tenths of a second
time.sleep(0.3)

for iteration in np.arange(number_frames_taken)+1:
    print('Iteration', iteration, '\n')

    # Obtain the second frame
    f2 = frame_read.frame
    cv2.imwrite(os.path.join(save_folder_frames, 'frame' + str(iteration+1) + '.png'), f2)
    print('Frame 2 obtained \n')

    # Estimate the optical flow
    flow = run(f1, f2)
    cv2.imwrite(os.path.join(save_folder_flows, 'flow' + str(iteration) + '.png'), flow)
    print('Optical flow', iteration, 'obtained \n')

    with torch.no_grad():

        # Put the first frame and the optical flow estimated in MATNet to segment the IMO
        image = Image.fromarray(f1).convert('RGB')
        flow = Image.fromarray(flow).convert('RGB')

        width, height = image.size

        image = imresize(image, inputRes)
        flow = imresize(flow, inputRes)

        image = image_transforms(image)
        flow = image_transforms(flow)

        image = image.unsqueeze(0)
        flow = flow.unsqueeze(0)

        image, flow = image.cuda(), flow.cuda()

        r5, r4, r3, r2 = encoder(image, flow)
        mask_pred, bdry_pred, p2, p3, p4, p5 = decoder(r5, r4, r3, r2)

        if use_flip:
            image_flip = flip(image, 3)
            flow_flip = flip(flow, 3)
            r5, r4, r3, r2 = encoder(image_flip, flow_flip)
            mask_pred_flip, bdry_pred_flip, p2, p3, p4, p5 =\
                decoder(r5, r4, r3, r2)

            mask_pred_flip = flip(mask_pred_flip, 3)
            bdry_pred_flip = flip(bdry_pred_flip, 3)

            mask_pred = (mask_pred + mask_pred_flip) / 2.0
            bdry_pred = (bdry_pred + bdry_pred_flip) / 2.0

        mask_pred = mask_pred[0, 0, :, :]
        mask_pred = Image.fromarray(mask_pred.cpu().detach().numpy() * 255).convert('L')

    save_file = os.path.join(save_folder_video, 'segmentation' + str(iteration) + '.png')
    mask_pred = mask_pred.resize((width, height))

    # Threshold the image to binarize the IMO

    save_file_thresh = os.path.join(save_folder_binarizations, 'segmentation_thresh' + str(iteration) + '.png')
    _, mask_thresh = cv2.threshold(np.array(mask_pred), threshold, 255, cv2.THRESH_BINARY)
    cv2.imwrite(save_file_thresh, mask_thresh)

    # Get binarized IMO limits at right, top, left and bottom

    mask_bool = np.array(mask_thresh, dtype=bool)  # from color to bool

    # Collapse matrix to both axes
    mask_bool_x = np.logical_or.reduce(mask_bool)
    mask_bool_y = np.logical_or.reduce(mask_bool.transpose())

    # Look for the IMO pixels
    mask_idx_x = np.where(mask_bool_x == True)
    mask_idx_y = np.where(mask_bool_y == True)

    # Get the limits
    if len(mask_idx_x[0]):
        left_side = mask_idx_x[0][0]
        right_side = mask_idx_x[0][-1]
        top_side = mask_idx_y[0][0]
        bottom_side = mask_idx_y[0][-1]

        # Paint the corresponding rectangle (over the image) and save it

        image_box = Image.fromarray(f1).convert('RGB')
        bounding_box = ImageDraw.ImageDraw(image_box)
        bounding_box.rectangle(((left_side, top_side), (right_side, bottom_side)), fill=None, outline='green', width=4)

        save_file_box = os.path.join(save_folder_boxes, 'segmentation_box' + str(iteration) + '.png')
        image_box.save(save_file_box)

    mask_pred.save(save_file)
    print('Segmentation', iteration, 'saved \n')

    # Update frame1 with the previous frame2
    f1 = f2
    print('Frame 1 obtained \n')

# Finalize streaming and connection
tello.streamoff()
tello.end()
