import csv
from enum import Enum
from time import time

import torch
from torchvision import transforms

import os
import glob
from tqdm import tqdm
from PIL import Image
from scipy.misc import imresize

from modules.MATNet import Encoder, Decoder
from utils.utils import check_parallel
from utils.utils import load_checkpoint_epoch


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

model_name = 'MATNet' # specify the model name
epoch = 0 # specify the epoch number
test_result_dir = 'MATNet/output/DAVIS16'

flow_methods = [FlowMethod.PWC,
                FlowMethod.Farneback, FlowMethod.TVL1,
                FlowMethod.Farneback_075, FlowMethod.TVL1_075,
                FlowMethod.Farneback_05, FlowMethod.TVL1_05]

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

val_set = 'MATNet/data/DAVIS2017/ImageSets/2016/val.txt'
with open(val_set) as f:
    seqs = f.readlines()
    seqs = [seq.strip() for seq in seqs]

for flow_method in flow_methods[:1]:

    test_dir = 'MATNet/data/DAVIS2017/JPEGImages/480p'
    test_flow_dir = 'MATNet/data/DAVIS2017/davis2017-flow-' + flow_method.value

    save_folder = '{}/{}_epoch{}_{}'.format(test_result_dir,
                                            model_name, epoch, flow_method.value)

    time_videos = []

    for video in tqdm(seqs):

        image_dir = os.path.join(test_dir, video)
        flow_dir = os.path.join(test_flow_dir, video)

        imagefiles = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:-1]
        flowfiles = sorted(glob.glob(os.path.join(flow_dir, '*.png')))

        time_video = {'sequence': video, 'time': 0}

        with torch.no_grad():
            for imagefile, flowfile in zip(imagefiles, flowfiles):

                time_start = time()

                image = Image.open(imagefile).convert('RGB')
                flow = Image.open(flowfile).convert('RGB')

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

                save_folder_video = save_folder + '/' + video
                if not os.path.exists(save_folder_video):
                    os.makedirs(save_folder_video)

                save_file = os.path.join(save_folder_video,
                                         os.path.basename(imagefile)[:-4] + '.png')
                mask_pred = mask_pred.resize((width, height))

                time_video['time'] += time() - time_start

                mask_pred.save(save_file)

            time_video['time'] /= len(imagefiles)
            time_videos.append(time_video)

    time_video = {'sequence': 'all sequences', 'time': sum([time['time'] for time in time_videos])}
    time_videos.append(time_video)

    with open(save_folder + '/times.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=time_videos[0].keys())
        writer.writeheader()
        writer.writerows(time_videos)
