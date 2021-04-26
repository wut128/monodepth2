# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
# from layers import disp_to_depth
from layers import *
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--load_model", type=str,
                        help='pass to load a pretrained weights')
    parser.add_argument("--simple_mode",
                        help='if set, choose the simple transformation from pre-image',
                        action='store_true')
    parser.add_argument("--integrate",
                        help='if set, integrate the pred_depth',
                        action='store_true')
    parser.add_argument("--filter_ratio",
                        type=float,
                        help="the ratio to keep the predicted depth",
                        default=0.7)


    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    #assert args.model_name is not None, \
        #"You must specify the --model_name parameter; see README.md for an example"
    K = np.array([[1.05, 0, 0.51, 0],
                  [0, 1.60, 0.57, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=np.float32)

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #download_model_if_doesnt_exist(args.model_name)
    #model_path = os.path.join("models", args.model_name)
    model_path = os.path.expanduser(args.load_model)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
    pose_decoder_path = os.path.join(model_path, "pose.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)


    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height'] # 192
    feed_width = loaded_dict_enc['width']  # 640
    K[0, :] *= feed_width
    K[1, :] *= feed_height
    inv_K = np.linalg.pinv(K)
    K = torch.from_numpy(K)
    inv_K = torch.from_numpy(inv_K)
    K = K.unsqueeze(0)
    inv_K = inv_K.unsqueeze(0)
    # K = K.to(device)
    # inv_K = inv_K.to(device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    # encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.HRDepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    # depth_decoder.to(device)
    depth_decoder.eval()

    print("   Loading pretrained pose-encoder")
    pose_encoder = networks.ResnetEncoder(18, False,2)
    loaded_dict_pose_enc = torch.load(pose_encoder_path, map_location=device)
    filtered_dict_pose_enc = {k: v for k, v in loaded_dict_pose_enc.items() if k in pose_encoder.state_dict()}
    pose_encoder.load_state_dict(filtered_dict_pose_enc)
    # pose_encoder.to(device)
    pose_encoder.eval()

    print("   Loading pretrained pose-decoder")
    pose_decoder = networks.PoseDecoder(np.array([64, 64, 128, 256, 512]),
                    num_input_features = 1,
                    num_frames_to_predict_for = 1)

    loaded_dict_pose = torch.load(pose_decoder_path, map_location=device)
    filtered_dict_pose = {k: v for k, v in loaded_dict_pose.items() if k in pose_decoder.state_dict()}
    pose_decoder.load_state_dict(filtered_dict_pose)

    # pose_decoder.to(device)
    pose_decoder.eval()


    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        output_directory = os.path.join(args.image_path,'test_result_integ_adap5')
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))

    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))
    number_num=[]
    paths_new = []
    # order the image number in paths
    for idx, image_p in enumerate(paths):
        base_name = os.path.basename(image_p)
        base_name = os.path.splitext(base_name)
        number_num.append(int(base_name[0]))

    # sort_num = np.array(number_num)
    number_num.sort()
    # ind = np.argsort(np.argsort(sort_num))
    for index, name in enumerate(number_num):
        paths_new.append(os.path.join(args.image_path, '{:010d}.{}'.format(name,args.ext)))
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths_new):
            outputs = []

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            folder_name = os.path.dirname(image_path)
            source_path = os.path.join(folder_name,"{:010d}.jpg".format(int(output_name)-2))
            if not os.path.isfile(source_path):
                continue
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            source_image = pil.open(source_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            source_image = source_image.resize((feed_width, feed_height), pil.LANCZOS)
            source_image = transforms.ToTensor()(source_image).unsqueeze(0)
            pose_input = [source_image, input_image]
            for input_pic in pose_input:
            # PREDICTION
                # input_pic = input_pic.to(device)
                features = encoder(input_pic)
                outputs.append(depth_decoder(features))

            pose_input = torch.cat(pose_input, 1)
            # pose_input = pose_input.to(device)
            pose_feature = [pose_encoder(pose_input)]
            axisangle, translation = pose_decoder(pose_feature)
            T = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=True)

            # record the output of depth decoder
            source_output = outputs[0]
            target_output = outputs[1]
            disp=[]
            disp.append(source_output[("disp", 0)])
            disp.append(target_output[("disp", 0)])

            # calculate the transform matrix
            npy_path = os.path.join(output_directory, "{:010d}_disp.npy".format(int(output_name)-2))
            if os.path.exists(npy_path):
                loadData = np.load(npy_path)
                disp_source = torch.from_numpy(loadData)
            else:
                disp_source = disp[0]
            # print("the dimension of disp[0] is",disp_source.shape)
            scaled_disp_source, depth_source = disp_to_depth(disp_source, 0.1, 100)

            back_project = BackprojectDepth(1, feed_height, feed_width)
            cam_points = back_project(depth_source, inv_K)
            project_3d = Project3D(1, feed_height, feed_width)
            pix_coords = project_3d(cam_points, K, T)

            # calculate the depth from predicted image
            # pred_image = pred_image.to(device)
            if not args.simple_mode:
                pred_image = F.grid_sample(
                    source_image, pix_coords,
                    padding_mode="border", align_corners=True)
                features = encoder(pred_image)
                output_pred = depth_decoder(features)
                disp_pred = output_pred[("disp", 0)]
            else:
                disp_pred = F.grid_sample(
                    disp_source, pix_coords,
                    padding_mode="border", align_corners=True)
            # scaled_disp_pred, depth_pred = disp_to_depth(disp_pred , 0.1, 100)
            disp_pred_resized = torch.nn.functional.interpolate(
                    disp_pred, (original_height, original_width), mode="bilinear", align_corners=False)

            # save and draw depth from predicted image
            disp_resized_np_source = disp_pred_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np_source, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np_source.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np_source)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
            if len(paths) > 1:
                name_dest_im = os.path.join(output_directory, "{}_disp_pred_batch.jpeg".format(output_name))
            elif not args.simple_mode:
                name_dest_im = os.path.join(output_directory, "{}_disp5_pred.jpeg".format(output_name))
            else:
                name_dest_im = os.path.join(output_directory, "{}_disp5_pred_simple.jpeg".format(output_name))
            # save the predicted depth image
            # im.save(name_dest_im)

            # Saving numpy file
            # output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            # scaled_disp, depth = disp_to_depth(disp[1], 0.1, 100)
            # The output of depth decoder is saved
            if args.integrate:
                # filter_ratio is used rather than adaptive filtering
                # disp_f = torch.where(disp[1] > disp_pred,disp[1],
                #                     disp[1]*(1-args.filter_ratio)+disp_pred*args.filter_ratio)
                # adaptive filtering

                # disp_f = torch.where(disp[1] > disp_pred,disp[1],
                #                     disp[1]*torch.clamp(disp_pred+0.4,0,1)+disp_pred*torch.clamp(1-disp_pred-0.4,0,1))
                # disp_f = torch.where(disp[1] > disp_pred, disp[1],
                #                      disp[1] * torch.clamp(torch.tanh(disp_pred-0.4)*0.3 + disp_pred + 0.4, 0, 1) + disp_pred * torch.clamp(
                #                          1 - torch.tanh(disp_pred-0.4)*0.3 - disp_pred - 0.4, 0, 1))
                # when disp_pred=0, ratio for disp_pred is 0.75, when disp_pred is 0.55 ratio is 0
                disp_f = torch.where(disp[1] > disp_pred, disp[1],
                                     disp[1] * torch.clamp(torch.tanh(disp_pred-0.4)*0.4 + disp_pred + 0.4, 0, 1) + disp_pred * torch.clamp(
                                         1 - torch.tanh(disp_pred-0.4)*0.4 - disp_pred - 0.4, 0, 1))
            else:
                disp_f = disp[1]
            np.save(name_dest_npy, disp_f.cpu().numpy())

            # process the target image of final result
            disp_resized = torch.nn.functional.interpolate(
                disp_f, (original_height, original_width), mode="bilinear", align_corners=False)
            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            if len(paths) > 1:
                name_dest_im = os.path.join(output_directory, "{}_disp_batch.jpeg".format(output_name))
            else:
                name_dest_im = os.path.join(output_directory, "{}_disp5.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')



if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
