from layers import *
import torch
from datasets.mono_dataset import pil_loader
from torchvision import transforms
from PIL import Image

"""Generate the warped (reprojected) color images for a minibatch.
Generated images are saved into the `outputs` dictionary.
"""
height = 192
width = 640
K = np.array([[1.05, 0, 0.51, 0],
            [0, 1.60, 0.57, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
K[0, :] *= width
K[1, :] *= height
inputs={}
inv_K = np.linalg.pinv(K)

inputs["K"] = torch.from_numpy(K)
inputs["inv_K"] = torch.from_numpy(inv_K)
outputs = {}
pic_pre = pil_loader()
pic_target = pil_loader()
pose_feats = {}
back_project = BackprojectDepth(1, height, width)
project_3d = Project3D(1, height, width)
resize = transforms.Resize((height, width),interpolation=Image.ANTIALIAS)

source_pic = resize(pic_pre)
target_pic = resize(pic_target)
# here is placed after test_simple3.py
disp = outputs[("disp", 0)]
disp = F.interpolate(
            disp, [height, width], mode="bilinear", align_corners=False)

_, depth = disp_to_depth(disp, 0.1, 100)

outputs[("depth", 0, 0)] = depth

axisangle, translation = self.models["pose"](pose_inputs)
outputs["axisangle"] = axisangle
outputs["translation"] = translation
T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)



# transform to 3D using depth info
cam_points = back_project(depth, inputs["inv_K"])
#  transform 3d to 2D using depth and T
pix_coords = project_3d[source_scale](
cam_points, inputs["K"], T)

outputs["sample"] = pix_coords
# outputs["sample"] is the transformed coordinates for each input scale
# outputs[("color",1,scale)] is the predicted image for frame_id=1
outputs["color"] = F.grid_sample(
            inputs["color"],
            outputs["sample"],
            padding_mode="border" ,align_corners=True)