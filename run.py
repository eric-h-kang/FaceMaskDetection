from PIL import Image


import sys
import os
import json


#file path for random covered images in the train dataset
tpc = '/datasets/MaskedFace-Net/train/covered' #train path covered
tpic = '/datasets/MaskedFace-Net/train/incorrect' #train path incorrect
tpu = '/datasets/MaskedFace-Net/train/uncovered' #train path uncovered

hopc = '/datasets/MaskedFace-Net/holdout/covered' #holdout path covered
hopic = '/datasets/MaskedFace-Net/holdout/incorrect' #holdout path incorrect
hopu = '/datasets/MaskedFace-Net/holdout/uncovered' #holdout path uncovered

vpc = '/datasets/MaskedFace-Net/validation/covered' #validation path covered
vpic = '/datasets/MaskedFace-Net/validation/incorrect' #validation path incorrect
vpu = '/datasets/MaskedFace-Net/validation/uncovered' #validation path uncovered

covered_path = '/datasets/MaskedFace-Net/train/covered/14931_Mask.jpg'
pil_im = Image.open(covered_path)
hopc = '/datasets/MaskedFace-Net/holdout/covered'

data_path = '/datasets/MaskedFace-Net/holdout'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories,labels))

test_params = 'test/testdata/test-params.json'


print(label_dict)
print(categories)



mask_image = Image.open(covered_path)

incorrect_path = '/datasets/MaskedFace-Net/train/incorrect/29943_Mask_Chin.jpg'

#Here is an idea of the kind of images shown in our database so far. They represent if masks are being worn properly

import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models


model = models.resnet50(pretrained=True)

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)
        
        return target_activations, x
    



def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


# if __name__ == '__main__':
#     """ python grad_cam.py <path_to_image>
#     1. Loads an image with opencv.
#     2. Preprocesses it for VGG19 and converts to a pytorch variable.
#     3. Makes a forward pass to find the category index with the highest score,
#     and computes intermediate activations.
#     Makes the visualization. """

#     args = get_args()

#     # Can work with any model, but it assumes that the model has a
#     # feature method, and a classifier method,
#     # as in the VGG models in torchvision.
    model_path = '../Config/inceptionResnetV1.pth'
    model = torch.load('../Config/inceptionResnetV1.pth',map_location=torch.device('cpu'))
    model.eval()
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)

covered_path = '/datasets/MaskedFace-Net/train/covered/14931_Mask.jpg'

def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param


#from src.associated_images import *
#from src.training_image_classifier import *


#import functions from files
sys.path.insert(0, 'src') # add library code to path

def main(targets):
    if 'test' in targets:
        
        
        import glob
        def get_incorrect(file_type,d):
            img_desc = []
            tifCounter = len(glob.glob1(file_type,d))
            split = file_type.split("/")
           # print(split)
            print("There are " + str(tifCounter) + " images of " + d + " in the " + split[3] + " folder" )
            img_desc.append(tifCounter)
            img_desc.append(d[6:-4].replace("_", " "))
            img_desc.append(split[3])
            return img_desc
    
        
        covered_path = '/datasets/MaskedFace-Net/train/covered/14931_Mask.jpg'
        
        all_types = ["*Mask_Mouth_Chin.jpg", "*Mask_Nose_Mouth.jpg", "*Mask_Chin.jpg", "*Mask_Nose_Mouth.jpg"]
        incorrect_folders = [tpic, hopic, vpic]
        image_stats = []
        for d in all_types:
            for i in incorrect_folders:
                x = get_incorrect(i,d)
                image_stats.append(x)
        print(image_stats)
       
        #instances_file = 
        p = load_params(test_params)
        #perform etl

        import os.path
        images  = covered_path
        grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda = False)
        img = cv2.imread(images, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        target_index = None
        mask = grad_cam(input, target_index)
        show_cam_on_image(img, mask)
        gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
        #print(model._modules.items())
        gb = gb_model(input, index=target_index)
        gb = gb.transpose((1, 2, 0))
        cam_mask = cv2.merge([mask, mask, mask])
        cam_gb = deprocess_image(cam_mask*gb)
        gb = deprocess_image(gb)

        cv2.imwrite('gb.jpg', gb)
        cv2.imwrite('cam_gb.jpg', cam_gb)
        print('GradCam image dispayed in same path')

    return

#first call to start data pipeline
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)