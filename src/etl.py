from PIL import Image
import os
import cv2

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


def get_id(path):
    data_path = '/datasets/MaskedFace-Net/train'
    categories = os.listdir(data_path)
    labels = [i for i in range(len(categories))]
    label_dict = dict(zip(categories,labels))
    print(label_dict)
    print(categories)
    print(labels)
    return None

def print_image(id_path):
    pil_im = Image.open(id_path)
    print(pil_im)
    
