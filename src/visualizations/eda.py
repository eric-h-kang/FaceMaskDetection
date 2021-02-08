from PIL import Image
import os
import pandas as pd
import seaborn as sns
import glob
all_file_paths = [tpc, tpic, tpu, hopc, hopic, hopu, vpc, vpic, vpu]

def count_images(d):
    image_desc = []
    image_count = []
    count = 0
    for path in os.listdir(d):
        if os.path.isfile(os.path.join(d, path)):
            count += 1
        image_count.append(count)
    split = d.split("/")
    print("There are " + str(count) + " images of " + split[4]+ " faces in the " + split[3] + " folder.")
    image_desc.append(count)
    image_desc.append(split[4])
    image_desc.append(split[3])
    return image_desc

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

def visualize_image():
    image_data = []
    for d in all_file_paths:
        x = count_images(d)
        image_data.append(x)
    image_data = pd.DataFrame(image_data)
    image_data.columns = ['count', 'mask category', 'dataset']
    sns.barplot(x = 'mask category', y = 'count', hue = 'dataset', data = image_data)
    
def visualize_stats():
    all_types = ["*Mask_Mouth_Chin.jpg", "*Mask_Nose_Mouth.jpg", "*Mask_Chin.jpg"]
    incorrect_folders = [tpic, hopic, vpic]
    image_stats = []
    for d in all_types:
        for i in incorrect_folders:
            x = get_incorrect(i,d)
            image_stats.append(x)
    image_stats = pd.DataFrame(image_stats)
    image_stats.columns = ["count", "Reason", "dataset"]
    sns.barplot(x = 'Reason', y = 'count', hue = 'dataset', data = image_stats)