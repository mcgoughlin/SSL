# create a json file from a directory of datasets of images
# each subdir is a dataset with a folder for images and labels
# i want to create a json file that has the path to each image only for each dataset

import os
import json
import argparse

#schema for json file is a dictionary with keys as dataset names and values as a list of image paths
#dataset name is the name of the folder that contains the images and labels

def create_json(path):
    #path is the path to the directory of datasets
    #create a dictionary with keys as dataset names and values as a list of image paths for the 'images' folder
    dataset_dict = {}
    for dataset in os.listdir(path):
        dataset_dict[dataset] = []
        dataset_path = os.path.join(path, dataset)
        images_path = os.path.join(dataset_path, 'images')
        if not os.path.exists(images_path): continue
        for image in os.listdir(images_path):
            image_path = os.path.join(images_path, image)
            dataset_dict[dataset].append({'image':image_path})
    return dataset_dict

def write_json(dataset_dict, path):
    #dataset_dict is the dictionary of dataset names and image paths
    #path is the path to the json file
    with open(path, 'w') as outfile:
        json.dump(dataset_dict, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to directory of datasets')
    parser.add_argument('--json_path', type=str, help='path to json file')
    args = parser.parse_args()
    dataset_dict = create_json(args.path)
    write_json(dataset_dict, args.json_path)

if __name__ == '__main__':
    main()