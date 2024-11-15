"""
@author: pedroofRodenas
Available at https://gist.github.com/pedrofrodenas/4ab13c119927da61153a881a20480815
"""
import os

import pandas as pd
from PIL import Image
import tensorflow as tf

def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

def make_containing_dirs(path_list):
    for path in path_list:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


class SaverMNIST():
    def __init__(self, image_train_path, image_test_path, csv_train_path,
                 csv_test_path):

        self._image_format = '.png'

        self.store_image_paths = [image_train_path, image_test_path]
        self.store_csv_paths = [csv_train_path, csv_test_path]

        make_dirs(self.store_image_paths)
        make_containing_dirs(self.store_csv_paths)

        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        self.data = mnist.load_data()

    def run(self):

        for collection, store_image_path, store_csv_path in zip(self.data,
                                                                self.store_image_paths,
                                                                self.store_csv_paths):

            labels_list = []
            paths_list = []

            for index, (image, label) in enumerate(zip(collection[0],
                                                       collection[1])):
                im = Image.fromarray(image)
                width, height = im.size
                image_name = str(index) + self._image_format

                # Build save path
                save_path = os.path.join(store_image_path, image_name)
                im.save(save_path)

                labels_list.append(label)
                paths_list.append(save_path)

            df = pd.DataFrame({'image_paths': paths_list, 'labels': labels_list})

            df.to_csv(store_csv_path)


def add_mnist_original_images_if_not_exist():
    folder_name = 'dataset'
    os.makedirs(folder_name, exist_ok=True)
    if len(os.listdir(folder_name)) == 0:
        mnist_saver = SaverMNIST(image_train_path=f'{folder_name}/train',
                                 image_test_path=f'{folder_name}/test',
                                 csv_train_path=f'{folder_name}/train.csv',
                                 csv_test_path=f'{folder_name}/test.csv')
        mnist_saver.run()
        print(f"The mnist images for FID evaluation have been added to the {folder_name}")
    else:
        print("The mnist images for FID evaluation already exist, nothing to be done")
