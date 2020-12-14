from scipy.misc import imread
import numpy as np
import os
import config_etc


class DataGen:

    def __init__(self):
        self.batch_flag = 0

        self.dir = '/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/ouput'
        all_file_list = os.listdir(self.dir)
        all_file_list.sort()

        # file name.
        self.rgbs_name = []
        self.fgs_name = []
        self.insfgs_name = []

        # image arrays.
        self.rgb_images = []
        self.fg_images = []
        self.insfg_images = []

        # add images as list.
        for titles in all_file_list:
            if 'origin_' in titles:
                print(titles + ": color image")
                # color images
                self.rgbs_name.append(titles)
            elif 'grayscale_' in titles:
                print(titles + ": foreground")
                # foregrounds
                self.fgs_name.append(titles)

            elif 'color_' in titles:
                print(titles + ": instance ground truth")
                self.insfgs_name.append(titles)

        print("\n==========================================")
        # number of total images.
        self.total_number = len(self.rgbs_name)
        # all images have same size.
        print("number of total data : {}".format(len(self.rgbs_name)))
        print("==========================================\n")

    def load_images(self):
        for image_names in self.rgbs_name:
            real_path = self.dir + "/" + image_names
            # load images.
            self.rgb_images.append(imread(real_path, mode='RGB'))

        return self.rgb_images

    def load_labels(self):
        for image_names in self.fgs_name:
            real_path = self.dir + "/" + image_names
            # load images.
            self.fg_images.append(imread(real_path, mode='L'))

        # change range  0 to 1
        # self.fg_images = self.fg_images / np.amax(self.fg_images)

        return self.fg_images

    def load_instance_labels(self):
        for image_names in self.insfgs_name:
            real_path = self.dir + "/" + image_names
            # load images.
            self.insfg_images.append(imread(real_path, mode='L'))

        return self.insfg_images

    # get total number.
    def getTotalNumber(self):
        return self.total_number

    # get image shpape.
    def getImageSize(self):
        # height, width.
        return np.shape(np.array(self.rgb_images))[1], np.shape(np.array(self.rgb_images))[2]

    # next batch.
    def next_batch_ins(self, total_images, total_labels, total_islabels):

        batch_size = config_etc.BATCH_SIZE

        sub_batch_x, sub_batch_y, sub_batch_z = total_images[
                                                self.batch_flag: self.batch_flag + batch_size], total_labels[
                                                                                                self.batch_flag:self.batch_flag + batch_size], total_islabels[
                                                                                                                                               self.batch_flag:self.batch_flag + batch_size]
        self.batch_flag = (self.batch_flag + batch_size) % len(total_images)
        return sub_batch_x, sub_batch_y, sub_batch_z

    # next batch.
    def next_batch(self, total_images, total_labels):

        batch_size = config_etc.BATCH_SIZE

        sub_batch_x, sub_batch_y = total_images[
                                   self.batch_flag: self.batch_flag + batch_size], total_labels[
                                                                                   self.batch_flag:self.batch_flag + batch_size]
        self.batch_flag = (self.batch_flag + batch_size) % len(total_images)
        return sub_batch_x, sub_batch_y
