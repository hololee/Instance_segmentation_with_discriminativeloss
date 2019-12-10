import numpy as np
from PIL import Image
from PIL import ImageFilter
import scipy.misc as misc
import matplotlib.pyplot as plt
import random


class dataCreator:
    def __init__(self, ships=(2, 5)):
        self.background_rotation = [0, 90, 180, 270]
        self.ship_counts = np.arange(ships[0], ships[1])

        self.ship_list = ["/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/ship1.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/ship2.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/ship3.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/ship4.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/ship5.png"]

        self.back_list = ["/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/back1.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/back2.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/back3.png",
                          "/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/source/back4.png"]

        self.color_map = [(229, 43, 80),
                          (255, 191, 0),
                          (153, 102, 204),
                          (251, 206, 177),
                          (127, 255, 212),
                          (0, 127, 255),
                          (137, 207, 240),
                          (245, 245, 220),
                          (0, 0, 255),
                          (0, 149, 182),
                          (138, 43, 226),
                          (222, 93, 131),
                          (205, 127, 50),
                          (150, 75, 0),
                          (127, 255, 0),
                          (114, 160, 193),
                          (176, 191, 26),
                          (240, 248, 255),
                          (241, 156, 187),
                          (77, 0, 64), ]

        # NOTICE : W by H
        self.coordinate_list = [(89, 87),
                                (246, 149),
                                (418, 74),
                                (65, 252),
                                (197, 356),
                                (354, 306),
                                (497, 229),
                                (60, 455),
                                (296, 484),
                                (454, 436), ]

    def _createBackgrounds(self):
        img = Image.open(np.random.choice(self.back_list, 1)[0])
        # rotate background.
        self.background = img.rotate(np.random.choice(self.background_rotation, 1)[0])
        self.background_gt = Image.new("RGBA", (512, 512), (0, 0, 0, 0))

    def _createShips(self):
        # choose number of ships.
        number_of_ship = np.random.choice(self.ship_counts, 1)[0]

        # chose locations of number of ships.
        coordinate_indexes = random.sample(range(len(self.coordinate_list)), number_of_ship)
        print(coordinate_indexes)

        box_image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
        box_image.paste(self.background)

        for idx, coordinate_index in enumerate(coordinate_indexes):
            coordinate = self.coordinate_list[coordinate_index]
            print(coordinate, coordinate_index)

            # choose ship and rotate.
            mg = misc.imread(np.random.choice(self.ship_list, 1)[0], mode='RGBA')
            ship_image = Image.fromarray(mg)

            ship_image = ship_image.rotate(np.random.randint(0, 365), resample=Image.BICUBIC, expand=True,
                                           fillcolor=(0, 0, 0, 0))

            # ship location offset.
            # offset = (coordinate[0] - ship_image.width, coordinate[1] - ship_image.height)
            offset = (coordinate[0] - ship_image.width // 2, coordinate[1] - ship_image.height // 2)
            # put ship to background.
            box_image.paste(ship_image, offset, ship_image)

            # NOTICE: put ship ground path to bacground_gt
            ship_image_gt = np.array(ship_image)
            # ship_image_gt = ship_image_gt[:, :, 0:3]
            # ship_image_gt[ship_image_gt[:, :, 0] > 0, 0] = self.color_map[idx][0]
            # ship_image_gt[ship_image_gt[:, :, 1] > 0, 1] = self.color_map[idx][1]
            # ship_image_gt[ship_image_gt[:, :, 2] > 0, 2] = self.color_map[idx][2]

            ship_image_gt_layer = ship_image_gt[:, :, 3]
            # plt.imshow(ship_image_gt_layer)
            # plt.show()
            ship_image_gt = np.zeros(ship_image_gt[:, :, 0:3].shape, dtype='uint8')
            ship_image_gt[ship_image_gt_layer > 5, 0] = self.color_map[idx][0]
            ship_image_gt[ship_image_gt_layer > 5, 1] = self.color_map[idx][1]
            ship_image_gt[ship_image_gt_layer > 5, 2] = self.color_map[idx][2]

            r, g, b = ship_image_gt[:, :, 0], ship_image_gt[:, :, 1], ship_image_gt[:, :, 2]
            ship_image_gt_mask = 0.2989 * r + 0.5870 * g + 0.1140 * b
            ship_image_gt_mask[ship_image_gt_mask > 0] = 255

            ship_image_gt_mask = np.expand_dims(ship_image_gt_mask, -1)
            ship_image_gt = np.concatenate([ship_image_gt, ship_image_gt_mask], -1).astype(np.uint8)

            ship_image_gt = Image.fromarray(ship_image_gt)

            self.background_gt.paste(ship_image_gt, offset, ship_image_gt)

        box_image = box_image.filter(ImageFilter.GaussianBlur(radius=0.7))
        # self.background_gt = self.background_gt.filter(ImageFilter.GaussianBlur(radius=0.5))

        # change to rgb.
        print("finish.")
        img = Image.new("RGB", box_image.size, (0, 0, 0))
        img.paste(box_image, mask=box_image.split()[3])  # 3 is the alpha channel

        gt = Image.new("RGB", self.background_gt.size, (0, 0, 0))
        gt.paste(self.background_gt, mask=self.background_gt.split()[3])  # 3 is the alpha channel

        # img.show()
        return np.array(img), np.array(gt)

    def generate_one(self):
        self._createBackgrounds()
        box_image, box_gt = self._createShips()
        box_gt_grayscale = box_gt.copy()
        box_gt_grayscale[box_gt_grayscale[:, :, 0] > 0, 0] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 0] > 0, 1] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 0] > 0, 2] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 1] > 0, 0] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 1] > 0, 1] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 1] > 0, 2] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 2] > 0, 0] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 2] > 0, 1] = 255
        box_gt_grayscale[box_gt_grayscale[:, :, 2] > 0, 2] = 255

        return box_image, box_gt, box_gt_grayscale


dataCreator = dataCreator()
for i in range(40):
    origin, color, grayscale = dataCreator.generate_one()
    misc.imsave('/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/test/origin_{0:03d}.png'.format(i),
                origin)
    misc.imsave('/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/test/color_{0:03d}.png'.format(i),
                color)
    misc.imsave('/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/test/grayscale_{0:03d}.png'.format(i),
                grayscale)
