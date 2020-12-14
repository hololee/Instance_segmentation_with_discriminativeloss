import method as md
import tensorflow as tf
import numpy as np
import os
from DataGen import DataGen
from placeHolders import placeHolders
import config_etc
import matplotlib.pyplot as plt
import scipy.misc
import datetime
import time

# threshold value
threshold_val = 40

# calculate time for file name.
current_milli_time = lambda: int(round(time.time() * 1000))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ready dataset.
dataG = DataGen()

# loaded images numpy array.
rgb_images = np.array(dataG.load_images())
fg_images = np.array(dataG.load_labels())
ins_images = np.array(dataG.load_instance_labels())

# reshape
fg_images = np.reshape(fg_images, [fg_images.shape[0], fg_images.shape[1], fg_images.shape[2], 1])
ins_images = np.reshape(ins_images, [ins_images.shape[0], ins_images.shape[1], ins_images.shape[2], 1])

print("rgb_images : " + str(np.shape(rgb_images)))
print("fg_images : " + str(np.shape(fg_images)))
print("ins_images : " + str(np.shape(ins_images)))

# create input place holder and apply.
ph = placeHolders(input_images=rgb_images, input_labels=fg_images)

## network set.
num_classes = 1

input = tf.zeros(shape=[10, 512, 512, 3])

# POINT ==== initial Network.=====
net = md.layer_Enet_initial(ph.input_data, name="initial")
print(ph.input_data.get_shape())
# net = md.layer_Enet_initial(input, name="initial")

# POINT :######################## Encoder Part ######################## conclude par1 & part2
# NOTICE ==== ver 1
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": True, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_0")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_1")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_2")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_3")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 1, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 64, "projection_ratio": 4},
                                training=ph.is_train, name="bottleneck1_4")

# NOTICE ==== ver 2
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": True, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck2_0")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck2_1")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 2},
                                training=ph.is_train,
                                name="bottleneck2_2")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "asymmetric_rate": 5},
                                training=ph.is_train,
                                name="bottleneck2_3")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 4},
                                training=ph.is_train,
                                name="bottleneck2_4")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                name="bottleneck2_5")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 8},
                                training=ph.is_train,
                                name="bottleneck2_6")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "asymmetric_rate": 5},
                                training=ph.is_train,
                                name="bottleneck2_7")
net = md.layer_enet_bottle_neck(net, layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                 "target_dim": 128, "projection_ratio": 4, "dilated_rate": 16},
                                training=ph.is_train,
                                name="bottleneck2_8")

# POINT : branch 1 from the net. =============================   Embedding branch   ====================================
embedding = md.layer_enet_bottle_neck(net,
                                      layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck3_1")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 128, "projection_ratio": 4, "dilated_rate": 2},
                                      training=ph.is_train,
                                      name="em_bottleneck3_2")
embedding = md.layer_enet_bottle_neck(embedding, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False,
                                                             "conv_size": 3,
                                                             "target_dim": 128, "projection_ratio": 4,
                                                             "asymmetric_rate": 5},
                                      training=ph.is_train,
                                      name="em_bottleneck3_3")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 128, "projection_ratio": 4, "dilated_rate": 4},
                                      training=ph.is_train,
                                      name="em_bottleneck3_4")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 2, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck3_5")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 128, "projection_ratio": 4, "dilated_rate": 8},
                                      training=ph.is_train,
                                      name="em_bottleneck3_6")
embedding = md.layer_enet_bottle_neck(embedding, layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False,
                                                             "conv_size": 3,
                                                             "target_dim": 128, "projection_ratio": 4,
                                                             "asymmetric_rate": 5},
                                      training=ph.is_train,
                                      name="em_bottleneck3_7")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 2, "type": "dilated", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 128, "projection_ratio": 4, "dilated_rate": 16},
                                      training=ph.is_train,
                                      name="em_bottleneck3_8")
# NOTICE ==== ver 4
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 4, "type": "transpose_conv", "down_sampling": False,
                                                  "conv_size": 3,
                                                  "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck4_0")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 4, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck4_1")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 4, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck4_2")

# NOTICE ==== ver 5
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 5, "type": "transpose_conv", "down_sampling": False,
                                                  "conv_size": 3,
                                                  "target_dim": 16, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck5_0")
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": 5, "type": "regular", "down_sampling": False, "conv_size": 3,
                                                  "target_dim": 16, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_bottleneck5_1")

# fullconv
embedding = md.layer_enet_bottle_neck(embedding,
                                      layer_type={"ver": "full_conv", "type": "transpose_conv", "down_sampling": False,
                                                  "conv_size": 3,
                                                  "target_dim": 2, "projection_ratio": 4}, training=ph.is_train,
                                      name="em_full_conv")

# POINT : branch 2 from the net. ============================   Segmentation branch   ===================================
segmentation = md.layer_enet_bottle_neck(net,
                                         layer_type={"ver": 2, "type": "regular", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck3_1")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "dilated", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4, "dilated_rate": 2},
                                         training=ph.is_train,
                                         name="seg_bottleneck3_2")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4,
                                                     "asymmetric_rate": 5},
                                         training=ph.is_train,
                                         name="seg_bottleneck3_3")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "dilated", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4, "dilated_rate": 4},
                                         training=ph.is_train,
                                         name="seg_bottleneck3_4")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "regular", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck3_5")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "dilated", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4, "dilated_rate": 8},
                                         training=ph.is_train,
                                         name="seg_bottleneck3_6")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "asymmetric", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4,
                                                     "asymmetric_rate": 5},
                                         training=ph.is_train,
                                         name="seg_bottleneck3_7")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 2, "type": "dilated", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 128, "projection_ratio": 4, "dilated_rate": 16},
                                         training=ph.is_train,
                                         name="seg_bottleneck3_8")
# NOTICE ==== ver 4
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 4, "type": "transpose_conv", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck4_0")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 4, "type": "regular", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck4_1")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 4, "type": "regular", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 64, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck4_2")

# NOTICE ==== ver 5
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 5, "type": "transpose_conv", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 16, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck5_0")
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": 5, "type": "regular", "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 16, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_bottleneck5_1")

# fullconv
segmentation = md.layer_enet_bottle_neck(segmentation,
                                         layer_type={"ver": "full_conv", "type": "transpose_conv",
                                                     "down_sampling": False,
                                                     "conv_size": 3,
                                                     "target_dim": 1, "projection_ratio": 4}, training=ph.is_train,
                                         name="seg_full_conv")

# ======================================= End seg======================================================================

# POINT : output of the each branch.
# predict
predict_images = segmentation
embedding_outputs = embedding

# POINT ##===============================   Calculate segmentation loss  =======================================########

# loss
# (batch_size, num_classes)
# num_classes = 2 : background, plant NOTICE: one-hot encode to image.
flat_logits = tf.reshape(tensor=predict_images, shape=(-1, 1))
flat_labels = tf.reshape(tensor=ph.ground_truth, shape=(-1, 1))
# flat_logits = tf.reshape(tensor=predict_images, shape=(-1, 2))
# flat_labels = tf.reshape(tensor=ph.ground_truth, shape=(-1, 2))

# more than two classes, use soft_max_cross_entropy.
# less than two classes, use sigmoid_cross_entropy.
# seg_cross_entropies = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
#                                                                             labels=flat_labels, dim=-1))

seg_cross_entropies = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(flat_labels, flat_logits))))

# seg_optimizer = tf.train.AdamOptimizer(learning_rate=ph.learning_rate).minimize(seg_cross_entropies)


# POINT ##===============================   Calculate embedding loss  ==========================================########

print("shape : {}".format(embedding_outputs.get_shape()))

pix_image_shape = (embedding_outputs.get_shape().as_list()[1], embedding_outputs.get_shape().as_list()[2])

instance_segmentation_loss, l_var, l_dist, l_reg = md.discriminative_loss(
    embedding_outputs, ph.ins_label, 2,
    pix_image_shape, 0.5, 3.0, 1.0, 1.0, 0.001
)

# POINT #================================= Calculate total loss =================================================#######

total_loss = seg_cross_entropies + instance_segmentation_loss
optimizer = tf.train.AdamOptimizer(learning_rate=ph.learning_rate).minimize(total_loss)

# POINT ##================================= Train two network.===================================================#######

# counts iteration.
BATCH_COUNT = dataG.getTotalNumber() // config_etc.BATCH_SIZE

# saver.
saver = tf.train.Saver(max_to_keep=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(config_etc.TOTAL_EPOCH):
        print("======= current epoch  : {} ======".format(epoch + 1))

        learn_rate = config_etc.LEARNING_RATE
        if epoch > 65:
            learn_rate = config_etc.LEARNING_RATE_v2
        if epoch > 105:
            learn_rate = config_etc.LEARNING_RATE_v3

        for batch_count in range(BATCH_COUNT):

            # get source batch
            batch_x, batch_y, batch_z = dataG.next_batch_ins(total_images=rgb_images, total_labels=fg_images,
                                                             total_islabels=ins_images)

            # train.
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            _, _ = sess.run([optimizer, extra_update_ops], feed_dict={ph.input_data: batch_x,
                                                                      ph.ground_truth: batch_y,
                                                                      ph.ins_label: batch_z,
                                                                      ph.is_train: True,
                                                                      ph.learning_rate: learn_rate})

            image_result_predict = sess.run(predict_images, feed_dict={ph.input_data: batch_x, ph.is_train: False})
            embedding_result_predict = sess.run(embedding_outputs,
                                                feed_dict={ph.input_data: batch_x, ph.is_train: False})

            # save the result of semantic segmentation at last epoch.
            if epoch == (config_etc.TOTAL_EPOCH - 1):
                for index, image in enumerate(image_result_predict):
                    # save image.
                    scipy.misc.imsave(
                        '/data1/LJH/Instance_segmentation_with_discriminativeloss/datset/result/instance_seg/out_epc{}_{}.png'.format(epoch + 1,
                                                                                            current_milli_time()),
                        np.squeeze(image))

            # calculate loss.
            if batch_count % 8 == 0:
                total_loss_val = sess.run(total_loss, feed_dict={ph.input_data: batch_x,
                                                                 ph.ins_label: batch_z,
                                                                 ph.ground_truth: batch_y,
                                                                 ph.is_train: False})
                seg_loss_val, ins_loss_val = sess.run([seg_cross_entropies, instance_segmentation_loss],
                                                      feed_dict={ph.input_data: batch_x,
                                                                 ph.ins_label: batch_z,
                                                                 ph.ground_truth: batch_y,
                                                                 ph.is_train: False})

                print("train_loss : {} , seg : {}, ins : {}".format(total_loss_val, seg_loss_val, ins_loss_val))
                print("image_result_predict # min : {} , max : {}".format(np.amin(image_result_predict),
                                                                          np.amax(image_result_predict)))

                # a = tf.nn.softmax(image_result_predict, dim=0)
                # image_result = sess.run(a)
                # after calculating loss. adjust softmax.
                # image_result_predict = image_result_predict / np.amax(image_result_predict)

                # process crf.
                # h, w = dataG.getImageSize()
                # output_crf = method.dense_crf(img=batch_x, probs=image_result_predict, n_iters=5)

                if epoch == (config_etc.TOTAL_EPOCH - 1):
                    try:
                        fig = plt.figure()
                        fig.set_size_inches(9, 4)  # 1800 x600
                        ax1 = fig.add_subplot(1, 3, 1)
                        ax2 = fig.add_subplot(1, 3, 2)
                        ax3 = fig.add_subplot(1, 3, 3)

                        ax1.imshow(batch_x[0])
                        # ax2.imshow(np.squeeze(batch_y[0]), cmap='jet') # g.t
                        # temp = image_result_predict[0]
                        temp = image_result_predict[0] * 255 // np.amax(image_result_predict[0])
                        temp[np.where(temp > threshold_val)] = 255
                        temp[np.where(temp <= threshold_val)] = 0

                        ax2.imshow(np.squeeze(temp), cmap='jet')

                        mask, _ = md.apply_clustering(
                            binary_seg_result=np.squeeze(temp),
                            instance_seg_result=embedding_result_predict[0])

                        ax3.imshow(mask)
                        plt.show()

                    except Exception as ex:
                        print("error occured:{0}", ex)

        if epoch == (config_etc.TOTAL_EPOCH - 1):
            # save model params.
            saver.save(sess, '/data1/LJH/Instance_segmentation_with_discriminativeloss/model')
