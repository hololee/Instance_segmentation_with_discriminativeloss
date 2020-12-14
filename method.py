import tensorflow as tf
import config_etc
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2

# convolution type.
TYPE_NORMAL = 'normal'
TYPE_ATROUS = 'atrous'

# activate functions.
FUNC_RELU = 'relu'
NONE = 'none'

# this color map is wrap the result of instance segmentation.
color_map = [np.array([229, 43, 80]),
             np.array([255, 191, 0]),
             np.array([153, 102, 204]),
             np.array([251, 206, 177]),
             np.array([127, 255, 212]),
             np.array([0, 127, 255]),
             np.array([137, 207, 240]),
             np.array([245, 245, 220]),
             np.array([0, 0, 255]),
             np.array([0, 149, 182]),
             np.array([138, 43, 226]),
             np.array([222, 93, 131]),
             np.array([205, 127, 50]),
             np.array([150, 75, 0]),
             np.array([127, 255, 0]),
             np.array([114, 160, 193]),
             np.array([176, 191, 26]),
             np.array([240, 248, 255]),
             np.array([241, 156, 187]),
             np.array([77, 0, 64]), ]


# deeplab model layer
def layers_deeplab(type, input_map, tar_dim, name, act_func, batch_norm, pooling={'size': 2, 'stride': 2}):
    weight = tf.Variable(
        tf.random_normal([3, 3, input_map.get_shape().as_list()[3], tar_dim], stddev=config_etc.TRAIN_STDDV), name=name)
    bias = tf.Variable([0.1])

    # choose type
    if type == TYPE_NORMAL:
        conv_result = tf.nn.conv2d(input_map, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
    elif type == TYPE_ATROUS:
        conv_result = tf.nn.atrous_conv2d(input_map, weight, rate=2, padding="VALID") + bias

    # activation
    if act_func == None:
        pass
    elif act_func == FUNC_RELU:
        conv_result = tf.nn.relu(conv_result, name + FUNC_RELU)

    # batch normalization
    if batch_norm.use_batch_norm:
        # using batch normalization.
        conv_result = tf.layers.batch_normalization(conv_result, center=True, scale=True, training=batch_norm.is_train)

    # max pooling.
    if pooling != None:
        conv_result = tf.nn.max_pool(conv_result, ksize=[1, pooling['size'], pooling['size'], 1],
                                     strides=[1, pooling['stride'], pooling['stride'], 1],
                                     padding='SAME')

    # print shape of array
    print(conv_result.shape)

    return conv_result


# Enet model layer
def layer_Enet_initial(input_map, name):
    # concate 13 conv features , 3 pooling result, output has 16 dim size.
    conv_weight = tf.Variable(
        tf.random_normal(shape=[3, 3, input_map.get_shape().as_list()[3], 13], stddev=config_etc.TRAIN_STDDV),
        name=name + "_filter")
    conv_bias = tf.Variable([0.1], name=name + "_bias")
    conv_part = tf.nn.conv2d(input_map, conv_weight, strides=[1, 2, 2, 1], padding="SAME",
                             name=name + "_conv_part") + conv_bias
    pooling_part = tf.nn.max_pool(input_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    concat = tf.concat(values=[conv_part, pooling_part], axis=-1)

    print("layer({}) : {}".format(name, concat.get_shape()))

    return concat


def layer_enet_bottle_neck(input_map, layer_type, training, name):
    """

    :param input_map:
    :param layer_type:
        dic format:
        "ver" : 1, 2  - bottleneck version 1 or 2
        "type" : "regular", "dilated" ,"transpose_conv", "asymmetric"
        "down_sampling" : True, False

        "conv_size" : (int) size
        "dilated_rate" : (int) rate
        "asymmetric_rate" : (int) rate
        "target_dim" : (int) dim
        "projection_ratio" : (int) ratio


    :param training:
    :param name:
    :return:
    """

    if layer_type["ver"] == 1:
        drop_param = 0.01
    else:
        drop_param = 0.1

    # 1by1 conv---------------------------------------------------------------------------------------------------------
    temp_val = (lambda x: 2 if layer_type["down_sampling"] == x else 1)(1)
    weight_1 = tf.Variable(
        tf.random_normal(
            shape=[temp_val, temp_val, input_map.get_shape().as_list()[3],
                   input_map.get_shape().as_list()[3] // layer_type["projection_ratio"]]))
    weighted = tf.nn.conv2d(input_map, weight_1, strides=[1, temp_val, temp_val, 1],
                            padding="SAME",
                            name=name + "_first_1by1")
    weighted = p_relu(weighted, name + "alpha1")
    weighted = tf.layers.batch_normalization(weighted, center=True, scale=True, training=training)
    print("┌ step_1by1 : {}".format(weighted.get_shape()))
    # CONV--------------------------------------------------------------------------------------------------------------
    if layer_type["type"] == "regular":
        weight_regular = tf.Variable(tf.random_normal(
            shape=[layer_type["conv_size"], layer_type["conv_size"], weighted.get_shape().as_list()[3],
                   input_map.get_shape().as_list()[3] // layer_type["projection_ratio"]],
            stddev=config_etc.TRAIN_STDDV), name=name + "_regular")
        weighted = tf.nn.conv2d(weighted, weight_regular, strides=[1, 1, 1, 1], padding="SAME",
                                name=name + "_regular")

    # 다른 conv 방식.
    elif layer_type["type"] == "dilated":
        weight_dilated = tf.Variable(tf.random_normal(
            shape=[layer_type["conv_size"], layer_type["conv_size"], weighted.get_shape().as_list()[3],
                   weighted.get_shape().as_list()[3]], stddev=config_etc.TRAIN_STDDV), name=name + "_dilated")
        weighted = tf.nn.atrous_conv2d(weighted, weight_dilated, rate=layer_type["dilated_rate"], padding="SAME")

    elif layer_type["type"] == "transpose_conv":
        reduced_depth = input_map.get_shape().as_list()[3] // layer_type["projection_ratio"]

        weight_deconv = tf.Variable(tf.random_normal(
            shape=[layer_type["conv_size"], layer_type["conv_size"], weighted.get_shape().as_list()[3],
                   weighted.get_shape().as_list()[3]], stddev=config_etc.TRAIN_STDDV), name=name + "_transpose_conv")

        weighted = tf.nn.conv2d_transpose(value=weighted, filter=weight_deconv,
                                          output_shape=[weighted.get_shape().as_list()[0],
                                                        weighted.get_shape().as_list()[1] * 2,
                                                        weighted.get_shape().as_list()[2] * 2, reduced_depth],
                                          strides=[1, 2, 2, 1], padding="SAME")


    elif layer_type["type"] == "asymmetric":
        asymmetric_w_1 = tf.Variable(
            tf.random_normal(
                [layer_type["asymmetric_rate"], 1, weighted.get_shape().as_list()[3],
                 weighted.get_shape().as_list()[3]],
                stddev=config_etc.TRAIN_STDDV), name=name + "_asymmetric1")
        weighted = tf.nn.conv2d(weighted, asymmetric_w_1, strides=[1, 1, 1, 1], padding="SAME")
        asymmetric_w_2 = tf.Variable(
            tf.random_normal(
                [1, layer_type["asymmetric_rate"], weighted.get_shape().as_list()[3],
                 weighted.get_shape().as_list()[3]],
                stddev=config_etc.TRAIN_STDDV), name=name + "_asymmetric2")
        weighted = tf.nn.conv2d(weighted, asymmetric_w_2, strides=[1, 1, 1, 1], padding="SAME")

    print("┌ step_conv_{} : {}".format(layer_type["type"], weighted.get_shape()))
    weighted = p_relu(weighted, name=name + "alpha2")
    weighted = tf.layers.batch_normalization(weighted, center=True, scale=True, training=training)

    # 1by1 conv---------------------------------------------------------------------------------------------------------
    weight_2 = tf.Variable(
        tf.random_normal(shape=[1, 1, weighted.get_shape().as_list()[3],
                                layer_type["target_dim"]]))
    weighted = tf.nn.conv2d(weighted, weight_2, strides=[1, 1, 1, 1],
                            padding="SAME",
                            name=name + "_second_1by1")
    print("┌ step_1by1 : {}".format(weighted.get_shape()))
    # dropout - regulaizer ---------------------------------------------------------------------------------------------------------
    weighted = tf.layers.dropout(weighted, rate=drop_param, training=training, name=name + "_dropout")

    # down sampling -----------------------------------------------------------------------------------------------------
    if layer_type["down_sampling"]:
        max_pool = tf.nn.max_pool(input_map, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                  name=name + "_downsamp")

        if layer_type["ver"] != "full_conv":
            inputs_shape = input_map.get_shape().as_list()
            depth_to_pad = abs(inputs_shape[3] - layer_type["target_dim"])

            # padding 0 dims
            paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, depth_to_pad]])
            max_pool = tf.pad(max_pool, paddings=paddings, name=name + '_padding')

            # TODO : why using add?
            weighted = tf.add(weighted, max_pool)

            print("┌ # step(down_sampling) : {}".format(weighted.get_shape()))

    print("layer({}) : {}".format(name, weighted.get_shape()))

    return weighted


def bi_linear_interpolation(input_map, original_map_size=(530, 500)):
    conv_result = tf.image.resize_images(input_map, size=original_map_size,
                                         method=tf.image.ResizeMethod.BILINEAR)

    # print shape of array
    print(conv_result.shape)

    return conv_result


def p_relu(_x, name):
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def discriminative_loss_single(
        prediction,
        correct_label,
        feature_dim,
        label_shape,
        delta_v,
        delta_d,
        param_var,
        param_dist,
        param_reg):
    """
    discriminative loss
    :param prediction: inference of network
    :param correct_label: instance label
    :param feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cut off variance distance
    :param delta_d: cut off cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    """
    correct_label = tf.reshape(
        correct_label, [label_shape[1] * label_shape[0]]
    )
    reshaped_pred = tf.reshape(
        prediction, [label_shape[1] * label_shape[0], feature_dim]
    )

    # calculate instance nums
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    # calculate instance pixel embedding mean vec
    segmented_sum = tf.unsorted_segment_sum(
        reshaped_pred, unique_id, num_instances)
    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(
        mu_band_rep,
        (num_instances *
         num_instances,
         feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff), axis=1)
    zero_vector = tf.zeros(1, dtype=tf.float32)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    l_reg = tf.reduce_mean(tf.norm(mu, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg


def discriminative_loss(prediction, correct_label, feature_dim, image_shape,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    """
    :return: discriminative loss and its three components
    """

    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(
            prediction[i], correct_label[i], feature_dim, image_shape, delta_v, delta_d, param_var, param_dist,
            param_reg)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_reg = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(
        cond, body, [
            correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, output_ta_reg, 0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation file hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def apply_clustering(binary_seg_result, instance_seg_result):
    # get embedding feats and coords

    binary_seg_result = _morphological_process(binary_seg_result)

    idx = np.where(binary_seg_result == 255)
    embedding_feats = instance_seg_result[idx]
    coordinate = np.vstack((idx[1], idx[0])).transpose()

    # dbscan cluster
    db = DBSCAN(eps=config_etc.DBSCAN_EPS, min_samples=config_etc.DBSCAN_MIN_SAMPLES)
    try:
        features = StandardScaler().fit_transform(embedding_feats)
        # features = StandardScaler().fit_transform(embedding_feats)
        db.fit(features)
        # db.fit(embedding_feats)
    except Exception as err:
        print("error: {0}".format(err))
        return None

    # TODO: check the output of clustering.
    db_labels = db.labels_
    unique_labels = np.unique(db_labels)

    num_clusters = len(unique_labels)

    cluster_centers = db.components_

    mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)

    if db_labels is None:
        return None, None

    coords = []

    for index, label in enumerate(unique_labels.tolist()):
        if label == -1:
            continue
        idx = np.where(db_labels == label)
        pix_coord_idx = tuple((coordinate[idx][:, 1], coordinate[idx][:, 0]))
        mask[pix_coord_idx] = color_map[index]
        coords.append(coordinate[idx])

    return mask, coords


# ====== crf ========
def apply_crf(original_image_path, output_image_path, final_result_path):
    # Get im{read,write} from somewhere.
    try:
        from cv2 import imread, imwrite
    except ImportError:
        # Note that, sadly, skimage unconditionally import scipy and matplotlib,
        # so you'll need them if you don't have OpenCV. But you probably have them.
        from skimage.io import imread, imsave

        imwrite = imsave
        # TODO: Use scipy instead.

    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

    fn_im = original_image_path
    fn_anno = output_image_path
    fn_output = final_result_path

    # fn_im = "/data1/LJH/cvpppnet/A1/plant002_rgb.png"
    # fn_anno = "/data1/LJH/cvpppnet/A1_predict/output_001.png"
    # fn_output = "/data1/LJH/cvpppnet/semantic_segmentation_usinng_crf/output.png"

    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)

    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread(fn_anno).astype(np.uint32)
    anno_lbl = anno_rgb[:, :, 0] + (anno_rgb[:, :, 1] << 8) + (anno_rgb[:, :, 2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print(
            "Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print(
            "If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    # else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=6, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(95, 95), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]
    # TODO: save image

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(3):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)

    # final result.
    final_result = MAP.reshape(img.shape)
    rt = np.zeros(final_result.shape)
    rt[np.where(final_result > 30)] = 255

    # save image.
    imwrite(fn_output, rt)

    fig = plt.figure()
    fig.set_size_inches(10, 2)  # 1800 x600

    ax1 = fig.add_subplot(1, 5, 1)
    ax2 = fig.add_subplot(1, 5, 2)
    ax3 = fig.add_subplot(1, 5, 3)
    ax4 = fig.add_subplot(1, 5, 4)
    ax5 = fig.add_subplot(1, 5, 5)

    ax1.set_title("origin")
    ax2.set_title("output")
    ax3.set_title("after CRF")
    ax4.set_title("masking : over.{}".format(30))
    ax5.set_title("final")

    ax1.imshow(img)
    ax2.imshow(anno_rgb)

    final_result = MAP.reshape(img.shape)

    ax3.imshow(final_result)

    temp = np.zeros(final_result.shape)
    temp[np.where(final_result > 30)] = 255

    # to decide the dividing range, using iou

    ax4.imshow(temp)

    temp2 = np.copy(img)
    temp2[np.where(final_result <= 30)] = 0

    ax5.imshow(temp2)

    # save image.
    imwrite(fn_output, temp2)

    plt.show()

    return rt
