import tensorflow as tf
import numpy as np
from PIL import Image
import os
from os import path as osp
from glob import glob
import model as modellib
from inference_config import inference_config
import imageio
import utils
import visualize
import matplotlib.pyplot as plt

slim = tf.contrib.slim


def main(argv=None):
    ROOT_DIR = os.getcwd()

    with tf.gfile.FastGFile("./model/mask_rcnn.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    print('Graph loaded.')

    

    with tf.Session() as sess:
        folder = 'model'
        file = '150_512.jpg'
        filename = os.path.join(ROOT_DIR,folder,file) #image of the size defined in the config
        image = imageio.imread(filename)
        print(image.shape)
        images = [image]
        print("Processing {} images".format(len(images)))
        for im in images:
             modellib.log("image", im)
        print('RGB image loaded and preprocessed.')
        molded_images, image_metas, windows = mold_inputs(images)
        print(molded_images.shape)
        print('Images meta: ',image_metas)
        img_ph = sess.graph.get_tensor_by_name('input_image:0')
        print(img_ph)
        img_meta_ph = sess.graph.get_tensor_by_name('input_image_meta:0')
        print(img_meta_ph)
        detectionsT = sess.graph.get_tensor_by_name('output_detections:0')
        print('Found ',detectionsT)
        mrcnn_classT = sess.graph.get_tensor_by_name('output_mrcnn_class:0')
        print('Found ',mrcnn_classT)
        mrcnn_bboxT = sess.graph.get_tensor_by_name('output_mrcnn_bbox:0')
        print('Found ', mrcnn_bboxT)
        mrcnn_maskT = sess.graph.get_tensor_by_name('output_mrcnn_mask:0')
        print('Found ', mrcnn_maskT)
        roisT = sess.graph.get_tensor_by_name('output_rois:0')
        print('Found ', roisT)
        
        np.set_printoptions(suppress=False,precision=4)
        print('Windows', windows.shape,' ',windows)
        detections = sess.run(detectionsT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas})
        #print('Detections: ',detections[0].shape, detections[0])
        mrcnn_class = sess.run(mrcnn_classT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas})
        #print('Classes: ',mrcnn_class[0].shape, mrcnn_class[0])
        mrcnn_bbox = sess.run(mrcnn_bboxT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas})
        #print('BBoxes: ',mrcnn_bbox[0].shape, mrcnn_bbox[0])
        mrcnn_mask = sess.run(mrcnn_maskT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas})
        #print('Masks: ',mrcnn_mask[0].shape )#, outputs1[0])
        rois = sess.run(roisT, feed_dict={img_ph: molded_images, img_meta_ph: image_metas})
        #print('Rois: ',rois[0].shape, rois[0])

        
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        r = results[0]
        #print(r)
        print (r['scores'][0])
        print (r['class_ids'][0])
        print (r['rois'][0])
        print (r['masks'][0].shape)

        class_names = ["BG","nuclei"]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], ax=get_ax())
    print('Done')
    return 0

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w,h = im.size
    for i in range(w):
        for j in range(h):
            r,g,b = im.getpixel((i,j))
            if r != g != b: return False
    return True


def mold_inputs(images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        print('IMAGE_PADDING: ',inference_config.IMAGE_PADDING)
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=inference_config.IMAGE_MIN_DIM,
                max_dim=inference_config.IMAGE_MAX_DIM,
                padding=inference_config.IMAGE_PADDING)
            print(image.shape)
            print('Image resized at: ', molded_image.shape)
            print(window)
            print(scale)
            """Takes RGB images with 0-255 values and subtraces
                   the mean pixel and converts it to float. Expects image
                   colors in RGB order."""
            molded_image = mold_image(molded_image, inference_config)
            print('Image molded')
            #print(a)
            """Takes attributes of an image and puts them in one 1D array."""
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([inference_config.NUM_CLASSES], dtype=np.int32))
            print('Meta of image prepared')
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +            # size=1
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta

def unmold_detections(detections, mrcnn_mask, image_shape, window):
    """Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)]
    mrcnn_mask: [N, height, width, num_classes]
    image_shape: [height, width, depth] Original size of the image before resizing
    window: [y1, x1, y2, x2] Box in the image where the real image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
    print('Number of detections: ',N)
    print('Window: ',window)
    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    print('boxes',boxes.shape,' ',boxes)
    class_ids = detections[:N, 4].astype(np.int32)
    print('Class_ids: ',class_ids.shape,' ',class_ids)
    scores = detections[:N, 5]
    print('Scores: ',scores.shape,' ',scores)
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]
    print('Masks: ',masks.shape)# masks)
    # Compute scale and shift to translate coordinates to image domain.
    print(image_shape[0])
    print(window[2] - window[0])
    h_scale = image_shape[0] / (window[2] - window[0])
    print('h_scale: ',h_scale)
    w_scale = image_shape[1] / (window[3] - window[1])
    print('w_scale: ',w_scale)
    scale = min(h_scale, w_scale)
    shift = window[:2]  # y, x
    print('shift: ',shift)
    scales = np.array([scale, scale, scale, scale])
    print('scales: ',scales)
    shifts = np.array([shift[0], shift[1], shift[0], shift[1]])
    print('shifts: ',shifts)
    # Translate bounding boxes to image domain
    boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
    print('boxes: ',boxes.shape,' ',boxes)
    # Filter out detections with zero area. Often only happens in early
    # stages of training when the network weights are still a bit random.
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1)\
        if full_masks else np.empty((0,) + masks.shape[1:3])

    return boxes, class_ids, scores, full_masks

if __name__ == '__main__':
    tf.app.run()
