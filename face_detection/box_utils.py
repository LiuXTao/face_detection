import numpy as np
from PIL import Image

#  非最大值限制
def nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.

       Arguments:
           boxes: a float numpy array of shape [n, 5],
               where each row is (xmin, ymin, xmax, ymax, score).
           overlap_threshold: a float number.
           mode: 'union' or 'min'.

       Returns:
           list with indices of the selected boxes
       """
    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []
    picked_list = []
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]
    # grab the coordinates of the bounding boxes
    area = (x2 - x1 +1.0)*(y2 - y1 +1.0)
    ids = np.argsort(score)    # argsort 变换为递增顺序

    while len(ids)>0:
        # grab the max one
        last = len(ids) - 1
        largest = ids[last]
        picked_list.append(largest)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[largest], x1[ids[:last]])
        iy1 = np.maximum(y1[largest], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[largest], x2[ids[:last]])
        iy2 = np.minimum(y2[largest], y2[ids[:last]])

        # width 和 height corner
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersection's area
        inter_area = w * h
        if mode == 'min':
            overlap = inter_area / np.minimum(area[largest], area[ids[:last]])
        elif mode == 'union':
            overlap = inter_area / (area[largest] + area[ids[:last]] - inter_area)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )
    return picked_list


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

       Arguments:
           bboxes: a float numpy array of shape [n, 5].

       Returns:
           a float numpy array of shape [n, 5],
               squared bounding boxes.
    """
    square_boxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2-y1+1.0
    w = x2-x1+1.0
    max_side = np.maximum(h, w)
    square_boxes[:, 0] = x1 + w*0.5 - max_side*0.5
    square_boxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_boxes[:, 2] = square_boxes[:, 0] + max_side - 1.0
    square_boxes[:, 3] = square_boxes[:, 1] + max_side - 1.0
    return square_boxes

def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.

        Arguments:
            bboxes: a float numpy array of shape [n, 5].
            offsets: a float numpy array of shape [n, 4].

        Returns:
            a float numpy array of shape [n, 5].
        """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = np.expand_dims(x2 - x1 + 1.0, 1)
    h = np.expand_dims(y2 - y1 + 1.0, 1)
    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this
    return bboxes

def get_image_boxes(bounding_box, img, size=24):
    """Cut out boxes from the image.

       Arguments:
           bounding_boxes: a float numpy array of shape [n, 5].
           img: an instance of PIL.Image.
           size: an integer, size of cutouts.

       Returns:
           a float numpy array of shape [n, 3, size, size].
       """
    num_boxes = len(bounding_box)
    width, height = img.size
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bounding_box, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')
        img_array = np.asarray(img, 'uint8')

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i]+1), :] = img_array[y[i]:(ey[i] + 1), x[i]:(ex[i]+1), :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, : , : , :] = _preprocess(img_box)

    return img_boxes





def correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
       with respect to cutouts.

       Arguments:
           bboxes: a float numpy array of shape [n, 5],
               where each row is (xmin, ymin, xmax, ymax, score).
           width: a float number.
           height: a float number.

       Returns:
           dy, dx, edy, edx: a int numpy arrays of shape [n],
               coordinates of the boxes with respect to the cutouts.
           y, x, ey, ex: a int numpy arrays of shape [n],
               corrected ymin, xmin, ymax, xmax.
           h, w: a int numpy arrays of shape [n],
               just heights and widths of boxes.

           in the following order:
               [dy, edy, dx, edx, y, ey, x, ex, w, h].
       """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 +1.0
    h = y2 - y1 +1.0
    num_boxes = bboxes.shape[0]
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.

    dx, dy, edx, edy = np.zeros((num_boxes, )), np.zeros((num_boxes, )), w.copy() - 1.0, h.copy() - 1.0

    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list

def _preprocess(img):
    """Preprocessing step before feeding the network.

        Arguments:
            img: a float numpy array of shape [h, w, c].

        Returns:
            a float numpy array of shape [1, c, h, w].
        """
    '''
        np.expand_dims(array, axis)
        
        axis = 0 , 1, 2, 3
        表示在第axis维增加 一个维度
        
        np.argmax：返回沿轴最大值的索引值
        
        
    '''
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = (img - 127.5) * 0.0078125
    return img

