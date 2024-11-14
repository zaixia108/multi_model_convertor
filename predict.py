# reference: Yolov4.cpp
import cv2
import numpy as np
import onnxruntime as rt
import random
import configparser
import json
import argparse


class box:
    def __init__(self, x=None, y=None, w=None, h=None, obj_prob=None, class_id=None, class_prob=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.obj_prob = obj_prob
        self.class_id = class_id
        self.class_prob = class_prob


def preprocess(img, width, height):
    flt_img = cv2.resize(img, (width, height))
    flt_img = np.float32(flt_img) * 1.0 / 255
    # HWC TO CHW
    result = flt_img.transpose(2, 0, 1)

    return result


def postprocess(outputs,
                anchors,
                width, height,
                src_width, src_height,
                n_classes,
                obj_threshold=0.7, iou_threshold=0.213):
    bboxes = []

    for output in outputs:

        bbox = []

        n_feature_1 = width // 8 * height // 8 * 3
        n_feature_2 = width // 16 * height // 16 * 3
        n_feature_3 = width // 32 * height // 32 * 3

        for features, anchor, side in zip((output[:, :n_feature_1, :],
                                           output[:, n_feature_1:(n_feature_1 + n_feature_2), :],
                                           output[:, (n_feature_1 + n_feature_2):, :]),
                                          (anchors[:6],
                                           anchors[6:12],
                                           anchors[12:]),
                                          (width // 8,
                                           width // 16,
                                           width // 32)):
            bbox += parse_yolo(features,
                               width, height,
                               src_width, src_height,
                               anchor, side, n_classes,
                               obj_threshold)
            # NMS
            bbox = nms(bbox, iou_threshold)

        bboxes.append(bbox)

    return bboxes


def nms(bboxes, iou_threshold):
    bboxes = sorted(bboxes, key=lambda bbox: bbox.obj_prob * bbox.class_prob, reverse=True)

    for i in range(len(bboxes)):
        if bboxes[i].obj_prob == 0:
            continue
        for j in range(i + 1, len(bboxes)):
            if bboxes[i].class_id == bboxes[j].class_id:
                if iou(bboxes[i], bboxes[j]) > iou_threshold:
                    bboxes[j].obj_prob = 0

    bboxes = list(bbox for bbox in bboxes if bbox.obj_prob > 0)
    return bboxes


def iou(det_a, det_b):
    center_a = (det_a.x + .5 * det_a.w, det_a.y + .5 * det_a.h)
    center_b = (det_b.x + .5 * det_b.w, det_b.y + .5 * det_b.h)
    left_up = (min(det_a.x, det_b.x), min(det_a.y, det_b.y))
    right_down = (max(det_a.x + det_a.w, det_b.x + det_b.w),
                  max(det_a.y + det_a.h, det_b.y + det_b.h))
    distance_d = (center_a[0] - center_b[0]) * (center_a[0] - center_b[0]) + (center_a[1] - center_b[1]) * (
                center_a[1] - center_b[1])
    distance_c = (left_up[0] - right_down[0]) * (left_up[0] - right_down[0]) + (left_up[1] - right_down[1]) * (
                left_up[1] - right_down[1])
    inter_l = det_a.x if det_a.x > det_b.x else det_b.x
    inter_t = det_a.y if det_a.y > det_b.y else det_b.y
    inter_r = det_a.x + det_a.w if det_a.x + det_a.w < det_b.x + det_b.w else det_b.x + det_b.w
    inter_b = det_a.y + det_a.h if det_a.y + det_a.h < det_b.y + det_b.h else det_b.y + det_b.h
    if (inter_b < inter_t or inter_r < inter_l):
        return 0
    inter_area = (inter_b - inter_t) * (inter_r - inter_l)
    union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area
    if (union_area == 0):
        return 0
    else:
        return inter_area / union_area - distance_d / distance_c


def parse_yolo(features,
               width, height,
               src_width, src_height,
               anchor, side, n_classes,
               obj_threshold):
    num = 3
    bboxes = []
    features = features.reshape(num, side, side, n_classes + 5)
    for row, col, n in np.ndindex(side, side, num):
        values = features[n, row, col, :]
        x_, y_, w_, h_, object_probability = values[:5]
        class_probabilities = values[5:]

        class_id = np.argmax(class_probabilities)
        object_probability = sigmoid(object_probability)
        class_probability = sigmoid(class_probabilities[class_id])

        if object_probability * class_probability < obj_threshold:
            continue

        # print(class_id, object_probability, class_probability)

        x = (sigmoid(x_) + col) / side * src_width
        y = (sigmoid(y_) + row) / side * src_height
        w = np.exp(w_) * anchor[2 * n] / width * src_width
        h = np.exp(h_) * anchor[2 * n + 1] / height * src_height

        bboxes.append(box(x=x - .5 * w,
                          y=y - .5 * h,
                          w=w,
                          h=h,
                          obj_prob=object_probability,
                          class_id=class_id,
                          class_prob=class_probability))

    return bboxes


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def draw_box(frame, bbox, thickness=2, color=(128, 128, 128), text=None):
    x_min = int(bbox.x)
    x_max = int(bbox.x + bbox.w)
    y_min = int(bbox.y)
    y_max = int(bbox.y + bbox.h)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
    if text:
        cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.75, color, 2)


def random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    a = 75
    rgba = [r, g, b, a]
    return tuple(rgba)


def generate_color_map(bboxes):
    unique_id = np.unique([bbox.class_id for bbox in bboxes])
    colors = [random_color() for _ in unique_id]

    return {id_: color_ for id_, color_ in zip(unique_id, colors)}


if __name__ == "__main__":

    # file paths
    """
    config_path = "./config"
    onnx_path = "../data/512-yolov4.onnx"
    sample_path = "../samples/person.jpg"
    names_path = "../coco.names"
    """
    parser = argparse.ArgumentParser(description="Inference on YOLO ONNX model")
    parser.add_argument('--config_path', type=str, default='config', help='config file')
    parser.add_argument('--onnx_path', type=str, default='yolov4.onnx', help='yolo onnx file')
    parser.add_argument('--sample_path', type=str, default='sample.jpg', help='image file')
    parser.add_argument('--names_path', type=str, default='coco.names', help='names file')
    parser.add_argument('--output_path', type=str, default=None, help='output image file')

    args = parser.parse_args()
    config_path = args.config_path
    onnx_path = args.onnx_path
    sample_path = args.sample_path
    names_path = args.names_path
    output_path = args.output_path

    config = configparser.ConfigParser()
    config.read(config_path)

    # yolo parameters in config file
    width = config.getint("yolo", "width")
    height = config.getint("yolo", "height")
    anchors = json.loads(config.get("yolo", "anchors"))
    n_classes = config.getint("yolo", "n_classes")
    obj_threshold = config.getfloat("yolo", "obj_threshold")
    iou_threshold = config.getfloat("yolo", "iou_threshold")

    with open(names_path, "r") as read_file:
        names = read_file.readlines()

    src_img = cv2.imread(sample_path)
    src_height, src_width, _ = src_img.shape
    img = preprocess(src_img, width, height)
    inputs = np.expand_dims(img, axis=0)

    # print("prep. shape: ", inputs.shape)
    # print("prep. type: ", inputs.dtype)

    sess = rt.InferenceSession(onnx_path)

    input_name = sess.get_inputs()[0].name
    # print("input name", input_name)
    # input_shape = sess.get_inputs()[0].shape
    # print("input shape", input_shape)
    # input_type = sess.get_inputs()[0].type
    # print("input type", input_type)

    output_name = sess.get_outputs()[0].name
    # print("output name", output_name)
    # output_shape = sess.get_outputs()[0].shape
    # print("output shape", output_shape)
    # output_type = sess.get_outputs()[0].type
    # print("output type", output_type)

    outputs = sess.run([output_name], {input_name: inputs})

    bboxes = postprocess(outputs,
                         anchors,
                         width, height,
                         src_width, src_height,
                         n_classes,
                         obj_threshold, iou_threshold)

    color_map = generate_color_map(bboxes[0])

    for bbox in bboxes[0]:
        name = names[bbox.class_id].replace("\n", "")
        draw_box(src_img, bbox, color=color_map[bbox.class_id],
                 text="{}-{:.2f}".format(name, bbox.obj_prob * bbox.class_prob))
    cv2.imshow("demo", src_img)
    if output_path:
        cv2.imwrite(output_path, src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()