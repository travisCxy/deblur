import cv2
import numpy as np

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255,255,0)]


def letterbox(img, new_shape=(1280, 1280), color=(255, 255, 255), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return  img, ratio, (dw, dh)

# 类别标签
# 2-square_cell
class SquareCellDetector(object):
    def __init__(self, triton_client):
        self.triton_client = triton_client
        self.CLASSES = ['square_cell']

    def detect(self, img, input_size=(1280, 1280), t=0.5,for_wrong_question=False):
        img_pad, ratio, pad = self.preprocess(img, input_size,for_wrong_question=for_wrong_question)
        outputs = self.triton_client.infer_square_cell_det(img_pad).copy()
        boxes, scores, clses = self.postprocess(outputs, ratio,pad)
        ret = {'boxes': boxes, 'scores': scores, 'clses': clses}
        # img = self.vis(img, boxes, scores, clses)
        # cv2.imwrite('show.jpg', img)
        return ret

    def postprocess(self, outputs, ratio,pad=(0,0)):
        boxes = []
        scores = []
        clses = []
        for i in range(len(outputs)):
            x1 = max(0, (outputs[i][1] - pad[0]) / ratio[0])
            y1 = max(0, (outputs[i][2] - pad[1]) / ratio[1])
            x2 = (outputs[i][3] - pad[0]) / ratio[0]
            y2 = (outputs[i][4] - pad[1]) / ratio[1]
            bbox = np.array([x1, y1, x2, y2])
            boxes.append(bbox)
            scores.append(outputs[i][6])
            clses.append(outputs[i][5])
        return boxes, scores, clses

    def preprocess1(self, img, input_size, swap=(2, 0, 1),for_wrong_question=False):
        # bgr 2 rgb
        img = img[:, :, ::-1]
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 255
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 255
        if img.shape[0]<img.shape[1] and for_wrong_question:
            new_w = img.shape[1]
            new_h = int(img.shape[0]*2)
            if img.shape[1]<1000:
                new_w = 1000
                #new_h = 2000
            new_img = np.ones((new_h, new_w, 3), dtype=np.uint8)*255
            new_img[:img.shape[0],:img.shape[1],:] = img
            img = new_img
        if max(img.shape[0],img.shape[1])<input_size[0]:
            r = min(1, 1)
            padded_img[: int(img.shape[0]), : int(img.shape[1])] = img
        else:
            r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
            resized_img = cv2.resize(
                img,
                (int(img.shape[1] * r), int(img.shape[0] * r)),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.uint8)
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = np.expand_dims(padded_img.transpose(swap), 0)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
        return padded_img, (r,r), (0, 0)


    def preprocess(self, img, input_size, swap=(2, 0, 1),for_wrong_question=False):
        img, ratio, pad = letterbox(img, input_size)
        img = img[:,:,::-1]
        h,w = img.shape[:2]
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 255
        padded_img[:h,:w,:] = img
        padded_img = np.expand_dims(padded_img.transpose(swap), 0)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
        return padded_img, ratio, pad

    def vis(self, img, boxes, scores, cls_ids, conf=0.25, class_names=None):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = colors[cls_id]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        return img


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    converter = trt.TrtGraphConverter(
        input_saved_model_dir="/mnt/data2/heping/servers/yyt_server_2.1/models/det/export",
        input_saved_model_tags=["tag"],
        input_saved_model_signature_key="model",
        minimum_segment_size=110,
        is_dynamic_op=True,
        maximum_cached_engines=20,
        use_calibration=False)
    converter.convert()
    converter.save("/mnt/data2/heping/servers/yyt_server_2.1/models/det/export_trt")

    ImgSplitter(0.4, "/share/hwhelper_yyt_new/models")
    print("done")

