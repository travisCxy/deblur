import cv2
import numpy as np

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# 类别标签
# 0-picture 1-table 2-underline
class ImgDetector2:
    def __init__(self, triton_client):
        self.triton_client = triton_client
        self.CLASSES = ['picture', 'table', 'underline']

    def detect(self, img, input_size=(1280, 1280), t=0.25,for_wrong_question=False):
        img_pad, ratio = self.preprocess(img, input_size,for_wrong_question=for_wrong_question)
        outputs = self.triton_client.infer_det2(img_pad).copy()
        boxes, scores, clses = self.postprocess(outputs, ratio)
        ret = {'boxes': boxes, 'scores': scores, 'clses': clses}
        # img = self.vis(img, boxes, scores, clses)
        # cv2.imwrite('show.jpg', img)
        return ret

    def postprocess(self, outputs, ratio):
        boxes = []
        scores = []
        clses = []
        for i in range(len(outputs)):
            boxes.append(outputs[i][1:5] / ratio)
            scores.append(outputs[i][6])
            clses.append(outputs[i][5])
        return boxes, scores, clses

    def preprocess(self, img, input_size, swap=(2, 0, 1),for_wrong_question=False):
        # bgr 2 rgb
        img = img[:, :, ::-1]
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 255
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 255
        if img.shape[0]<img.shape[1] and for_wrong_question:
            new_img = np.ones((img.shape[1], img.shape[1], 3), dtype=np.uint8)*255
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
        return padded_img, r

    def vis(self, img, boxes, scores, cls_ids, conf=0.5, class_names=None):
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

