import argparse
import numpy as np
from PIL import Image
import os
import tqdm
import cv2


EPS = 1E-5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Coins crop script by yolo markup.'
    )

    parser.add_argument('--images', type=str, required=True,
                        help='Path to folder with images.')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to folder with labels.')
    parser.add_argument('--result', type=str, required=True,
                        help='Path to folder with result cropped images')
    parser.add_argument('--merge-threshold', type=float, required=False,
                        default=0.2)

    return parser.parse_args()


def basefilename(name):
    spl = name.split('.')[:-1]
    res = ''
    for s in spl:
        res += s
    return res


def read_labels_as_boxes(labels_path, img_shape):
    with open(labels_path, 'r') as f:
        res = [
            {
                'label': line.split(' ')[0],
                'box': [
                    int(float(v)*img_shape[1 - i % 2])
                    for i, v in enumerate(line.split(' ')[1:])
                ]
            }
            for line in f
        ]

    return [
        {
            'label': box['label'],
            'box': [
                box['box'][0] - box['box'][2] // 2,
                box['box'][1] - box['box'][3] // 2,
                box['box'][0] - box['box'][2] // 2 + box['box'][2],
                box['box'][1] - box['box'][3] // 2 + box['box'][3]
            ]
        }
        for box in res
    ]


def dir_list(dir):
    dlst = os.listdir(dir)

    for d in dlst:
        if '.DS_Store' in d:
            del dlst[dlst.index(d)]

    return [
        os.path.join(dir, d)
        for d in dlst
    ]


def random_color():
    return (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255)
    )


def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=0)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=0)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def clustering_boxes(boxes, merge_threshold=0.2):
    res_clusters = [[boxes[0]]]
    for box in boxes[1:]:
        added = False
        for cluster in res_clusters:
            max_iou = 0
            for cl_box in cluster:
                between_iou = iou(
                    np.array(box['box']), np.array(cl_box['box'])
                )
                if between_iou > max_iou:
                    max_iou = between_iou
            if max_iou > merge_threshold:
                cluster.append(box)
                added = True
                break
        if not added:
            res_clusters.append([box])

    return res_clusters


def box_to_box(box):
    return [
        box[0],
        box[1],
        box[0] + box[2],
        box[1] + box[3]
    ]


def crop_cluster(img, cluster):
    cluster_points = np.array([
        p
        for box in cluster
        for p in [
            [box['box'][0], box['box'][1]],
            [box['box'][2], box['box'][3]]
        ]
    ])

    bound_box = box_to_box(cv2.boundingRect(cluster_points))

    crop_area = img[
                    bound_box[1]:bound_box[3],
                    bound_box[0]:bound_box[2]
                ].copy()

    boxes = [
        {
            'label': box['label'],
            'box': [p - bound_box[i % 2] for i, p in enumerate(box['box'])]
        }
        for box in cluster
    ]

    for box in boxes:
        for i, p in enumerate(box['box']):
            if p > crop_area.shape[1 - i % 2]:
                print('EEERRRORRRR!')
            assert not p > crop_area.shape[1 - i % 2]

    return crop_area, boxes


def save_boxes(boxes_path, boxes, image_shape):
    with open(boxes_path, 'w') as f:
        for box in boxes:
            x0, y0 = box['box'][:2]
            w, h = box['box'][2] - x0, box['box'][3] - y0
            box_line = str((x0 + w // 2) / image_shape[1]) + ' ' + \
                       str((y0 + h // 2) / image_shape[0]) + ' ' + \
                       str(
                            w / image_shape[1]
                       ) + ' ' + \
                       str(
                            h / image_shape[0]
                       )

            f.write('{} {}\n'.format(box['label'], box_line))


def main():
    args = parse_args()

    assert 0 <= args.merge_threshold <= 1

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    labels_pathes = dir_list(args.labels)

    total_index = 1

    for img_path in tqdm.tqdm(dir_list(args.images)):
        img_name = basefilename(os.path.basename(img_path))
        labels_path = None
        for labp in labels_pathes:
            if img_name == basefilename(os.path.basename(labp)):
                labels_path = labp
                break

        if labels_path is None:
            print(img_path)

        assert labels_path is not None

        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)

        bounding_boxes = read_labels_as_boxes(labels_path, img.shape)

        boxes_clusters = clustering_boxes(
            bounding_boxes,
            args.merge_threshold
        )

        for i in range(len(boxes_clusters)):
            crop_img, bbs = crop_cluster(img, boxes_clusters[i])
            save_boxes(
                os.path.join(args.result, '{}.txt'.format(total_index)),
                bbs,
                crop_img.shape
            )
            Image.fromarray(crop_img).save(
                os.path.join(args.result, '{}.jpg'.format(total_index))
            )

            total_index += 1

            # for box in bbs:
            #     crop_img = cv2.rectangle(
            #         crop_img, tuple(box['box'][:2]), tuple(box['box'][2:]),
            #         (0, 255, 0),
            #         5
            #     )
            #
            # window_name = 'Crop boxes'
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # cv2.imshow(window_name, crop_img)
            # cv2.waitKey(0)
            # cv2.destroyWindow(window_name)

        # for cls in boxes_clusters:
        #     color = random_color()
        #     for box in cls:
        #         img = cv2.rectangle(
        #             img, tuple(box['box'][:2]), tuple(box['box'][2:]),
        #             color,
        #             5
        #         )
        #
        # window_name = 'Boxes'
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(window_name, img)
        # cv2.waitKey(0)
        # cv2.destroyWindow(window_name)


if __name__ == '__main__':
    main()