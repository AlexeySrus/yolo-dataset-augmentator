import argparse
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from multiprocessing import Pool
import cv2


EPS = 1E-5


def parse_args():
    parser = argparse.ArgumentParser(
        description='Coins crop script by yolo markup.'
    )

    parser.add_argument('--input-folder', type=str, required=True,
                        help='Path to folder with images.')
    parser.add_argument('--backgrounds-folder', type=str, required=True,
                        help='Path to folder which contains backgounds images.')
    parser.add_argument('--result', type=str, required=True,
                        help='Path to folder with yolo dataset.')
    parser.add_argument('--dataset-size', type=int, default=10000,
                        required=False,
                        help='Size of generating dataset,')
    parser.add_argument('--min-samples-count', type=int, default=1,
                        required=False,
                        help='Minimum count objects on one image.')
    parser.add_argument('--max-samples-count', type=int, default=100,
                        required=False,
                        help='Maximum count objects on one image.')
    parser.add_argument('--num-processes', required=False, type=int,
                        default=1)

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


def dir_list(dir):
    dlst = os.listdir(dir)

    for d in dlst:
        if '.DS_Store' in d:
            del dlst[dlst.index(d)]

    return [
        os.path.join(dir, d)
        for d in dlst
    ]


def ring(size, thickness=-1):
    res = np.zeros(shape=(size, size, 3), dtype=np.uint8)
    radius = size // 2 - thickness // 2 if thickness > 0 else size // 2
    res = cv2.circle(res, (size // 2, size // 2), radius, (255, 0, 0),
                     thickness)
    return res[:, :, 0]


def upper_threshold_binarization(img, threshold=127):
    _img = img.copy()
    _img[img <= threshold] = 0
    _img[img > threshold] = 255
    return _img


def in_ellipse_square(img_shape, approx=5):
    el = cv2.ellipse2Poly(
        (img_shape[1] // 2, img_shape[0] // 2),
        (img_shape[1] // 2, img_shape[0] // 2),
        0, 0, 360, 5
    )
    return cv2.contourArea(el)


def coin_mask_generating(_inp_img):
    inp_img = cv2.cvtColor(_inp_img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(inp_img, (15,) * 2, 15)
    blured = (inp_img * 0.5 + blured * 0.5).astype(np.uint8)
    th_th, th_img = cv2.threshold(blured, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    m_step = upper_threshold_binarization(th_img, 10)
    m_step = upper_threshold_binarization(
        cv2.morphologyEx(m_step, cv2.MORPH_OPEN, ring(5)), 10)

    contours, _ = cv2.findContours(255 - m_step, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    img_el_sq = in_ellipse_square(inp_img.shape)
    contours.sort(key=lambda c: np.abs(cv2.contourArea(c) - img_el_sq))
    mask = np.zeros_like(inp_img)
    mask = cv2.drawContours(mask, [contours[0]], -1, (255), cv2.FILLED)

    return mask


def gen_sample(data):
    id = data['id']

    choosen_background = data['backgounds_pathes'][
        np.random.randint(0, len(data['backgounds_pathes']) - 1)
    ]

    choosen_background = np.array(
        Image.open(choosen_background).convert('RGB'),
        np.uint8
    )

    result_boxes = []

    samples_count = np.random.randint(data['min'], data['max'])
    for i in range(samples_count):
        choosen_img_path = data['images_pathes'][
            np.random.randint(0, len(data['images_pathes']))
        ]
        choosen_labels_path = None
        for lab_path in data['labels_pathes']:
            if basefilename(
                    os.path.basename(lab_path)
            ) == basefilename(os.path.basename(choosen_img_path)):
                choosen_labels_path = lab_path
                break

        assert choosen_labels_path is not None

        sample_image = np.array(
            Image.open(choosen_img_path).convert('RGB'), dtype=np.uint8
        )

        k = np.random.randint(20, 150) / 100

        sample_image = cv2.resize(sample_image, None, fx=k, fy=k)

        if choosen_background.shape[1] < sample_image.shape[1]:
            sample_image = cv2.resize(
                sample_image,
                None,
                fx=choosen_background.shape[1] / (sample_image.shape[1] + 1) / 2,
                fy=choosen_background.shape[1] / (sample_image.shape[1] + 1) / 2
            )

        if choosen_background.shape[0] < sample_image.shape[0]:
            sample_image = cv2.resize(
                sample_image,
                None,
                fx=choosen_background.shape[0] / (sample_image.shape[0] + 1) / 2,
                fy=choosen_background.shape[0] / (sample_image.shape[0] + 1) / 2
            )

        sample_boxes = read_labels_as_boxes(
            choosen_labels_path, sample_image.shape
        )

        gen_pos_x = np.random.randint(
            0, choosen_background.shape[1] - sample_image.shape[1] + 1
        )

        gen_pos_y = np.random.randint(
            0, choosen_background.shape[0] - sample_image.shape[0] + 1
        )

        if gen_pos_x < 0 or gen_pos_y < 0:
            print('AKA AKA {} {}'.format(gen_pos_x, gen_pos_y))

        for box in sample_boxes:
            result_boxes.append(
                {
                    'label': box['label'],
                    'box': [
                        gen_pos_x + box['box'][0],
                        gen_pos_y + box['box'][1],
                        gen_pos_x + box['box'][2],
                        gen_pos_y + box['box'][3],
                    ]
                }
            )

        coin_mask = coin_mask_generating(sample_image)

        choosen_background[
            gen_pos_y:gen_pos_y + sample_image.shape[0],
            gen_pos_x:gen_pos_x + sample_image.shape[1]
        ][coin_mask == 255] = sample_image.copy()[coin_mask == 255]

    Image.fromarray(choosen_background).save(os.path.join(
        data['save_images_path'],
        '{}.jpg'.format(id)
    ))

    save_boxes(
        os.path.join(
            data['save_labels_path'],
            '{}.txt'.format(id)
        ),
        result_boxes,
        choosen_background.shape
    )


def imap_unordered_bar(func, args, n_processes=8):
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def main():
    args = parse_args()

    images_pathes = [
        imgp
        for imgp in dir_list(args.input_folder) if '.jpg' in imgp
    ]

    labels_pathes = [
        labp
        for labp in dir_list(args.input_folder) if '.txt' in labp
    ]

    backgounds_pathes = dir_list(args.backgrounds_folder)

    data = {
        'images_pathes': images_pathes,
        'labels_pathes': labels_pathes,
        'backgounds_pathes': backgounds_pathes,
        'min': args.min_samples_count,
        'max': args.max_samples_count,
        'save_images_path': os.path.join(args.result, 'images/'),
        'save_labels_path': os.path.join(args.result, 'labels/')
    }

    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    if not os.path.isdir(os.path.join(args.result, 'images/')):
        os.makedirs(os.path.join(args.result, 'images/'))

    if not os.path.isdir(os.path.join(args.result, 'labels/')):
        os.makedirs(os.path.join(args.result, 'labels/'))

    input_data = []

    for i in range(args.dataset_size):
        data_sample = data.copy()
        data_sample['id'] = i + 1
        input_data.append(data_sample.copy())

    imap_unordered_bar(gen_sample, input_data, args.num_processes)


if __name__ == '__main__':
    main()