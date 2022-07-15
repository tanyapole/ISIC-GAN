import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image

def get_points(bbox):
    point1 = (int(bbox[0]), int(bbox[1]))
    point2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
    return point1, point2    

def build_bboxes(arr):
    arr = (arr > 0) * 1
    arr = arr.astype(np.uint8)

    cnts = cv2.findContours(arr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    bboxes = []
    for c in list(filter(lambda _x: len(_x) > 2, cnts)):
        M = cv2.moments(c)
        if M["m00"] > 0:
            bboxes.append(cv2.boundingRect(c))
            
    res = np.zeros_like(arr)
    for b in bboxes:
        p1, p2 = get_points(b)
        res[p1[1]:p2[1], p1[0]:p2[0]] = 1
    return res

def process_folder(source_fldr, target_fldr):
    target_fldr.mkdir(exist_ok=False, parents=True)
    failed_counter = 0
    for pt in tqdm(list(source_fldr.iterdir())):
        try:
            arr = np.array(Image.open(pt))
            image = build_bboxes(arr)
            Image.fromarray(image).save(target_fldr / pt.name)
        except:
            failed_counter += 1
            with open('failed.txt', 'a') as g:
                g.write(pt.name + '\n')
    with open(f'failed_{source_fldr.name}_{target_fldr.name}.txt', 'a') as g:
        g.write(f'Total failed: {failed_counter}')

def main():
    process_folder(Path('./images2/attribute_512p'), Path('./images2/bboxes_attribute_512p'))
    process_folder(Path('./images2/seg_512p'), Path('./images2/bboxes_seg_512p'))

    # source_fldr = Path('./images2/attribute_512p')
    # pt = source_fldr / 'ISIC_0012511_attribute_globules.png'
    # arr = read_image(pt, 384, is_image=False)
    # image = show_bboxes(arr)
    # Image.fromarray(image).save('tmp.png')


if __name__ == '__main__':
    main()