import cv2
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torchvision.transforms.functional as TFF
from PIL import Image

def read_image(pt, size, is_image=True):
    img = Image.open(pt)
    H,W, = np.array(img).shape[:2]
    img2 = TFF.center_crop(img, min(H,W))
    interp_mode = Image.BILINEAR if is_image else Image.NEAREST
    img3 = TFF.resize(img2, size, interpolation=interp_mode)
#     return img3
    arr = np.array(img3)
    return arr

def show_bboxes(arr):
    arr = (arr > 0) * 1
    arr = arr.astype(np.uint8)

    cnts = cv2.findContours(arr.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    image = 255-arr.copy()
    image = np.stack([image]*3, axis=-1)
    for c in list(filter(lambda _x: len(_x) > 2, cnts)):
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)
            bbox = cv2.boundingRect(c)
            point1 = (int(bbox[0]), int(bbox[1]))
            point2 = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            cv2.rectangle(image, point1, point2, (0,0,255))
            # cv2.putText(image, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

def main():
    source_fldr = Path('./images2/attribute_512p')
    target_fldr = Path('./images2/show_bboxes')
    target_fldr.mkdir(exist_ok=False, parents=True)
    failed_counter = 0
    for pt in tqdm(list(source_fldr.iterdir())):
        try:
            arr = np.array(Image.open(pt))
            # arr = read_image(pt, 384, is_image=False)
            image = show_bboxes(arr)
            Image.fromarray(image).save(target_fldr / pt.name)
        except:
            failed_counter += 1
            with open('failed.txt', 'a') as g:
                g.write(pt.name + '\n')
    with open('failed.txt', 'a') as g:
        g.write(f'Total failed: {failed_counter}')

    # source_fldr = Path('./images2/attribute_512p')
    # pt = source_fldr / 'ISIC_0012511_attribute_globules.png'
    # arr = read_image(pt, 384, is_image=False)
    # image = show_bboxes(arr)
    # Image.fromarray(image).save('tmp.png')


if __name__ == '__main__':
    main()