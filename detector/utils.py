# Raspli (^-^), GNU AGPL-3.0 license

import cv2
from typing import Union, Tuple


def isgetimage(source: str) -> Tuple[bool, Union[object, None]]:
    '''
    Returns (True, Image) if the source file is an image,
    otherwise (False, None).
    '''
    potential_image = cv2.imread(source)
    if not potential_image is None:
        return (True, potential_image)
    else:
        return (False, None)


def highlight_boxes(image: object, boxes: object) -> object:
    '''
    Changes the image according to the detected faces (boxes).
    '''

    for i, xyxy in enumerate(boxes.xyxy):
        # highlight boxes
        x1, y1, x2, y2 = xyxy.int().tolist()
        image = cv2.rectangle(
            image, (x1, y1), (x2, y2), (0,255,0),
            thickness=2
        )
        # highlight confidence text
        text, font_scale = "{:.2f}".format(boxes.conf[i].item()), 2
        image = cv2.rectangle(
            image, (x1, y1 - font_scale * 12), (x1 + len(text) * font_scale * 9, y1), (0,192,0),
            thickness=-1
        )
        image = cv2.putText(
            image, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, font_scale, (255,255,255),
            thickness=2
        )

    return image
