# Raspli (^-^), GNU AGPL-3.0 license

import cv2
from ultralytics import YOLO
from utils import isgetimage, highlight_boxes
from argparse import ArgumentParser, BooleanOptionalAction
import logging
from os.path import exists
from os import system

logging.basicConfig()
logger = logging.getLogger("RASPLI DETECTOR")
logger.setLevel(logging.INFO)

argparser = ArgumentParser()
argparser.add_argument('--model', nargs='?', default=None)
argparser.add_argument('--source', nargs='?', default=None)
argparser.add_argument('-save', action=BooleanOptionalAction, default=False)
args = argparser.parse_args()

if args.model == None:
    logger.info("Argument --model is undefined. Use lightweight pretrained model 'yolov8n-face.pt'")
    if not exists('weights'):
        system('mkdir weights')
    if not exists('weights/yolov8n-face.pt'):
        system('gdown 1jG8_C_P0SbnzYROZORe7CiJO6oMgp7eZ --output weights/yolov8n-face.pt')
    args.model = 'weights/yolov8n-face.pt'

if args.source == None:
    logger.info("Argument --source is undefined. Use example image 'futurists.jpg'")
    if not exists('sources'):
        system('mkdir sources')
    if not exists('sources/futurists.jpg'):
        system('gdown 1W-nerxVoH0C7-9psG_BMC0slMjPojLrH --output sources/futurists.jpg')
    args.source = 'sources/futurists.jpg'

if args.save == True and not exists('runs'):
    system('mkdir runs')


def detect_on_image(model: YOLO, image: object, save: bool):
    '''
    Face detection in the image
    with output and a strict option to save the result.
    '''

    boxes = model.predict(image)[0].boxes
    image_result = highlight_boxes(image, boxes)
    
    cv2.imshow("Raspli Detector", image_result)
    if save == True:
        cv2.imwrite('runs/output.jpg', image_result)
        logger.info("Result image saved to 'runs/output.jpg'")


def detect_on_video(model: YOLO, source: str, save: bool):
    '''
    Face detection on video or videostream
    with output and a strict option to save the result.
    '''

    cap = cv2.VideoCapture(source)

    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save == True:
        rec_fourcc = cv2.VideoWriter_fourcc(*'mp4v') # video codec init
        rec = cv2.VideoWriter('runs/output.mp4', rec_fourcc, fps, frame_size)

    while cap.isOpened():

        ret, frame = cap.read()
        # if frame is empty (capture completed) -> stop
        if frame is None:
            break
            
        boxes = model.predict(frame)[0].boxes
        frame_result = highlight_boxes(frame, boxes)

        cv2.imshow("Raspli Detector", frame_result)
        if save == True:
            rec.write(frame_result)

        # if key 'q' is pressed -> stop
        if cv2.waitKey() & 0xFF == ord('q'):
            break

        logger.debug(f"[VIDEO] FPS {fps} FRAME {cap.get(cv2.CAP_PROP_POS_FRAMES)} {frame_size[0]}x{frame_size[1]}")
        logger.debug(f"[VIDEO] FACES {len(boxes)}")

    cap.release()
    if save == True:
        rec.release()
        logger.info("Record saved to 'runs/output.mp4'")
    cv2.destroyAllWindows()


if __name__ == '__main__':

    model = YOLO(args.model)

    isgetimage_result = isgetimage(args.source)
    if isgetimage_result[0] == True:

        detect_on_image(model, isgetimage_result[1], args.save)
    else:
        detect_on_video(model, args.source, args.save)
