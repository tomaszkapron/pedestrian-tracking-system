import cv2
from src.utils.PicBBox import BBox
from src.utils.PicBBox import PicBBox


def loadFramesGenerator(pathToDataFolder):
    with open(f'{pathToDataFolder}/bboxes.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            frame_name = line.strip()

            frame = cv2.imread(f'{pathToDataFolder}/frames/{frame_name}', cv2.IMREAD_COLOR)

            bbox_num = int(f.readline())
            bbox_list = []
            for i in range(bbox_num):
                bbox = f.readline()
                bbox_list.append(BBox(bbox))

            yield PicBBox(frame, bbox_list)
