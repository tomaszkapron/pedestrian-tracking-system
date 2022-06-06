import cv2
from src.utils.BBox import BBox
from src.utils.PicBBox import PicBBox


def loadFrames(pathToDataFolder) -> dict:
    data = dict()
    with open(f'{pathToDataFolder}/bboxes.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            frame_name = line.strip()

            frame = cv2.imread(f'{pathToDataFolder}/frames/{frame_name}', cv2.IMREAD_GRAYSCALE)

            bbox_num = int(f.readline())
            bbox_list = []
            for i in range(bbox_num):
                bbox = f.readline()
                bbox_list.append(BBox(bbox))

            data[frame_name] = PicBBox(frame, bbox_list)

    return data


def main():
    data = loadFrames("../data")

    for key, frame in data.items():
        frame.printPicBBox()

    print(0)


if __name__ == '__main__':
    main()
