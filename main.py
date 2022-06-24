import argparse
from pathlib import Path

from src.utils.loadData import loadFramesGenerator
from src.ROITracker import ROITracker
from src.FeatureMatcher import FeatureMatcher
from src.HistogramMatcher import HistMatcher
from src.GraphModel import GraphModel

# TODO Jakość kodu i raport (5/5)
# TODO Raport przejrzysty i wyczerpujący.
# TODO Kod przejrzysty i dobrze udokumentowany.

# TODO Skuteczność śledzenia 0.0 (0/5)
# TODO [0.00, 0.0] - 0.0
# TODO (0.0, 0.1) - 0.5
# TODO [0.1, 0.2) - 1.0
# TODO [0.2, 0.3) - 1.5
# TODO [0.3, 0.4) - 2.0
# TODO [0.4, 0.5) - 2.5
# TODO [0.5, 0.6) - 3.0
# TODO [0.6, 0.7) - 3.5
# TODO [0.7, 0.8) - 4.0
# TODO [0.8, 0.9) - 4.5
# TODO [0.9, 1.0) - 5.0

# stderr:
# Traceback (most recent call last):
#   File "main.py", line 4, in <module>
#     from src.utils.loadData import loadFramesGenerator
#   File "/home/janw/dydaktyka/2021_2022/lato/SI/projekt/evaluation/Kapron_Tomasz/src/utils/loadData.py", line 2, in <module>
#     from src.utils.PicBBox import BBox
#   File "/home/janw/dydaktyka/2021_2022/lato/SI/projekt/evaluation/Kapron_Tomasz/src/utils/PicBBox.py", line 28, in <module>
#     class PicBBox:
#   File "/home/janw/dydaktyka/2021_2022/lato/SI/projekt/evaluation/Kapron_Tomasz/src/utils/PicBBox.py", line 29, in PicBBox
#     def __init__(self, frame: np.ndarray, bboxList: list[BBox]):
# TypeError: 'type' object is not subscriptable

def processData():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    args = parser.parse_args()
    dataPath = Path(args.data_path)

    FM = FeatureMatcher()
    HM = HistMatcher()
    GM = GraphModel()

    dataGen = loadFramesGenerator(dataPath)
    tracker = ROITracker(FM, HM, GM)

    try:
        while frame := next(dataGen):
            tracker.update(frame)

    except StopIteration:
        exit(0)


if __name__ == '__main__':
    processData()
