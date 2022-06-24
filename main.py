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
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 0.90) - 4.0
# TODO [0.90, 0.95) - 4.5
# TODO [0.95, 1.00) - 5.0

# stderr:
# python3: can't open file 'main.py': [Errno 2] No such file or directory

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
