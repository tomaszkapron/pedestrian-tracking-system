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

# TODO Skuteczność śledzenia 0.457 (2.5/5)
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

# TODO Brakuje jednej linijki na wyjściu.

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
