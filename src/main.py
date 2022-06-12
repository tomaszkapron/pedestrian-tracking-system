import argparse
from pathlib import Path

from src.utils.loadData import loadFrames, loadFramesGenerator
from src.ROITracker import ROITracker
from src.FeatureMatcher import FeatureMatcher
from src.HistogramMatcher import HistMatcher
from src.GraphModel import GraphModel


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
