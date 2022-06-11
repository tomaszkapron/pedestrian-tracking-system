from src.utils.loadData import loadFrames
from src.ROITracker import ROITracker
from src.FeatureMatcher import FeatureMatcher
from src.HistogramMatcher import HistMatcher


def main():
    data = loadFrames("../data")
    FM = FeatureMatcher()
    HM = HistMatcher()
    firstPass = True

    for name, frame in data.items():
        if firstPass:
            tracker = ROITracker(frame, FM, HM)
            firstPass = False
            continue

        tracker.update(frame)


if __name__ == '__main__':
    main()

