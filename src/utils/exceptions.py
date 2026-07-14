class CattleWeightError(Exception):
    pass


class SegmentationError(CattleWeightError):
    pass


class CalibrationError(CattleWeightError):
    pass


class PredictionError(CattleWeightError):
    pass
