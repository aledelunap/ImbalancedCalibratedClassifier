# Imbalanced Calibrated Classifiers

This repository contains a single class `ImbalancedCalibratedClassifiers` that improves probability estimations for minority class instances on imbalanced classification problems. Based on [Improving class probability estimates for imbalanced data by BC Wallace](https://link.springer.com/article/10.1007/s10115-013-0670-6) and `CalibratedClassifierCV`, the class creates bagging classifiers induced on balanced calibration datasets. The resulting probability is the average of the individual probabilities of all estimators.
