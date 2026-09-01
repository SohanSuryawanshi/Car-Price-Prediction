"""
Custom sklearn-compatible transformers.

Every transformer here follows the fit/transform contract, so each one only
learns parameters from whatever data is passed to .fit() which, inside a
Pipeline, will always be the current training fold. This is what keeps the
whole pipeline leakage-safe without any manual bookkeeping.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

class DomainSanityClipper(BaseEstimator, TransformerMixin):
    """Clips year and odometer to fixed, plausible real-world ranges.

    These bounds are business rules, not statistics learned from data, so
    applying them identically to train and test isn't leakage.
    """

    def __init__(self, year_min=config.YEAR_MIN,
                 year_max_offset=config.YEAR_MAX_OFFSET,
                 odo_min=config.ODOMETER_MIN, odo_max=config.ODOMETER_MAX):

        self.year_min = year_min
        self.year_max_offset = year_max_offset
        self.odo_min = odo_min
        self.odo_max = odo_max

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=["year", "odometer", "lat", "long"]).copy()
        year_max = pd.Timestamp.now().year + self.year_max_offset
        X["year"] = X["year"].clip(self.year_min, year_max)
        X["odometer"] = X["odometer"].clip(self.odo_min, self.odo_max)

        return X


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clips numeric columns to [Q1 - factor*IQR, Q3 + factor*IQR].

    Bounds ARE learned from data, so this must be fit on train only —
    handled automatically as long as it lives inside a Pipeline.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1, q3 = X.quantile(0.25), X.quantile(0.75)

        iqr = q3-q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1)


class CarAgeEngineer(BaseEstimator, TransformerMixin):
    """Replaces 'year' with 'car_age' to avoid year/car_age collinearity.

    Input columns: [year, odometer, lat, long]
    Output columns: [car_age, odometer, lat, long]
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=["year", "odometer", "lat", "long"]).copy()
        current_year = pd.Timestamp.now().year
        X["car_age"] = current_year - X["year"]
        return X[["car_age", "odometer", "lat", "long"]]


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Keeps the top_n most frequent categories per column (learned on train),
    replaces everything else with 'other'. Prevents one-hot encoding from
    exploding on high-cardinality columns like `model`.
    """

    def __init__(self, top_n=config.RARE_CATEGORY_TOP_N):
        self.top_n = top_n

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.keep_categories_ = {
            col: X[col].value_counts().head(self.top_n).index.tolist()
            for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            keep = self.keep_categories_[col]
            X[col] = X[col].where(X[col].isin(keep), other="other")
        return X