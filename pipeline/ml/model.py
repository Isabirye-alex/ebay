from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Dict, Any
from utils.timing import timeit
from utils.validators import ReusableFunctions
from utils.logging import setup_logger
import pandas as pd
import numpy as np


class PriceModel:
    """
    Regression model for predicting eBay item sale price (TotalPrice).

    Pipeline: Imputer → Scaler → GradientBoostingRegressor
    Evaluation: MAE, RMSE, R²
    """

    REQUIRED_COLUMNS: List = [
        "condition",
        "sold_date",
        "price",
        "category",
        "title",
        "seller_feedback_rating",
        "seller_feedback_count",
        "TotalPrice",
    ]

    DROP_COLUMNS: List = [
        "price",      # leakage — TotalPrice = price + shipping
        "shipping",   # leakage
        "title",      # raw text — not yet encoded
        "sold_date",  # raw datetime — encode separately if needed
    ]

    TARGET_COLUMN = "TotalPrice"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.df          = dataframe.copy()
        self.test_size   = test_size
        self.random_state = random_state
        self.pipe        = self._default_pipeline()
        self.metrics: Dict[str, Any] = {}
        self.logger      = setup_logger(self.__class__.__name__)

        ReusableFunctions.validate_schema(
            self.df, self.REQUIRED_COLUMNS, self.logger
        )

    def _default_pipeline(self) -> Pipeline:
        """
        Build default regression pipeline.

        Returns:
            Pipeline: Imputer → Scaler → GradientBoostingRegressor
        """
        return Pipeline([
            ("imputer",    SimpleImputer(strategy="median")),
            ("scaler",     StandardScaler()),
            ("regressor",  GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=self.random_state,
            )),
        ])

    def _prepare_features(self):
        """
        Separate feature matrix from target. Drop leakage and non-numeric columns.

        Returns:
            X (pd.DataFrame): Numeric feature matrix.
            y (pd.Series): Target prices.

        Raises:
            ValueError: If DataFrame is empty, target is missing, or
                no usable features remain.
        """
        if self.df.empty:
            raise ValueError("Input DataFrame is empty.")

        y = self.df[self.TARGET_COLUMN]

        if y.isnull().any():
            raise ValueError("Target column contains missing values.")

        cols_to_drop = [
            col for col in self.DROP_COLUMNS + [self.TARGET_COLUMN]
            if col in self.df.columns
        ]
        X = self.df.drop(columns=cols_to_drop)

        # Drop constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            self.logger.warning(f"Dropping constant columns: {constant_cols}")
            X = X.drop(columns=constant_cols)

        # Keep numeric only
        non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            self.logger.warning(f"Dropping non-numeric columns: {non_numeric}")
            X = X.select_dtypes(include=["number"])

        if X.shape[1] == 0:
            raise ValueError("No usable features remaining after preprocessing.")

        self.logger.info(f"Features: {X.columns.tolist()}")
        self.logger.info(f"Feature shape: {X.shape} | Target shape: {y.shape}")

        return X, y

    def _split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform train/test split.

        Note: stratify is NOT used — TotalPrice is continuous.

        Returns:
            X_train, X_test, y_train, y_test
        """
        self.logger.info(
            f"Splitting data — test_size={self.test_size}, "
            f"random_state={self.random_state}"
        )

        if len(X) != len(y):
            raise ValueError(
                f"Length mismatch: X={len(X)}, y={len(y)}"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            # No stratify — regression target is continuous
        )

        self.logger.info(
            f"Train size: {len(X_train)} | Test size: {len(X_test)}"
        )

        return X_train, X_test, y_train, y_test

    def _evaluate(self, y_test: pd.Series, preds: np.ndarray) -> None:
        """
        Compute and store regression evaluation metrics.

        Metrics:
            MAE  — average absolute dollar error
            RMSE — penalises large errors more than MAE
            R²   — proportion of price variance explained by the model

        Args:
            y_test (pd.Series): True prices.
            preds (np.ndarray): Predicted prices.
        """
        mae  = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2   = r2_score(y_test, preds)

        self.metrics["mae"]  = round(mae,  4)
        self.metrics["rmse"] = round(rmse, 4)
        self.metrics["r2"]   = round(r2,   4)

        self.logger.info(f"MAE:  ${mae:.2f}")
        self.logger.info(f"RMSE: ${rmse:.2f}")
        self.logger.info(f"R²:   {r2:.4f}")

        print(f"\n=== Regression Results ===")
        print(f"MAE:  ${mae:.2f}  (avg prediction is off by this many dollars)")
        print(f"RMSE: ${rmse:.2f} (penalises large errors more)")
        print(f"R²:   {r2:.4f}   (1.0 = perfect, 0.0 = learned nothing)")

    def cross_validate_model(self, n_splits: int = 5) -> pd.Series:
        """
        Run k-fold cross-validation.

        Uses KFold, not StratifiedKFold — regression has no class labels.

        Returns:
            pd.Series: Mean MAE, RMSE, R² across all folds.
        """
        if not self.metrics:
            raise ValueError("Model has not been trained. Call train() first.")

        X, y = self._prepare_features()

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_validate(
            self.pipe, X, y,
            cv=cv,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
        )

        results = pd.Series({
            "MAE":  -scores["test_neg_mean_absolute_error"].mean(),
            "RMSE": -scores["test_neg_root_mean_squared_error"].mean(),
            "R2":    scores["test_r2"].mean(),
        })

        self.logger.info(f"Cross-validation results: {results.to_dict()}")
        print("\nCross-Validation Results:")
        print(results)

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importances from the trained regressor.

        Returns:
            pd.DataFrame: Features ranked by importance, descending.
        """
        if not self.metrics:
            raise ValueError("Model has not been trained. Call train() first.")

        X, _ = self._prepare_features()
        regressor = self.pipe.named_steps["regressor"]

        importance_df = pd.DataFrame({
            "Feature":    X.columns.tolist(),
            "Importance": regressor.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        self.logger.info("Feature importances extracted.")
        print(importance_df)

        return importance_df

    @timeit
    def train(self) -> Dict[str, Any]:
        """
        Execute the full regression training pipeline.

        Steps:
            1. Prepare features
            2. Train/test split (no stratification)
            3. Fit pipeline
            4. Predict and evaluate

        Returns:
            Dict: MAE, RMSE, R² metrics.
        """
        self.logger.info("Starting regression training pipeline")

        X, y             = self._prepare_features()
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        self.logger.info("Fitting pipeline")
        self.pipe.fit(X_train, y_train)

        preds = self.pipe.predict(X_test) 
        self._evaluate(y_test, preds)

        self.get_feature_importance()

        self.cross_validate_model()

        self.logger.info("Training completed")
        return self.metrics