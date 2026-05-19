from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
class ModelTrainer:
    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):
        model=XGBClassifier()
        with mlflow.start_run():
            model.fit(X_train, y_train)
            predictions=model.predict(X_test)
            accuracy=accuracy_score(
                y_test,
                predictions
            )
            print("Accuracy: ",accuracy)
            mlflow.log_metric(
                "accuracy",
                accuracy

            )
            joblib.dump(
                model,
                "artifacts/model.pkl"
            )
        