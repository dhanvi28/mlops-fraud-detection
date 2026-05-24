from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


class DataTransformation:

    def transform_data(self, df):

        X = df.drop("Class", axis=1)
        feature_names = X.columns.tolist()
        joblib.dump(
            feature_names,
            "artifacts/features.pkl"
        )
        

        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)

        X_test_scaled = scaler.transform(X_test)

        joblib.dump(
            scaler,
            "artifacts/scaler.pkl"
        )

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test
        )