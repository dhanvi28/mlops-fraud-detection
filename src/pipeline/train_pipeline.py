from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
df=DataIngestion().load_data(
    "data/creditcard.csv"
)
X_train,x_test,y_train,y_test=\
    DataTransformation().transform_data(df)
ModelTrainer().train(
    X_train,
    y_train,
    x_test,
    y_test
)