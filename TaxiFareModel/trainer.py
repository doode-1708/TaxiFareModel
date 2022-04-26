# imports
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)

        return rmse


if __name__ == "__main__":
    # get data
    df = get_data(nrows=100)
    print(f"Shape df: {df.shape}")

    # clean data
    df_clean = clean_data(df)
    print(f"Shape df_clean: {df_clean.shape}")

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    print(f"Shape X: {X.shape}, Shape y: {y.shape}")

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()

    # evaluate
    rsme = trainer.evaluate(X_val, y_val)
    print(f"rsme: {rsme}")
