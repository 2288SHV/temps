p3

import boto3
import sagemaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

session = sagemaker.Session()
bucket = session.default_bucket()
prefix = "linear-regression-demo"

print("Session ready")
print("Default S3 bucket:", bucket)




# Generate simple linear data
X = np.linspace(0, 10, 100)
y = 3 * X + 5 + np.random.randn(100) * 2   # y = 3x + 5 + noise

# Convert to DataFrame
df = pd.DataFrame({"X": X, "y": y})

df.head()


# Save CSV locally
df.to_csv("train.csv", index=False)

# Upload to S3
train_s3_path = session.upload_data("train.csv", bucket=bucket, key_prefix=prefix)

train_s3_path




# Prepare data
X_reshaped = X.reshape(-1, 1)

# Train model
model = LinearRegression()
model.fit(X_reshaped, y)

# Predict
y_pred = model.predict(X_reshaped)

print("Model trained!")
print("Slope (coef):", model.coef_[0])
print("Intercept:", model.intercept_)



mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)


plt.figure(figsize=(8, 5))
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.title("Linear Regression Model Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()



-------------------------------------------------------

p4

import pandas as pd
import numpy as np
import seaborn as sns

# Load titanic dataset
df_titanic = sns.load_dataset("titanic")

df_titanic.head()   # Just to verify loaded



df_titanic.head()



df_titanic.info()


df_titanic.describe()




df_titanic['age'] = df_titanic['age'].fillna(df_titanic['age'].median())
df_titanic['age'].isna().sum()   # Verify



df_titanic = df_titanic.drop(columns=['deck'])
df_titanic.head()  # Verify column removed





df_titanic['embarked'] = df_titanic['embarked'].fillna(df_titanic['embarked'].mode()[0])
df_titanic['embarked'].isna().sum()



df_titanic['sex'] = df_titanic['sex'].map({'male': 0, 'female': 1})
df_titanic.head()




df_titanic = pd.get_dummies(df_titanic, columns=['embarked', 'pclass'], drop_first=True)

df_titanic.head()



df_titanic['family_size'] = df_titanic['sibsp'] + df_titanic['parch'] + 1
df_titanic[['sibsp', 'parch', 'family_size']].head()



df_titanic.head()


--------------------------------------------------------------------------------

p5


# rf_train_hpo.py  (RandomForest + Hyperparameter Tuning)

import argparse
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)

    args = parser.parse_args()

    # Training data path
    train_path = "/opt/ml/input/data/train/train.csv"
    val_path = "/opt/ml/input/data/validation/validation.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Split
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    X_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, -1]

    # Model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print("Validation Accuracy:", acc)

    # Save model
    model_path = "/opt/ml/model/model.joblib"
    joblib.dump(model, model_path)




import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.tuner import HyperparameterTuner, IntegerParameter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()
prefix = "iris-hpo-v2"

print("Bucket:", bucket)




iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["label"] = iris.target

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)

train_s3 = session.upload_data("train.csv", bucket=bucket, key_prefix=prefix + "/train")
val_s3 = session.upload_data("validation.csv", bucket=bucket, key_prefix=prefix + "/validation")

print("Train:", train_s3)
print("Validation:", val_s3)



estimator = SKLearn(
    entry_point="rf_train_hpo.py",   # NEW FILE NAME
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.2-1"
)




hyperparameters = {
    "n_estimators": IntegerParameter(10, 200),
    "max_depth": IntegerParameter(2, 15),
}

objective_metric = "Validation Accuracy"

metric_definitions = [
    {
        "Name": "Validation Accuracy",
        "Regex": "Validation Accuracy: ([0-9\\.]+)"
    }
]






tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name=objective_metric,
    hyperparameter_ranges=hyperparameters,
    metric_definitions=metric_definitions,
    max_jobs=5,
    max_parallel_jobs=2
)





tuner.fit({
    "train": train_s3,
    "validation": val_s3
})





predictor = tuner.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)





test_sample = np.array([[5.6, 2.9, 4.3, 1.3]])

print("Prediction:", predictor.predict(test_sample))



session.delete_endpoint(predictor.endpoint_name)
print("Endpoint deleted successfully!")
















