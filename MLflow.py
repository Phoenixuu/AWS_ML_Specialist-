import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# run
with mlflow.start_run():
    # Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)


    # Log tham số, kết quả và mô hình
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")


# #Cai dat 
# pip install mlflow
# #Chay chuong trinh
# mlflow ui
