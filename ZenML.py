#zenml tập trung vào việc xây dựng, quản lý và triển khai các pipeline, là công cụ hỗ trợ quản lý workflow. Quản lý và tự động hóa toàn bộ quy trình pipeline, bao gồm tích hợp với test metrix, phù hợp với quản lý

# Import các thư viện
import zenml
from zenml.pipelines import pipeline
from zenml.steps import step
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Định nghĩa một bước trong pipeline: Huấn luyện mô hình
@step
def data_preprocessing():
    # Tải dữ liệu Iris
    data = load_iris()
    X = data.data
    y = data.target
    
    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

@step
def train_model(X_train, y_train):
    # Khởi tạo mô hình RandomForest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Huấn luyện mô hình
    model.fit(X_train, y_train)
    
    return model

@step
def evaluate_model(model, X_test, y_test):
    # Dự đoán kết quả từ mô hình
    y_pred = model.predict(X_test)
    
    # Tính độ chính xác của mô hình
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    
    return accuracy

# Tạo một pipeline
@pipeline
def model_pipeline():
    X_train, X_test, y_train, y_test = data_preprocessing()
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

# Chạy pipeline
pipeline_instance = model_pipeline()
pipeline_instance.run()

