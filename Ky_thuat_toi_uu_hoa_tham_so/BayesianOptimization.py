import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load data
iowa_file_path = 'train.csv'  # Đảm bảo rằng bạn có file 'train.csv' ở đúng đường dẫn
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Định nghĩa hàm mục tiêu để tối ưu hóa tham số
def objective(params):
    model = XGBRegressor(
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        n_estimators=params['n_estimators'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=1
    )
    model.fit(train_X, train_y)
    
    # Dự đoán và tính toán MAE
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    
    # Trả về giá trị MAE để tối ưu hóa (chúng ta cần giảm MAE)
    return {'loss': mae, 'status': STATUS_OK}

# Xác định không gian tìm kiếm cho các tham số
space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),  # Tốc độ học
    'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),  # Độ sâu của cây
    'n_estimators': hp.choice('n_estimators', [50, 100, 200]),  # Số cây trong rừng
    'subsample': hp.uniform('subsample', 0.6, 1.0),  # Tỷ lệ mẫu
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),  # Tỷ lệ đặc trưng sử dụng trong mỗi cây
}

# Khởi tạo một đối tượng Trials để lưu lại các kết quả của quá trình tìm kiếm
trials = Trials()

# Tối ưu hóa các tham số với Bayesian Optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

print("Best parameters found: ", best)

