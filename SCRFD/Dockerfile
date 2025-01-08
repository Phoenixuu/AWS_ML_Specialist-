FROM python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-dev \
    python3-pip \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install numpy opencv-python onnxruntime pybind11

# Cài đặt thư viện cho TensorRT
# RUN apt-get install -y libnvinfer-dev

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn vào container
COPY . /app

# Xây dựng dự án với CMake
RUN mkdir -p build && cd build && cmake .. && make

# Cấu hình cổng nếu cần (ví dụ API hoặc dịch vụ web)
EXPOSE 5000

# Chạy tệp Python chính
CMD ["python3", "main.py"]

