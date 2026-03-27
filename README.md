# CSC4005 – Lab 0 Starter Kit
## Lab 0: Environment Setup & Smoke-Test Pipeline

## Mục tiêu
Lab 0 giúp sinh viên:
- thiết lập môi trường Python cho học phần CSC4005,
- cài đặt các thư viện cốt lõi,
- chạy thử một pipeline học sâu tối thiểu,
- sinh log, figure, checkpoint để chứng minh môi trường hoạt động,
- chuẩn bị nền tảng cho các lab tiếp theo.

## Yêu cầu phần mềm
- Python 3.10 hoặc 3.11
- pip
- Khuyến nghị: VS Code / PyCharm / Jupyter

## Bước 1. Tạo môi trường
### Cách 1: dùng venv
```bash
conda create -n csc4005-dl python=3.10 -y
```

Kích hoạt môi trường:

- Windows:
```bash
conda activate csc4005-dl
```

## Bước 2. Cài thư viện
```bash
pip install -r requirements.txt
```

## Bước 3. Kiểm tra môi trường
```bash
python check_env.py
```

## Bước 4. Chạy smoke-test pipeline
```bash
python run_smoke_test.py
```

## Output mong đợi
Sau khi chạy thành công, hệ thống sẽ sinh ra:
- `outputs/logs/env_check.txt`
- `outputs/logs/smoke_test_log.txt`
- `outputs/figures/loss_curve.png`
- `outputs/checkpoints/smoke_model.pt`

## Cách nộp bài
Sinh viên nộp lại toàn bộ project sau khi đã chạy xong, bao gồm:
- README đã cập nhật thông tin máy và kết quả chạy
- requirements.txt
- check_env.py
- run_smoke_test.py
- config
- toàn bộ thư mục outputs
- ảnh chụp màn hình terminal hoặc output thành công

## Sinh viên cần cập nhật README này
- Họ tên:
- MSSV:
- Lớp:
- Hệ điều hành:
- Python version:
- Torch version:
- Thiết bị chạy: CPU / GPU
- Ghi chú lỗi gặp phải khi setup (nếu có):
