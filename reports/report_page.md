# CSC4005 Setup Report

## Thiết lập Môi trường
- Tạo conda environment `csc4005-dl` với Python 3.10: `conda create -n csc4005-dl python=3.10 -y`
- Cài đặt tất cả dependencies từ `requirements.txt` bằng `pip install -r requirements.txt`
- Cải thiện: Cài đặt `torchaudio` riêng từ PyTorch channel để tránh xung đột phiên bản

## Kiểm tra & Xác thực
- ✅ **check_env.py**: Kiểm tra environment thành công (NumPy 2.2.6, Pandas 2.3.3, PyTorch 2.11.0+CPU)
- ✅ **run_smoke_test.py**: Chạy 3 epochs training, độ chính xác tăng từ 52% → 74%, loss giảm liên tục

## Vấn đề & Giải pháp
- **Vấn đề 1**: Python 3.9 được kích hoạt thay vì 3.10 → **Giải pháp**: Dùng `conda run -n csc4005-dl` để chạy scripts
- **Vấn đề 2**: ModuleNotFoundError: 'torchaudio' → **Giải pháp**: `conda install -c pytorch torchaudio -y`

## Kết luận
Environment đã setup hoàn toàn. Pipeline hoạt động bình thường với loss curve và checkpoint model được lưu lại.
