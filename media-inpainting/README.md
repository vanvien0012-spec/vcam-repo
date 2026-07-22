# AI Media Inpainting

Ứng dụng chỉnh sửa ảnh và video để xóa các vùng không mong muốn (watermark, vật thể, chi tiết do AI tạo ra...) bằng kỹ thuật **inpainting**.

Hỗ trợ nhiều backend cho ảnh và video, từ thuật toán đơn giản chạy offline đến API AI chất lượng cao.

## Tính năng

### Ảnh
- **OpenCV Inpaint** (Telea / Navier-Stokes) — chạy offline, nhanh, phù hợp vùng nhỏ
- **LaMa** — model AI chạy local, chất lượng xóa vùng lớn tốt hơn OpenCV
- **Stability AI** — API cloud, chất lượng cao
- **OpenAI DALL-E** — API cloud, hiểu prompt tốt

### Video
- **OpenCV** — xử lý từng frame, đơn giản, có thể giật với vùng lớn
- **E2FGVI** — video inpainting model chuyên dụng, mượt hơn (yêu cầu cài đặt thủ công)

## Cài đặt

### Bắt buộc

```bash
cd media-inpainting
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Tùy chọn

Cài thêm backend tùy nhu cầu:

```bash
# LaMa (chạy local, chất lượng cao)
pip install simple-lama-inpainting

# OpenAI API
pip install openai

# E2FGVI (nâng cao, xem hướng dẫn riêng)
```

## Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt tại: http://localhost:7860

## Cách dùng

### Xóa vùng trên ảnh
1. Vào tab **Image Inpainting**
2. Chọn backend (OpenCV / LaMa / Stability AI / OpenAI)
3. Upload ảnh
4. Dùng cọ đỏ vẽ lên vùng cần xóa
5. Nhập API key nếu dùng Stability AI / OpenAI
6. Bấm **Inpaint Image**

### Xóa vùng trên video
1. Vào tab **Video Inpainting**
2. Upload video
3. Upload một frame đầu tiên và vẽ mask (mask sẽ áp dụng cho toàn bộ video)
4. Chọn backend và bấm **Inpaint Video**

## API Keys

- **Stability AI**: https://platform.stability.ai/
- **OpenAI**: https://platform.openai.com/

Nhập key trực tiếp trong giao diện ứng dụng.

## Lưu ý

- OpenCV phù hợp với vùng nhỏ, đơn giản.
- LaMa tự động tải model khi chạy lần đầu (khoảng vài trăm MB).
- Stability AI và OpenAI cần internet + API key có số dư.
- E2FGVI là placeholder trong code; cần cài đặt thủ công từ repo gốc: https://github.com/MCG-NJU/E2FGVI

## Lộ trình phát triển

- [x] Image inpainting với OpenCV
- [x] Tích hợp LaMa inpainting
- [x] Tích hợp Stability AI / OpenAI API
- [x] Video inpainting cơ bản với OpenCV
- [ ] Triển khai E2FGVI hoàn chỉnh
- [ ] Hỗ trợ batch processing nhiều file
- [ ] Giao diện vẽ mask chính xác hơn trên video
