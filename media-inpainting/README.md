# AI Media Inpainting

Ứng dụng chỉnh sửa ảnh và video để xóa các vùng không mong muốn (watermark, vật thể, chi tiết do AI tạo ra...) bằng kỹ thuật **inpainting**.

## Tính năng hiện tại

- **Image inpainting**: Upload ảnh, vẽ mask, chọn thuật toán, xem kết quả
- **Video inpainting**: Upload video, vẽ mask trên frame mẫu, xử lý từng frame
- **Thuật toán**: OpenCV Inpaint (Telea / Navier-Stokes) — chạy offline, không cần GPU
- **Kết quả**: Tự động lưu vào thư mục `outputs/`

## Cài đặt

```bash
cd media-inpainting
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt tại: http://localhost:7860

## Cách dùng

### Xóa vùng trên ảnh
1. Vào tab **Image Inpainting**
2. Upload ảnh
3. Dùng cọ đỏ vẽ lên vùng cần xóa
4. Chọn thuật toán (Telea thường cho kết quả tốt hơn)
5. Bấm **Inpaint Image**

### Xóa vùng trên video
1. Vào tab **Video Inpainting**
2. Upload video
3. Upload một frame đầu tiên và vẽ mask (hoặc dùng ảnh cùng kích thước với video)
4. Bấm **Inpaint Video**

## Lộ trình phát triển

- [x] Image inpainting với OpenCV
- [ ] Tích hợp LaMa inpainting cho kết quả chất lượng cao hơn
- [ ] Video inpainting với model chuyên dụng (ProPainter / E2FGVI)
- [ ] Tích hợp API Stability AI / OpenAI
- [ ] Hỗ trợ batch processing nhiều file

## Lưu ý

- OpenCV inpainting phù hợp với vùng nhỏ, đơn giản.
- Với vùng lớn hoặc phức tạp, cần dùng model AI như LaMa hoặc Stable Diffusion inpainting.
