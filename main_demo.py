import os
import cv2
import numpy as np
from tabulate import tabulate
from src.rdh_engine import RDH_TwoLayer_Engine
from src.visualizer import print_presentation_tables, save_academic_plots

def create_fallback_test_image(filepath: str):
    """Tạo ảnh test mẫu 512x512 (gồm vùng trơn, vùng chuyển sắc và vùng nhiễu phức tạp) nếu thư mục ảnh trống."""
    H, W = 512, 512
    y, x = np.ogrid[:H, :W]
    # Tạo nền chuyển sắc mượt (vùng mịn -> sẽ rơi vào H1, H2)
    img = np.clip(90 + 40 * np.sin(x / 60.0) + 30 * np.cos(y / 60.0), 20, 235).astype(np.float64)

    # Tạo các khối hình học ở giữa
    mask_circle = (x - 256) ** 2 + (y - 256) ** 2 < 130 ** 2
    img[mask_circle] = 160

    # Tạo vùng kết cấu phức tạp như lông thú (sẽ rơi vào H12..H16)
    np.random.seed(42)
    texture_region = (x > 320) & (y > 320)
    img[texture_region] += np.random.normal(0, 18, size=np.sum(texture_region))

    img = np.clip(img, 10, 245).astype(np.uint8)
    cv2.imwrite(filepath, img)

def main():
    input_dir = os.path.join("data", "input_images")
    stego_dir = os.path.join("results", "stego_images")
    rec_dir = os.path.join("results", "recovered_images")
    plots_dir = os.path.join("results", "plots")

    for d in [input_dir, stego_dir, rec_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    # Quét tất cả ảnh trong thư mục data/input_images/
    valid_exts = ('.bmp', '.png', '.tif', '.tiff', '.jpg', '.jpeg')
    image_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(valid_exts)]

    if not image_files:
        sample_path = os.path.join(input_dir, "Synthetic_Lena_512.bmp")
        print("[!] Thư mục 'data/input_images/' đang trống. Đang tạo ảnh test mẫu 512x512 để chạy demo...")
        create_fallback_test_image(sample_path)
        image_files = ["Synthetic_Lena_512.bmp"]

    # Khởi tạo Engine với Kc = 2 chiều PCA và m = 16 cụm K-means như bài báo
    engine = RDH_TwoLayer_Engine(n_components=2, n_clusters=16)
    target_payload = 10000  # Nhúng 10.000 bits (5.000 bits Lớp 1 + 5.000 bits Lớp 2)

    all_summary = []

    for img_name in image_files:
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Chuẩn hóa kích thước về 512x512 nếu ảnh khác kích thước chuẩn
        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

        print(f"\n>>> Đang xử lý ảnh '{img_name}' (512x512) với tải trọng {target_payload:,} bits...")
        res = engine.run_full_pipeline(img, total_payload=target_payload)

        # In bảng phân tích chi tiết ra màn hình
        print_presentation_tables(img_name, res)

        # Lưu ảnh Stego, ảnh khôi phục và các đồ thị
        base = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(stego_dir, f"{base}_stego.png"), res['stego_img'])
        cv2.imwrite(os.path.join(rec_dir, f"{base}_recovered.png"), res['recovered_img'])
        save_academic_plots(img, img_name, res, out_dir=plots_dir)

        all_summary.append([
            img_name,
            f"{res['embedded_bits_total']:,}",
            f"{res['psnr_layer1']:.2f} dB",
            f"{res['psnr_stego']:.2f} dB",
            f"{res['icso_layer1_emp']['elapsed_sec']:.2f}s",
            "MSE = 0.0 ( Đạt )" if res['mse_recovery'] == 0 else f"MSE = {res['mse_recovery']}"
        ])

    if len(all_summary) > 1:
        print("\n" + "=" * 88)
        print(" BẢNG TỔNG HỢP TRÊN TOÀN BỘ TẬP ẢNH TEST (TƯƠNG ỨNG BẢNG 4 TRONG BÀI BÁO)")
        print("=" * 88)
        print(tabulate(
            all_summary,
            headers=["Tên ảnh", "Payload (bits)", "PSNR Lớp 1", "PSNR Ảnh Stego", "Thời gian ICSO", "Khôi phục không mất dữ liệu"],
            tablefmt="fancy_grid"
        ))

    print(f"\n[HOÀN TẤT] Toàn bộ ảnh Stego, file CSV và 3 biểu đồ học thuật đã được lưu tại thư mục 'results/'!")

if __name__ == "__main__":
    main()