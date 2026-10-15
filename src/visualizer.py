import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def print_presentation_tables(image_name: str, result: dict):
    """In các bảng kết quả trực quan ra Terminal để trình diễn trước hội đồng."""
    print("\n" + "=" * 88)
    print(f" BÁO CÁO KẾT QUẢ THỰC NGHIỆM TRÊN ẢNH: {image_name.upper()}")
    print("=" * 88)

    # 1. Bảng kết quả nén chiều bằng PCA
    var_ratios = result['pca_variance_ratio']
    pca_rows = [
        [f"Thành phần chính PC{i+1}", f"{var_ratios[i]*100:.2f}%", f"{np.sum(var_ratios[:i+1])*100:.2f}%"]
        for i in range(min(5, len(var_ratios)))
    ]
    print("\n[1] PHÂN TÍCH GIẢM CHIỀU ĐẶC TRƯNG BẰNG THUẬT TOÁN PCA (Từ K=10 đặc trưng):")
    print(tabulate(pca_rows, headers=["Thành phần", "Phương sai giữ lại", "Tích lũy"], tablefmt="fancy_grid"))

    # 2. Bảng 16 biểu đồ tần suất con (Tái hiện Bảng 3 trong bài báo)
    metrics = result['icso_layer1_emp']['cluster_metrics']
    table3_rows = []
    for m_item in metrics:
        table3_rows.append([
            f"H_{m_item['cluster_id']}",
            m_item['pixel_count'],
            f"{m_item['entropy']:.4f}",
            f"({m_item['p_lk']}, {m_item['p_rk']})",
            m_item['W_k'],
            f"{m_item['D_k']:.1f}"
        ])

    total_W1 = result['icso_layer1_emp']['total_W']
    total_D1 = result['icso_layer1_emp']['total_D']
    table3_rows.append([
        "TỔNG LỚP 1 (If)",
        sum(x['pixel_count'] for x in metrics),
        "Tăng dần (H1->H16)",
        "32 điểm tối ưu",
        total_W1,
        f"{total_D1:.1f}"
    ])

    print("\n[2] BẢNG THỐNG KÊ 16 BIỂU ĐỒ CON (K-MEANS + ICSO) - TƯƠNG ỨNG BẢNG 3 BÀI BÁO:")
    print(tabulate(
        table3_rows,
        headers=["Biểu đồ con", "Số Pixel", "Entropy", "Điểm nhúng (p_lk, p_rk)", "Dung lượng W_k (bits)", "Độ méo D_k"],
        tablefmt="fancy_grid"
    ))

    # 3. Bảng tổng hợp hiệu suất & kiểm chứng tính đảo ngược
    summary_rows = [
        ["Tổng dung lượng nhúng (2 lớp If + Is)", f"{result['embedded_bits_total']:,} bits"],
        ["Chất lượng ảnh sau Lớp 1 (If PSNR)", f"{result['psnr_layer1']:.2f} dB"],
        ["Chất lượng ảnh Stego cuối cùng (PSNR)", f"{result['psnr_stego']:.2f} dB (MSE = {result['mse_stego']:.4f})"],
        ["Thời gian tối ưu hóa ICSO (Lớp 1)", f"{result['icso_layer1_emp']['elapsed_sec']:.3f} giây"],
        ["Sai số khôi phục ảnh gốc MSE(I, I_rec)", f"{result['mse_recovery']:.6f} (Đảo ngược hoàn toàn = 0)"],
        ["Kiểm tra chuỗi bit bí mật trích xuất", "KHỚP 100% (BER = 0.00%)" if result['bit_recovery_100_percent'] else "LỖI"]
    ]
    print("\n[3] KIỂM CHỨNG CHẤT LƯỢNG ẢNH STEGO & TÍNH ĐẢO NGƯỢC HOÀN TOÀN (REVERSIBILITY):")
    print(tabulate(summary_rows, headers=["Chỉ số đánh giá", "Kết quả đạt được"], tablefmt="fancy_grid"))


def save_academic_plots(original_img: np.ndarray, image_name: str, result: dict, out_dir: str = "results/plots"):
    """Vẽ và lưu 3 biểu đồ học thuật phục vụ báo cáo NCKH."""
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]

    # --- HÌNH 1: So sánh đường cong biểu đồ tần suất con H1 (vùng mịn) -> H16 (vùng nhiễu) ---
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    sub_hists = result['sub_histograms_layer1']
    clusters_to_plot = [0, 1, 3, 7, 15]  # H1, H2, H4, H8, H16
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#7f7f7f']
    markers = ['o', 's', '^', 'd', 'x']

    x_vals = np.arange(-12, 13)
    for idx_c, col, mk in zip(clusters_to_plot, colors, markers):
        h_info = sub_hists[idx_c]
        y_vals = [h_info['hist'].get(int(v), 0) for v in x_vals]
        ax.plot(x_vals, y_vals, marker=mk, color=col, linewidth=1.8, markersize=4,
                label=f"H_{idx_c+1} (Entropy = {h_info['entropy']:.3f})")

    ax.set_title(f"Phân bố Sai số dự đoán trên các Biểu đồ con (PCA + K-means) - Ảnh {base_name}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Sai số dự đoán (Prediction Error e)", fontsize=10)
    ax.set_ylabel("Tần suất (Số lượng điểm ảnh)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_1_multi_histograms.png"))
    plt.close(fig)

    # --- HÌNH 2: Đường cong hội tụ của thuật toán ICSO (Có nhiễm sắc thể thực nghiệm vs Ngẫu nhiên) ---
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    hist_emp = result['icso_layer1_emp']['history_D']
    hist_rand = result['icso_layer1_rand']['history_D']
    gens = np.arange(1, len(hist_emp) + 1)

    ax.plot(gens, hist_emp, color='#d62728', linewidth=2.2, label="ICSO có nhiễm sắc thể thực nghiệm (Đề xuất)")
    ax.plot(gens, hist_rand, color='#1f77b4', linestyle='--', linewidth=1.8, label="ICSO khởi tạo ngẫu nhiên hoàn toàn")
    ax.set_title(f"Đường cong hội tụ Độ méo D theo số thế hệ G của thuật toán ICSO ({base_name})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Số thế hệ (Generation G)", fontsize=10)
    ax.set_ylabel("Tổng độ méo nhúng (Distortion D)", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_2_icso_convergence.png"))
    plt.close(fig)

    # --- HÌNH 3: Trực quan hóa Ảnh gốc - Bản đồ phân cụm K-means - Ảnh Stego - Ảnh khôi phục ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    axes[0].imshow(original_img, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(f"(a) Ảnh gốc ({base_name})", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    im1 = axes[1].imshow(result['cluster_map_2d'], cmap='turbo', vmin=1, vmax=16)
    axes[1].set_title("(b) Bản đồ 16 cụm (PCA + K-means)", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(result['stego_img'], cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"(c) Ảnh Stego (PSNR = {result['psnr_stego']:.2f} dB)", fontsize=10, fontweight='bold')
    axes[2].axis('off')

    diff_img = np.abs(original_img.astype(np.int32) - result['recovered_img'].astype(np.int32))
    axes[3].imshow(diff_img, cmap='hot', vmin=0, vmax=1)
    axes[3].set_title(f"(d) Sai số khôi phục |I - I_rec| (MSE={result['mse_recovery']:.1f})", fontsize=10, fontweight='bold')
    axes[3].axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{base_name}_3_visual_pipeline.png"))
    plt.close(fig)

    # Lưu bảng thống kê 16 cụm ra file CSV để tiện copy vào Word/Excel
    df_metrics = pd.DataFrame(result['icso_layer1_emp']['cluster_metrics'])
    df_metrics.to_csv(os.path.join("results", f"{base_name}_table3_metrics.csv"), index=False, encoding='utf-8-sig')