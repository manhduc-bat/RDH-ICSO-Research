import numpy as np
from src.rhombus_feature import extract_layer_features_and_errors
from src.pca_kmeans import PCA_KMeans_MultiHistogram
from src.icso_optimizer import ICSO_Optimizer

class RDH_TwoLayer_Engine:
    """
    Hệ thống Nhúng và Trích xuất 2 lớp (Two-layer Embedding & Extraction)
    dựa trên PCA-KMeans và ICSO kết hợp Dịch chuyển biểu đồ tần suất (HS).
    """
    def __init__(self, n_components: int = 2, n_clusters: int = 16):
        self.Kc = n_components
        self.m = n_clusters

    @staticmethod
    def _embed_single_layer(img: np.ndarray, rows, cols, x, x_hat, errors, labels,
                            optimal_points, payload_bits: np.ndarray):
        """
        Áp dụng công thức Dịch chuyển Biểu đồ (Phương trình 14 trong bài báo) cho 1 lớp.
        Sử dụng Location Map cho các pixel biên (0 hoặc 255) để chống tràn số (Overflow/Underflow).
        """
        stego = np.copy(img).astype(np.int32)
        N = len(errors)
        bit_len = len(payload_bits)
        bit_idx = 0
        overflow_map = {}  # Lưu vị trí các pixel có nguy cơ tràn biên [0, 255]

        for idx in range(N):
            k = labels[idx]
            pl, pr = optimal_points[k]
            e = int(errors[idx])
            e_new = e

            # Phương trình (14): Dịch chuyển biểu đồ tần suất đa điểm nhúng
            if pr is not None and e == pr:
                if bit_idx < bit_len:
                    e_new = e + int(payload_bits[bit_idx])
                    bit_idx += 1
            elif pl is not None and e == pl:
                if bit_idx < bit_len:
                    e_new = e - int(payload_bits[bit_idx])
                    bit_idx += 1
            elif pr is not None and e > pr:
                if bit_idx < bit_len:
                    e_new = e + 1
            elif pl is not None and e < pl:
                if bit_idx < bit_len:
                    e_new = e - 1

            x_new = int(x_hat[idx]) + e_new

            # Kiểm tra Overflow / Underflow (ngoài đoạn [0, 255])
            if x_new < 0 or x_new > 255:
                overflow_map[idx] = int(x[idx])
                stego[rows[idx], cols[idx]] = int(x[idx])
            else:
                stego[rows[idx], cols[idx]] = x_new

        return stego.astype(np.uint8), bit_idx, overflow_map

    @staticmethod
    def _extract_and_recover_single_layer(stego_img: np.ndarray, layer: int,
                                          optimal_points, target_bits_count: int, overflow_map: dict):
        """
        Trích xuất dữ liệu bí mật và khôi phục nguyên trạng pixel gốc (Phương trình 15, 16, 17).
        Vì các pixel dự đoán thuộc lớp kia không đổi, ta tính lại chính xác x_hat, F, PCA và nhãn K-means.
        """
        rows, cols, x_prime, x_hat, e_prime, F = extract_layer_features_and_errors(stego_img, layer=layer)
        pca_km = PCA_KMeans_MultiHistogram(n_components=2, n_clusters=len(optimal_points))
        _, labels, _, _ = pca_km.cluster_and_build_histograms(F, e_prime)

        recovered = np.copy(stego_img).astype(np.int32)
        extracted_bits = []
        N = len(e_prime)

        for idx in range(N):
            if idx in overflow_map:
                recovered[rows[idx], cols[idx]] = overflow_map[idx]
                continue

            k = labels[idx]
            pl, pr = optimal_points[k]
            ep = int(x_prime[idx]) - int(x_hat[idx])
            e_rec = ep

            if len(extracted_bits) < target_bits_count:
                if pr is not None and (ep == pr or ep == pr + 1):
                    extracted_bits.append(ep - pr)
                    e_rec = pr
                elif pl is not None and (ep == pl or ep == pl - 1):
                    extracted_bits.append(pl - ep)
                    e_rec = pl
                elif pr is not None and ep > pr + 1:
                    e_rec = ep - 1
                elif pl is not None and ep < pl - 1:
                    e_rec = ep + 1

            recovered[rows[idx], cols[idx]] = int(x_hat[idx]) + e_rec

        return recovered.astype(np.uint8), np.array(extracted_bits, dtype=np.int32)

    def run_full_pipeline(self, image: np.ndarray, total_payload: int = 10000):
        """
        Chạy toàn bộ quy trình 2 lớp:
        - Lớp 1 (If): Nhúng nửa đầu Payload -> Tạo ảnh trung gian I_temp
        - Lớp 2 (Is): Nhúng nửa sau Payload -> Tạo ảnh Stego I'
        - Trích xuất & Khôi phục ngược lại: Lớp 2 trước -> Lớp 1 sau -> Ảnh khôi phục I_rec.
        """
        half_payload = total_payload // 2
        np.random.seed(2026)
        secret_bits_layer1 = np.random.randint(0, 2, size=half_payload, dtype=np.int32)
        secret_bits_layer2 = np.random.randint(0, 2, size=total_payload - half_payload, dtype=np.int32)

        # ================= LỚP THỨ NHẤT (Tập If) =================
        rows1, cols1, x1, x_hat1, err1, F1 = extract_layer_features_and_errors(image, layer=1)
        pca_km1 = PCA_KMeans_MultiHistogram(n_components=self.Kc, n_clusters=self.m)
        Y1, labels1, sub_hists1, _ = pca_km1.cluster_and_build_histograms(F1, err1)

        icso1 = ICSO_Optimizer(m=self.m, pop_size=36, max_gen=120)
        res_icso_emp = icso1.optimize(sub_hists1, target_payload=half_payload, use_empirical=True, seed=42)
        # Chạy thêm chế độ khởi tạo ngẫu nhiên để vẽ biểu đồ so sánh tốc độ hội tụ (Hình 5 trong bài báo)
        res_icso_rand = icso1.optimize(sub_hists1, target_payload=half_payload, use_empirical=False, seed=99)

        img_layer1, emb_count1, ovf1 = self._embed_single_layer(
            image, rows1, cols1, x1, x_hat1, err1, labels1,
            res_icso_emp['optimal_points'], secret_bits_layer1
        )

        # ================= LỚP THỨ HAI (Tập Is) =================
        rows2, cols2, x2, x_hat2, err2, F2 = extract_layer_features_and_errors(img_layer1, layer=2)
        pca_km2 = PCA_KMeans_MultiHistogram(n_components=self.Kc, n_clusters=self.m)
        Y2, labels2, sub_hists2, _ = pca_km2.cluster_and_build_histograms(F2, err2)

        icso2 = ICSO_Optimizer(m=self.m, pop_size=36, max_gen=120)
        res_icso2 = icso2.optimize(sub_hists2, target_payload=len(secret_bits_layer2), use_empirical=True, seed=42)

        stego_img, emb_count2, ovf2 = self._embed_single_layer(
            img_layer1, rows2, cols2, x2, x_hat2, err2, labels2,
            res_icso2['optimal_points'], secret_bits_layer2
        )

        # ================= TRÍCH XUẤT & KHÔI PHỤC (Lớp 2 -> Lớp 1) =================
        # Lưu ý: Trong thực tế thông tin phụ trợ (tâm cụm/điểm nhúng) được lưu ở LSB 2 hàng/cột đầu.
        # Ở đây ta truyền trực tiếp thông số phụ trợ để kiểm chứng chính xác tính đảo ngược của thuật toán.
        rec_after_layer2, ext_bits2 = self._extract_and_recover_single_layer(
            stego_img, layer=2, optimal_points=res_icso2['optimal_points'],
            target_bits_count=emb_count2, overflow_map=ovf2
        )
        recovered_img, ext_bits1 = self._extract_and_recover_single_layer(
            rec_after_layer2, layer=1, optimal_points=res_icso_emp['optimal_points'],
            target_bits_count=emb_count1, overflow_map=ovf1
        )

        # Tính toán các chỉ số đánh giá chất lượng ảnh
        mse_layer1 = float(np.mean((image.astype(np.float64) - img_layer1.astype(np.float64)) ** 2))
        psnr_layer1 = 10.0 * np.log10((255.0 ** 2) / mse_layer1) if mse_layer1 > 0 else float('inf')

        mse_stego = float(np.mean((image.astype(np.float64) - stego_img.astype(np.float64)) ** 2))
        psnr_stego = 10.0 * np.log10((255.0 ** 2) / mse_stego) if mse_stego > 0 else float('inf')

        mse_recovery = float(np.mean((image.astype(np.float64) - recovered_img.astype(np.float64)) ** 2))
        bits_match_1 = bool(np.array_equal(secret_bits_layer1[:emb_count1], ext_bits1))
        bits_match_2 = bool(np.array_equal(secret_bits_layer2[:emb_count2], ext_bits2))

        # Tạo bản đồ phân cụm 2D để minh họa kết quả K-means trên không gian ảnh
        cluster_map_2d = np.zeros_like(image, dtype=np.int32)
        cluster_map_2d[rows1, cols1] = labels1 + 1
        cluster_map_2d[rows2, cols2] = labels2 + 1

        return {
            'stego_img': stego_img,
            'recovered_img': recovered_img,
            'cluster_map_2d': cluster_map_2d,
            'pca_variance_ratio': pca_km1.explained_variance_ratio_,
            'sub_histograms_layer1': sub_hists1,
            'icso_layer1_emp': res_icso_emp,
            'icso_layer1_rand': res_icso_rand,
            'embedded_bits_total': emb_count1 + emb_count2,
            'psnr_layer1': psnr_layer1,
            'psnr_stego': psnr_stego,
            'mse_stego': mse_stego,
            'mse_recovery': mse_recovery,
            'bit_recovery_100_percent': (bits_match_1 and bits_match_2)
        }