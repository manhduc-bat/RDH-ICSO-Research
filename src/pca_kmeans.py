import numpy as np

class PCA_KMeans_MultiHistogram:
    """
    Thực hiện:
    1. Thuật toán PCA giảm chiều ma trận đặc trưng từ K=10 xuống Kc chiều.
    2. Thuật toán K-means với tâm khởi tạo cố định theo công thức bài báo.
    3. Sắp xếp m=16 biểu đồ tần suất con theo Entropy thông tin tăng dần.
    """
    def __init__(self, n_components: int = 2, n_clusters: int = 16, max_iter: int = 80, tol: float = 1e-4):
        self.Kc = n_components
        self.m = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.explained_variance_ratio_ = None
        self.corr_matrix_before_ = None

    def fit_transform_pca(self, F: np.ndarray) -> np.ndarray:
        """
        Thuật toán PCA tự cài đặt bằng Numpy:
        - Chuẩn hóa Z-score cho ma trận đặc trưng F (N x K)
        - Phân rã trị riêng ma trận hiệp phương sai
        - Chiếu xuống không gian Kc chiều tạo ma trận đặc trưng nén Y (N x Kc)
        """
        mean = np.mean(F, axis=0)
        std = np.std(F, axis=0) + 1e-8
        F_norm = (F - mean) / std

        N = F_norm.shape[0]
        # Ma trận hiệp phương sai / tương quan (K x K)
        cov_matrix = (F_norm.T @ F_norm) / (N - 1)
        self.corr_matrix_before_ = cov_matrix

        # Tính trị riêng (eigenvalues) và vector riêng (eigenvectors)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sắp xếp trị riêng theo thứ tự giảm dần
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Cố định dấu của vector riêng (để người gửi và người nhận luôn ra cùng ma trận Y)
        for j in range(eigenvectors.shape[1]):
            max_abs_idx = np.argmax(np.abs(eigenvectors[:, j]))
            if eigenvectors[max_abs_idx, j] < 0:
                eigenvectors[:, j] *= -1

        self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)

        # Ma trận chuyển đổi W (K x Kc) và ma trận đặc trưng nén Y (N x Kc)
        W = eigenvectors[:, :self.Kc]
        Y = F_norm @ W
        return Y

    @staticmethod
    def compute_entropy(errors_cluster: np.ndarray) -> float:
        """Tính Entropy thông tin của một biểu đồ tần suất con."""
        if len(errors_cluster) == 0:
            return 0.0
        _, counts = np.unique(errors_cluster, return_counts=True)
        probs = counts / len(errors_cluster)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    def cluster_and_build_histograms(self, F: np.ndarray, errors: np.ndarray):
        """
        Chạy PCA -> K-means -> Sắp xếp cụm theo Entropy tăng dần.
        """
        # Bước 1: Giảm chiều bằng PCA
        Y = self.fit_transform_pca(F)
        N = Y.shape[0]

        # Bước 2: Sắp xếp thứ tự mẫu theo thành phần chính thứ nhất (độ phức tạp tăng dần)
        # và khởi tạo m tâm cụm cố định theo đúng công thức: mu_k = Y[ceil(k * N / m)]
        order = np.argsort(Y[:, 0])
        Y_sorted = Y[order]

        init_indices = [min(int(np.ceil((k + 1) * N / self.m)) - 1, N - 1) for k in range(self.m)]
        centroids = Y_sorted[init_indices].astype(np.float64)

        # Bước 3: Lặp K-means
        labels = np.zeros(N, dtype=np.int32)
        for _ in range(self.max_iter):
            # Tính khoảng cách Euclid từ mọi điểm trong Y tới m tâm cụm
            dists = np.linalg.norm(Y[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            new_centroids = np.copy(centroids)
            for k in range(self.m):
                pts = Y[new_labels == k]
                if len(pts) > 0:
                    new_centroids[k] = np.mean(pts, axis=0)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            labels = new_labels
            if shift < self.tol:
                break

        # Bước 4: Tính Entropy cho từng cụm và sắp xếp lại nhãn từ 0..m-1 theo Entropy tăng dần
        # (giống Bảng 3 trong bài báo: H1 có Entropy nhỏ nhất -> H16 có Entropy lớn nhất)
        raw_entropies = []
        for k in range(self.m):
            ent = self.compute_entropy(errors[labels == k])
            raw_entropies.append(ent)

        sorted_cluster_order = np.argsort(raw_entropies)
        remap = {old_k: new_k for new_k, old_k in enumerate(sorted_cluster_order)}
        final_labels = np.vectorize(remap.get)(labels)

        # Xây dựng m biểu đồ con H_1 ... H_m
        sub_histograms = []
        sorted_entropies = []
        for k in range(self.m):
            cluster_err = errors[final_labels == k]
            ent = self.compute_entropy(cluster_err)
            sorted_entropies.append(ent)

            vals, counts = np.unique(cluster_err, return_counts=True)
            hist_dict = dict(zip(vals.tolist(), counts.tolist()))
            sub_histograms.append({
                'cluster_id': k + 1,
                'pixel_count': int(len(cluster_err)),
                'entropy': ent,
                'hist': hist_dict,
                'errors': cluster_err
            })

        return Y, final_labels, sub_histograms, sorted_entropies