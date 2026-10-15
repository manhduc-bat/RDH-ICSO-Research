import numpy as np
import time

class ICSO_Optimizer:
    """
    Thuật toán tối ưu hóa Crisscross cải tiến (ICSO) tìm 2m điểm nhúng tối ưu
    (p_l1, p_r1, ..., p_lm, p_rm) cho m biểu đồ tần suất con.
    - Tập ứng viên đỉnh phải p_rk: {0, 1, 2, 3, 4, 5, 6, 7, +inf} (9 lựa chọn, chỉ số 0..8)
    - Tập ứng viên đỉnh trái p_lk: {-1, -2, -3, -4, -5, -6, -7, -8, -inf} (9 lựa chọn, chỉ số 0..8)
    Trong đó chỉ số 8 đại diện cho +/-inf (tức là KHÔNG nhúng tin vào phía đó của biểu đồ con).
    """
    LEFT_CANDIDATES = [-1, -2, -3, -4, -5, -6, -7, -8, None]   # index 0..8 (8 -> -inf)
    RIGHT_CANDIDATES = [0, 1, 2, 3, 4, 5, 6, 7, None]          # index 0..8 (8 -> +inf)

    def __init__(self, m: int = 16, pop_size: int = 36, max_gen: int = 120,
                 pc: float = 0.85, pv: float = 0.60, pm: float = 0.12):
        self.m = m
        self.dim = 2 * m  # 32 chiều cho 16 biểu đồ con
        self.M = pop_size
        self.G_max = max_gen
        self.pc = pc      # Xác suất giao phối ngang
        self.pv = pv      # Xác suất giao phối dọc
        self.pm = pm      # Xác suất đột biến

    def _precompute_lookup_tables(self, sub_histograms):
        """
        Tiền tính toán ma trận Dung lượng W và Độ méo D cho từng ứng viên (0..8)
        trên mỗi biểu đồ con để tăng tốc độ đánh giá Fitness lên hàng nghìn lần.
        """
        W_left = np.zeros((self.m, 9), dtype=np.float64)
        D_left = np.zeros((self.m, 9), dtype=np.float64)
        W_right = np.zeros((self.m, 9), dtype=np.float64)
        D_right = np.zeros((self.m, 9), dtype=np.float64)

        for k, h_info in enumerate(sub_histograms):
            errs = h_info['errors']
            if len(errs) == 0:
                continue

            # Tính cho 8 ứng viên bên trái: p_lk in {-1, ..., -8} (index 8 là -inf có W=0, D=0)
            for idx in range(8):
                pl = self.LEFT_CANDIDATES[idx]
                w_val = np.sum(errs == pl)
                shift_val = np.sum(errs < pl)
                W_left[k, idx] = w_val
                D_left[k, idx] = 0.5 * w_val + shift_val

            # Tính cho 8 ứng viên bên phải: p_rk in {0, ..., 7} (index 8 là +inf có W=0, D=0)
            for idx in range(8):
                pr = self.RIGHT_CANDIDATES[idx]
                w_val = np.sum(errs == pr)
                shift_val = np.sum(errs > pr)
                W_right[k, idx] = w_val
                D_right[k, idx] = 0.5 * w_val + shift_val

        return W_left, D_left, W_right, D_right

    def evaluate_population(self, pop: np.ndarray, W_left, D_left, W_right, D_right, target_payload: int):
        """
        Bước 6: Đánh giá độ phù hợp (Fitness) cho tất cả nhiễm sắc thể.
        Mục tiêu: Tối thiểu hóa tổng độ méo D sao cho Tổng dung lượng W >= target_payload.
        """
        pop_len = pop.shape[0]
        left_genes = pop[:, 0::2]   # Các chiều chẵn (0, 2, ..., 30): đỉnh trái p_lk
        right_genes = pop[:, 1::2]  # Các chiều lẻ (1, 3, ..., 31): đỉnh phải p_rk

        cluster_idx = np.arange(self.m)[np.newaxis, :]
        total_W = np.sum(W_left[cluster_idx, left_genes] + W_right[cluster_idx, right_genes], axis=1)
        total_D = np.sum(D_left[cluster_idx, left_genes] + D_right[cluster_idx, right_genes], axis=1)

        # Phạt nặng các nhiễm sắc thể không đủ dung lượng nhúng (Penalty function)
        deficit = np.maximum(0.0, target_payload - total_W)
        fitness = total_D + deficit * 1e5
        return fitness, total_W, total_D

    def init_population(self, use_empirical: bool = True) -> np.ndarray:
        """
        Bước 1 & 2: Mã hóa nhiễm sắc thể 2m chiều và khởi tạo quần thể.
        Nếu use_empirical=True: Chèn các nhiễm sắc thể thực nghiệm (ưu tiên nhúng ở các cụm
        Entropy thấp H1..H4 và đặt inf ở các cụm nhiễu H10..H16) giúp hội tụ cực nhanh.
        """
        pop = np.random.randint(0, 9, size=(self.M, self.dim), dtype=np.int32)

        if use_empirical:
            # Nhiễm sắc thể thực nghiệm 1..8: Đồng nhất điểm nhúng hoặc chỉ nhúng các cụm đầu
            for c_val in range(min(8, self.M)):
                pop[c_val, :] = c_val
            # Các nhiễm sắc thể thực nghiệm ưu tiên cụm mịn (H1..H4 nhúng tại đỉnh 0/-1, các cụm sau tắt = 8)
            for cut in range(1, min(self.m, self.M - 8)):
                chrom = np.full(self.dim, 8, dtype=np.int32)  # 8 là +/-inf (không nhúng)
                chrom[:2 * cut] = 0                           # Nhúng tại đỉnh cao nhất (-1 và 0) cho các cụm đầu
                pop[8 + cut - 1] = chrom
        return pop

    def horizontal_crossover(self, pop: np.ndarray) -> np.ndarray:
        """
        Bước 3: Giao phối ngang (Horizontal Crossover) giữa hai nhiễm sắc thể khác nhau
        trên cùng một chiều để mở rộng khả năng tìm kiếm toàn cục.
        """
        offspring = np.copy(pop)
        perm = np.random.permutation(self.M)
        for i in range(0, self.M - 1, 2):
            if np.random.rand() < self.pc:
                p1, p2 = perm[i], perm[i + 1]
                mask = np.random.rand(self.dim) < 0.5
                offspring[p1, mask], offspring[p2, mask] = pop[p2, mask], pop[p1, mask]
        return offspring

    def vertical_crossover(self, pop: np.ndarray) -> np.ndarray:
        """
        Bước 4: Giao phối dọc (Vertical Crossover) giữa hai chiều khác nhau trong CÙNG
        một nhiễm sắc thể để thoát khỏi cực trị cục bộ mà không phá hỏng các chiều đã tối ưu.
        """
        offspring = np.copy(pop)
        for i in range(self.M):
            if np.random.rand() < self.pv:
                d1, d2 = np.random.choice(self.dim, size=2, replace=False)
                offspring[i, d1], offspring[i, d2] = offspring[i, d2], offspring[i, d1]
        return offspring

    def mutation_and_repair(self, pop: np.ndarray) -> np.ndarray:
        """
        Bước 5: Đột biến (Mutation) và chuẩn hóa miền giá trị hợp lệ [0..8].
        """
        mut_mask = np.random.rand(pop.shape[0], self.dim) < self.pm
        random_genes = np.random.randint(0, 9, size=pop.shape, dtype=np.int32)
        offspring = np.where(mut_mask, random_genes, pop)
        return np.clip(offspring, 0, 8)

    def optimize(self, sub_histograms, target_payload: int, use_empirical: bool = True, seed: int = 42):
        """
        Chạy toàn bộ chu trình 7 bước của ICSO và trả về 2m điểm nhúng tối ưu + lịch sử hội tụ.
        """
        np.random.seed(seed)
        t0 = time.time()

        W_left, D_left, W_right, D_right = self._precompute_lookup_tables(sub_histograms)
        pop = self.init_population(use_empirical=use_empirical)

        fitness, total_W, total_D = self.evaluate_population(pop, W_left, D_left, W_right, D_right, target_payload)

        history_best_D = []
        best_chrom = None
        best_fit = float('inf')
        best_W = 0
        best_D = 0

        for gen in range(self.G_max):
            # Giao phối ngang -> Giao phối dọc -> Đột biến
            child_hc = self.horizontal_crossover(pop)
            child_vc = self.vertical_crossover(child_hc)
            offspring = self.mutation_and_repair(child_vc)

            # Đánh giá quần thể con
            off_fit, off_W, off_D = self.evaluate_population(
                offspring, W_left, D_left, W_right, D_right, target_payload
            )

            # Bước 7: Chính sách 'Giữ lại nhóm ưu tú' (Elite Retention Policy)
            combined_pop = np.vstack([pop, offspring])
            combined_fit = np.concatenate([fitness, off_fit])
            combined_W = np.concatenate([total_W, off_W])
            combined_D = np.concatenate([total_D, off_D])

            # Lọc bỏ các nhiễm sắc thể trùng lặp để duy trì tính đa dạng quần thể
            _, unique_indices = np.unique(combined_pop, axis=0, return_index=True)
            u_pop = combined_pop[unique_indices]
            u_fit = combined_fit[unique_indices]
            u_W = combined_W[unique_indices]
            u_D = combined_D[unique_indices]

            order = np.argsort(u_fit)
            if len(order) >= self.M:
                sel = order[:self.M]
                pop, fitness, total_W, total_D = u_pop[sel], u_fit[sel], u_W[sel], u_D[sel]
            else:
                # Bổ sung cá thể ngẫu nhiên nếu số cá thể duy nhất ít hơn M
                pad_n = self.M - len(order)
                pad_pop = np.random.randint(0, 9, size=(pad_n, self.dim), dtype=np.int32)
                pad_fit, pad_W, pad_D = self.evaluate_population(
                    pad_pop, W_left, D_left, W_right, D_right, target_payload
                )
                pop = np.vstack([u_pop[order], pad_pop])
                fitness = np.concatenate([u_fit[order], pad_fit])
                total_W = np.concatenate([u_W[order], pad_W])
                total_D = np.concatenate([u_D[order], pad_D])

            if fitness[0] < best_fit:
                best_fit = fitness[0]
                best_chrom = np.copy(pop[0])
                best_W = total_W[0]
                best_D = total_D[0]

            history_best_D.append(float(best_D))

        elapsed = time.time() - t0

        # Giải mã nhiễm sắc thể tốt nhất ra 2m điểm nhúng (p_lk, p_rk) và thống kê từng cụm
        optimal_points = []
        cluster_metrics = []
        for k in range(self.m):
            l_idx = int(best_chrom[2 * k])
            r_idx = int(best_chrom[2 * k + 1])
            pl = self.LEFT_CANDIDATES[l_idx]
            pr = self.RIGHT_CANDIDATES[r_idx]
            wk = float(W_left[k, l_idx] + W_right[k, r_idx])
            dk = float(D_left[k, l_idx] + D_right[k, r_idx])

            optimal_points.append((pl, pr))
            cluster_metrics.append({
                'cluster_id': k + 1,
                'pixel_count': sub_histograms[k]['pixel_count'],
                'entropy': sub_histograms[k]['entropy'],
                'p_lk': '-inf' if pl is None else str(pl),
                'p_rk': '+inf' if pr is None else str(pr),
                'W_k': int(wk),
                'D_k': dk
            })

        return {
            'optimal_points': optimal_points,
            'cluster_metrics': cluster_metrics,
            'total_W': int(best_W),
            'total_D': float(best_D),
            'history_D': history_best_D,
            'elapsed_sec': elapsed
        }