import numpy as np

def extract_layer_features_and_errors(image: np.ndarray, layer: int = 1):
    """
    Chia ảnh theo dạng bàn cờ vua (Rhombus pattern) và trích xuất:
    - Tọa độ các pixel thuộc lớp hiện tại (If nếu layer=1, Is nếu layer=2)
    - Giá trị dự đoán hình thoi (x_hat) và sai số dự đoán (e = x - x_hat)
    - Ma trận K = 10 đặc trưng cục bộ (F) phản ánh độ phức tạp kết cấu khối 5x5.
    """
    img = image.astype(np.float64)
    H, W = img.shape

    # Bỏ qua viền 2 hàng đầu/cuối và 2 cột đầu/cuối để đủ lân cận 5x5 và lưu thông tin phụ trợ
    r_grid, c_grid = np.meshgrid(np.arange(2, H - 2), np.arange(2, W - 2), indexing='ij')

    if layer == 1:
        mask = (r_grid + c_grid) % 2 == 0  # Tập If (Lớp 1)
    else:
        mask = (r_grid + c_grid) % 2 == 1  # Tập Is (Lớp 2)

    rows = r_grid[mask]
    cols = c_grid[mask]

    # Pixel gốc tại vị trí trung tâm (i, j)
    x = img[rows, cols]

    # 4 pixel lân cận trực tiếp tạo hình thoi 3x3: v1(trên), v2(trái), v3(dưới), v4(phải)
    v1 = img[rows - 1, cols]      # I_{i-1, j}
    v2 = img[rows, cols - 1]      # I_{i, j-1}
    v3 = img[rows + 1, cols]      # I_{i+1, j}
    v4 = img[rows, cols + 1]      # I_{i, j+1}

    # Dự đoán hình thoi (Phương trình 12 & 13)
    x_hat = np.floor((v1 + v2 + v3 + v4) / 4.0)
    errors = (x - x_hat).astype(np.int32)

    # 8 pixel lân cận vòng ngoài trong cửa sổ 5x5 (Hình 2c trong bài báo)
    u_tl = img[rows - 2, cols - 1]  # I_{i-2, j-1}
    u_tr = img[rows - 2, cols + 1]  # I_{i-2, j+1}
    u_lt = img[rows - 1, cols - 2]  # I_{i-1, j-2}
    u_rt = img[rows - 1, cols + 2]  # I_{i-1, j+2}
    u_lb = img[rows + 1, cols - 2]  # I_{i+1, j-2}
    u_rb = img[rows + 1, cols + 2]  # I_{i+1, j+2}
    u_bl = img[rows + 2, cols - 1]  # I_{i+2, j-1}
    u_br = img[rows + 2, cols + 1]  # I_{i+2, j+1}

    # --- THIẾT KẾ K = 10 ĐẶC TRƯNG ĐỘ PHỨC TẠP CỤC BỘ (Mục 2.1 & Phụ lục) ---
    # f1, f2, f3: Độ lệch theo phương dọc, ngang và chéo chính của 4 lân cận sát
    f1 = np.abs(v1 - v3)
    f2 = np.abs(v2 - v4)
    f3 = np.abs(v1 + v3 - v2 - v4)

    # f4: Tổng trị tuyệt đối hiệu 4 điểm ảnh liền kề (Phương trình 21)
    dv1 = np.abs(v1 - v2)
    dv2 = np.abs(v2 - v3)
    dv3 = np.abs(v3 - v4)
    dv4 = np.abs(v4 - v1)
    f4 = dv1 + dv2 + dv3 + dv4

    # f5: Tổng của 4 gradient trên đường chéo bên trái (Phương trình 22, 23)
    ldg1 = np.abs(u_tl - v1) + np.abs(v2 - v1)
    ldg2 = np.abs(v1 - v4) + np.abs(v3 - v4)
    ldg3 = np.abs(v2 - v3) + np.abs(u_bl - v3)
    ldg4 = np.abs(u_lt - v2) + np.abs(u_lb - v2)
    f5 = ldg1 + ldg2 + ldg3 + ldg4

    # f6: Tổng của 4 gradient đường chéo bên phải (Phương trình 24, 25)
    rdg1 = np.abs(u_tr - v1) + np.abs(v4 - v1)
    rdg2 = np.abs(u_rt - v4) + np.abs(u_rb - v4)
    rdg3 = np.abs(v4 - v3) + np.abs(u_br - v3)
    rdg4 = np.abs(v1 - v2) + np.abs(v3 - v2)
    f6 = rdg1 + rdg2 + rdg3 + rdg4

    # f7: Tổng chênh lệch tuyệt đối giữa các cặp gradient chéo trái (Phương trình 26)
    ldgs = [ldg1, ldg2, ldg3, ldg4]
    f7 = sum(np.abs(ldgs[k] - ldgs[i]) for k in range(3) for i in range(k + 1, 4))

    # f8: Tổng chênh lệch tuyệt đối giữa các cặp gradient chéo phải (Phương trình 27)
    rdgs = [rdg1, rdg2, rdg3, rdg4]
    f8 = sum(np.abs(rdgs[k] - rdgs[i]) for k in range(3) for i in range(k + 1, 4))

    # f9: Phương sai mẫu của 4 hiệu số liền kề (đặc trưng độ đồng nhất hướng)
    dv_stack = np.vstack([dv1, dv2, dv3, dv4])
    f9 = np.var(dv_stack, axis=0)

    # f10: Phương sai cục bộ của toàn bộ 12 điểm ảnh bối cảnh trong khối 5x5
    neighbors_12 = np.vstack([v1, v2, v3, v4, u_tl, u_tr, u_lt, u_rt, u_lb, u_rb, u_bl, u_br])
    f10 = np.var(neighbors_12, axis=0)

    # Ghép thành ma trận đặc trưng F kích thước (N x 10)
    F = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

    return rows, cols, x.astype(np.int32), x_hat.astype(np.int32), errors, F