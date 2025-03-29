import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import rs_helper

def detect_stego_image(rs_values, threshold=0.1):
    """
    判斷圖片是否可能含有隱寫資訊

    :param rs_values: 3 個通道的 (Rm, Sm, R-m, S-m) 組成的 list
    :param threshold: 判斷隱寫的閾值 (越大越嚴格)
    :return: True (有隱寫), False (無隱寫)
    """
    for rm, sm, r_neg_m, s_neg_m in rs_values:
        diff_rm = abs(rm - r_neg_m)  # 計算 Rm 和 R-m 的差異
        diff_sm = abs(sm - s_neg_m)  # 計算 Sm 和 S-m 的差異
        
        # 如果任何通道的差異大於閾值，則判斷為隱寫
        if diff_rm > threshold or diff_sm > threshold:
            return True
    return False


if __name__ == "__main__":
    # 參數設定
    mask_size = (8, 8)
    mask = np.random.randint(0, 2, size=mask_size)

    # 顯示 Mask 圖
    plt.title('Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.show()

    # 📂 讀取影像
    img_path = input("📂 Enter image file name: ")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('int16')

    # 調整圖片大小以適應 mask
    img_h, img_w = img.shape[:2]
    img_h += (mask_size[0] - img_h % mask_size[0]) % mask_size[0]
    img_w += (mask_size[1] - img_w % mask_size[1]) % mask_size[1]
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

    plt.title('Image')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # 計算 RS 分析的四個參數（分别針對 R, G, B 三個通道）
    channels = [img[:, :, i] for i in range(3)]  # 分别取 R, G, B 三個通道
    results = [rs_helper([channel], mask) for channel in channels]  # 分别計算每個通道

    # # 輸出结果
    # for i, color in enumerate(["Red", "Green", "Blue"]):
    #     rm, sm, r_neg_m, s_neg_m = results[i]
    #     print(f"{color} Channel - Rm: {rm:.6f}, R-m: {r_neg_m:.6f}, Sm: {sm:.6f}, S-m: {s_neg_m:.6f}")

    # 檢測是否有隱寫
    is_stego = detect_stego_image(results)

    # 顯示結果
    if is_stego:
        print(f"⚠️ {img_path} 可能含有隱寫資訊！")
    else:
        print(f"✅ {img_path} 看起來是未修改的原始圖片。")
