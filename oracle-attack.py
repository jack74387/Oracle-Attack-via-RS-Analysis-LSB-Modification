import cv2
import numpy as np
import matplotlib.pyplot as plt
from RS_Analysis import detect_stego_image  # 載入隱寫檢測函數
from util import rs_helper  # 假設這是 RS 分析的輔助工具

def modify_lsb(image):
    """
    修改影像的 LSB (最低有效位元)，以進行 Oracle Attack。
    
    :param image: 影像資料 (numpy array)
    :return: 修改 LSB 後的影像
    """
    # 修改影像的 LSB
    image_copy = image.copy()
    image_copy = image_copy & ~1  # 清除 LSB
    return image_copy


def perform_oracle_attack(img, mask, max_modifications=50):
    """
    使用 Oracle Attack 方法對圖像進行多次修改，直到成功不被偵測。
    
    :param img: 影像資料 (numpy array)
    :param mask: 用來進行 RS 分析的 mask
    :param max_modifications: 最大修改次數
    :return: 成功不被偵測的修改次數
    """
    # 計算影像的 RS 分析
    channels = [img[:, :, i] for i in range(3)]  # 分別取 R, G, B 三個通道
    results = [rs_helper([channel], mask) for channel in channels]  # 計算每個通道的 RS 值
    print(f"RS 分析 p 值: {results}")
    # 使用隱寫檢測
    is_stego = detect_stego_image(results)

    # 只有當圖像是隱寫的時才進行修改
    modification_count = 0

    if is_stego:  # 如果圖像是隱寫，開始修改 LSB
        while is_stego and modification_count < max_modifications:
            print(f"⚠️ 第 {modification_count+1} 次修改 LSB，進行 Oracle Attack...")
            img = modify_lsb(img)  # 修改 LSB
            results = [rs_helper([channel], mask) for channel in [img[:, :, i] for i in range(3)]]  # 重新計算 RS 分析
            print(f"RS 分析 p 值: {results}")
            is_stego = detect_stego_image(results)  # 再次檢測是否含有隱寫
            modification_count += 1

    return modification_count, img



# 讀取圖像文件
img_path = input("📂 輸入圖片檔案名稱: ")
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('int16')

# 設置 mask 大小
mask_size = (8, 8)
mask = np.random.randint(0, 2, size=mask_size)

# 顯示原始影像
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')
plt.show()

# 開始進行 Oracle Attack
modification_count, modified_img = perform_oracle_attack(img, mask)

# 顯示修改後的影像
plt.title('Modified Image (After Oracle Attack)')
plt.imshow(modified_img)
plt.axis('off')
plt.show()

# 顯示最終結果
if modification_count < 50:
    print(f"✅ 攻擊成功！經過 {modification_count} 次修改 LSB 後，影像已無法被偵測為隱寫。")
else:
    print(f"⚠️ 即使經過 {modification_count} 次修改，影像仍然被偵測為隱寫！")      