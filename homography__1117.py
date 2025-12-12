# 画像 'GoPro1117.jpg' からのキャリブレーション結果 (手動8点)
import numpy as np
import cv2

# 元画像上のピクセル座標 (左上角→右上角→右下角→左下角→上辺中央→右辺中央→下辺中央→左辺中央)
source_points = np.float32([
        [478, 97],
        [803, 94],
        [1159, 646],
        [213, 685],
        [642, 96],
        [926, 290],
        [664, 677],
        [400, 248]
    ])

# 対応する出力画像上の座標
# この座標は物理的なアスペクト比を維持しています。
destination_points = np.float32([
        [0.00, 0.00],
        [800.00, 0.00],
        [800.00, 1236.36],
        [0.00, 1236.36],
        [400.00, 0.00],
        [800.00, 618.18],
        [400.00, 1236.36],
        [0.00, 618.18]
    ])

# 【推奨】8点全ての情報から、最も精度の高いホモグラフィー行列を計算
# RANSACアルゴリズムにより、クリック誤差など外れ値に対する耐性も向上します。
homography_matrix, status = cv2.findHomography(source_points, destination_points)
# --- 計算結果の出力 ---
print("■ ホモグラフィー行列 (GUIツールに貼り付けてください):")
print(homography_matrix)
print("\n■ 出力サイズ (GUIツールに入力してください):")
output_width = destination_points[1][0]
output_height = destination_points[2][1]
print(f"幅: {int(output_width)}")
print(f"高さ: {int(output_height)}")

# 例: 鳥瞰図への変換
# output_width = 800
# output_height = 1236
# bird_eye_view = cv2.warpPerspective(original_image, homography_matrix, (output_width, output_height))