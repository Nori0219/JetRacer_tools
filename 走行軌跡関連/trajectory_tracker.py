import cv2
import numpy as np
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import threading

class KalmanTracker:
    """カルマンフィルターを使用した軌跡追跡クラス"""
    
    def __init__(self):
        # 4状態（x, y, vx, vy）、2観測値（x, y）のカルマンフィルター
        self.kalman = cv2.KalmanFilter(4, 2)
        
        # 測定行列 (観測値 = 位置のみ)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 状態遷移行列 (等速度モデル)
        dt = 1.0  # フレーム間隔
        self.kalman.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # プロセスノイズ共分散行列
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # 測定ノイズ共分散行列
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        
        # エラー共分散行列
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        
        self.initialized = False
        self.prediction_confidence = 0.0
    
    def initialize(self, x, y):
        """初期位置でカルマンフィルターを初期化"""
        self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        self.initialized = True
    
    def predict(self):
        """次の状態を予測"""
        if not self.initialized:
            return None
        
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def update(self, x, y):
        """観測値で状態を更新"""
        if not self.initialized:
            self.initialize(x, y)
            return (x, y)
        
        measurement = np.array([x, y], dtype=np.float32)
        self.kalman.correct(measurement)
        
        # 予測の信頼度を計算
        predicted = self.predict()
        if predicted:
            distance = np.sqrt((x - predicted[0])**2 + (y - predicted[1])**2)
            self.prediction_confidence = max(0, 1.0 - distance / 100.0)
        
        return (int(self.kalman.statePost[0]), int(self.kalman.statePost[1]))
    
    def get_velocity(self):
        """現在の速度を取得"""
        if not self.initialized:
            return (0, 0)
        return (self.kalman.statePost[2], self.kalman.statePost[3])

class AdaptiveParameterOptimizer:
    """環境に応じてパラメーターを最適化するクラス"""
    
    def __init__(self):
        self.frame_brightness_history = []
        self.motion_intensity_history = []
        self.detection_confidence_history = []
        self.adaptive_threshold = 35
        self.blur_kernel_size = 11
        self.min_contour_area = 400
        self.max_distance_threshold = 150
        
    def analyze_frame_conditions(self, frame, frame_delta):
        """フレームの条件を分析"""
        # 明度の計算
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.frame_brightness_history.append(brightness)
        if len(self.frame_brightness_history) > 30:
            self.frame_brightness_history.pop(0)
        
        # 動きの強度を計算
        motion_intensity = np.mean(frame_delta)
        self.motion_intensity_history.append(motion_intensity)
        if len(self.motion_intensity_history) > 30:
            self.motion_intensity_history.pop(0)
    
    def optimize_threshold(self):
        """適応的閾値の最適化"""
        if len(self.motion_intensity_history) < 5:
            return self.adaptive_threshold
        
        avg_motion = np.mean(self.motion_intensity_history[-10:])
        
        # 動きが少ない場合は閾値を下げ、多い場合は上げる
        if avg_motion < 5:
            self.adaptive_threshold = max(20, self.adaptive_threshold - 1)
        elif avg_motion > 15:
            self.adaptive_threshold = min(60, self.adaptive_threshold + 1)
        
        return self.adaptive_threshold
    
    def optimize_blur_kernel(self):
        """ブラーカーネルサイズの最適化"""
        if len(self.frame_brightness_history) < 5:
            return (self.blur_kernel_size, self.blur_kernel_size)
        
        avg_brightness = np.mean(self.frame_brightness_history[-10:])
        
        # 暗い環境ではより強いブラーを適用
        if avg_brightness < 100:
            self.blur_kernel_size = min(15, self.blur_kernel_size + 2)
        elif avg_brightness > 180:
            self.blur_kernel_size = max(7, self.blur_kernel_size - 1)
        
        # 奇数にする
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
            
        return (self.blur_kernel_size, self.blur_kernel_size)
    
    def optimize_contour_area(self):
        """輪郭面積の閾値最適化"""
        if len(self.detection_confidence_history) < 10:
            return self.min_contour_area
        
        avg_confidence = np.mean(self.detection_confidence_history[-10:])
        
        # 検出信頼度が低い場合は面積閾値を下げる
        if avg_confidence < 0.5:
            self.min_contour_area = max(200, self.min_contour_area - 50)
        elif avg_confidence > 0.8:
            self.min_contour_area = min(800, self.min_contour_area + 50)
        
        return self.min_contour_area
    
    def update_detection_confidence(self, confidence):
        """検出信頼度を更新"""
        self.detection_confidence_history.append(confidence)
        if len(self.detection_confidence_history) > 20:
            self.detection_confidence_history.pop(0)

def enhanced_motion_detection(prev_gray, current_gray, optimizer):
    """拡張された動体検出"""
    # 適応的パラメーターを取得
    blur_kernel = optimizer.optimize_blur_kernel()
    threshold_val = optimizer.optimize_threshold()
    min_area = optimizer.optimize_contour_area()
    
    # ブラー処理
    prev_blurred = cv2.GaussianBlur(prev_gray, blur_kernel, 0)
    curr_blurred = cv2.GaussianBlur(current_gray, blur_kernel, 0)
    
    # フレーム差分
    frame_delta = cv2.absdiff(prev_blurred, curr_blurred)
    
    # 適応的閾値処理
    thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)[1]
    
    # モルフォロジー処理の強化
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # 輪郭検出
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_point = None
    confidence = 0.0
    
    if contours:
        # 面積と形状を考慮した最適な輪郭選択
        valid_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > min_area]
        
        if valid_contours:
            # 最大面積の輪郭を選択
            largest_contour, area = max(valid_contours, key=lambda x: x[1])
            
            # 形状の妥当性をチェック
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                confidence = min(1.0, circularity * 2)  # 円形度を信頼度に変換
                
                if confidence > 0.3:  # 最低限の形状要件
                    M = cv2.moments(largest_contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        detected_point = (cx, cy)
    
    return detected_point, confidence, frame_delta

def process_video_and_generate_outputs(video_path, track_path, progress_callback):
    """
    カルマンフィルターと適応的パラメーター最適化を使用した軌跡検出・生成
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画ファイルを開けませんでした:\n{video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 

    # 追跡システムの初期化
    kalman_tracker = KalmanTracker()
    optimizer = AdaptiveParameterOptimizer()
    
    raw_trajectory = []
    kalman_trajectory = []
    confidence_scores = []
    
    ret, prev_frame = cap.read()
    if not ret: raise IOError("動画からフレームを読み込めませんでした。")
        
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # --- 1. 拡張軌跡検出 ---
    for frame_idx in range(1, total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        progress_callback(int((frame_idx / total_frames) * 60), 
                         f"Step 1/3: 高精度軌跡検出中... ({frame_idx}/{total_frames})")

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 拡張された動体検出
        detected_point, confidence, frame_delta = enhanced_motion_detection(
            prev_gray, current_gray, optimizer
        )
        
        # フレーム条件の分析
        optimizer.analyze_frame_conditions(frame, frame_delta)
        optimizer.update_detection_confidence(confidence)
        
        # カルマンフィルターによる追跡
        if detected_point:
            # 検出された点をカルマンフィルターで更新
            kalman_point = kalman_tracker.update(detected_point[0], detected_point[1])
            kalman_trajectory.append(kalman_point)
        else:
            # 検出失敗時は予測値を使用
            predicted = kalman_tracker.predict()
            if predicted and kalman_tracker.prediction_confidence > 0.3:
                kalman_trajectory.append(predicted)
                confidence = kalman_tracker.prediction_confidence * 0.5  # 予測値の信頼度は低めに
            else:
                kalman_trajectory.append(None)
        
        raw_trajectory.append(detected_point)
        confidence_scores.append(confidence)
        prev_gray = current_gray
    
    # --- 2. 軌跡の後処理とフィルタリング ---
    progress_callback(60, "Step 2/3: 軌跡の後処理中...")
    
    # 信頼度ベースのフィルタリング
    filtered_trajectory = []
    for i, (point, conf) in enumerate(zip(kalman_trajectory, confidence_scores)):
        if point is None:
            filtered_trajectory.append(None)
            continue
            
        # 信頼度が低い点は除外
        if conf < 0.2:
            filtered_trajectory.append(None)
            continue
            
        # 速度チェック（物理的制約）
        if len(filtered_trajectory) > 0 and filtered_trajectory[-1] is not None:
            prev_point = filtered_trajectory[-1]
            distance = np.sqrt((point[0] - prev_point[0])**2 + (point[1] - prev_point[1])**2)
            if distance > optimizer.max_distance_threshold:
                filtered_trajectory.append(None)
                continue
        
        filtered_trajectory.append(point)

    # 最終的な移動平均フィルター（信頼度重み付き）
    final_trajectory_with_none = []
    window_size = 7
    
    for i in range(len(filtered_trajectory)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(filtered_trajectory), i + window_size // 2 + 1)
        
        window_points = []
        window_weights = []
        
        for j in range(start_idx, end_idx):
            if filtered_trajectory[j] is not None and j < len(confidence_scores):
                window_points.append(filtered_trajectory[j])
                window_weights.append(confidence_scores[j])
        
        if window_points:
            # 重み付き平均
            total_weight = sum(window_weights)
            if total_weight > 0:
                weighted_x = sum(p[0] * w for p, w in zip(window_points, window_weights)) / total_weight
                weighted_y = sum(p[1] * w for p, w in zip(window_points, window_weights)) / total_weight
                final_trajectory_with_none.append((int(weighted_x), int(weighted_y)))
            else:
                final_trajectory_with_none.append(None)
        else:
            final_trajectory_with_none.append(None)

    final_trajectory = [p for p in final_trajectory_with_none if p]

    # --- 3. 出力ファイルの生成 ---
    base_path = os.path.dirname(video_path)
    csv_path = os.path.join(base_path, "trajectory_enhanced.csv")
    img_path = os.path.join(base_path, "trajectory_enhanced.png")
    video_out_path = os.path.join(base_path, "trajectory_video_enhanced.mp4")
    stats_path = os.path.join(base_path, "tracking_stats.csv")

    # CSV出力（拡張版）
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "confidence", "raw_x", "raw_y"])
        for i, (final_p, conf, raw_p) in enumerate(zip(final_trajectory_with_none, confidence_scores, raw_trajectory)):
            row = [i+1]
            if final_p:
                row.extend([final_p[0], final_p[1], f"{conf:.3f}"])
            else:
                row.extend([None, None, f"{conf:.3f}"])
            
            if raw_p:
                row.extend([raw_p[0], raw_p[1]])
            else:
                row.extend([None, None])
            writer.writerow(row)

    # 統計情報の出力
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_frames", total_frames-1])
        writer.writerow(["detected_points", sum(1 for p in raw_trajectory if p is not None)])
        writer.writerow(["final_points", len(final_trajectory)])
        writer.writerow(["avg_confidence", f"{np.mean(confidence_scores):.3f}"])
        writer.writerow(["detection_rate", f"{len(final_trajectory)/(total_frames-1)*100:.1f}%"])

    # 静止画出力
    track_image = cv2.imread(track_path)
    if len(final_trajectory) > 1:
        # メイン軌跡（青）
        cv2.polylines(track_image, [np.array(final_trajectory)], isClosed=False, color=(255, 0, 0), thickness=3)
        # 信頼度の高い点をハイライト（緑）
        high_conf_points = [p for p, c in zip(final_trajectory_with_none, confidence_scores) 
                           if p and c > 0.8]
        for point in high_conf_points:
            cv2.circle(track_image, point, 3, (0, 255, 0), -1)
    
    cv2.imwrite(img_path, track_image)

    # 動画出力
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    height, width, _ = prev_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    ret, frame = cap.read()  # 最初のフレームを読み直し
    
    for i in range(1, min(total_frames, len(final_trajectory_with_none) + 1)):
        ret, frame = cap.read()
        if not ret: break
        
        progress_callback(60 + int((i / total_frames) * 40), 
                         f"Step 3/3: 高品質追跡ビデオ生成中... ({i}/{total_frames})")

        # その時点までの軌跡を描画
        current_trajectory_points = []
        
        # 有効な軌跡点を収集（現在のフレームまで）
        for j in range(min(i, len(final_trajectory_with_none))):
            point = final_trajectory_with_none[j]
            if point is not None:
                current_trajectory_points.append(point)
        
        # 軌跡線を青色で描画（polylines使用でスムーズな線）
        if len(current_trajectory_points) > 1:
            cv2.polylines(frame, [np.array(current_trajectory_points)], 
                         isClosed=False, color=(255, 100, 0), thickness=3)  # 青色の軌跡
        
        # 信頼度に応じた点の可視化（軌跡上の点として）
        for j in range(min(i, len(final_trajectory_with_none))):
            point = final_trajectory_with_none[j]
            if point is None:
                continue
            
            # 対応する信頼度を取得（インデックス調整）
            confidence = confidence_scores[j] if j < len(confidence_scores) else 0
            
            # 信頼度に応じた点の色とサイズ
            if confidence > 0.7:
                # 高信頼度：明るい緑の小さな点
                cv2.circle(frame, point, 2, (0, 255, 0), -1)
            elif confidence > 0.4:
                # 中信頼度：黄色の点
                cv2.circle(frame, point, 2, (0, 255, 255), -1)
            elif confidence > 0.2:
                # 低信頼度：オレンジの点
                cv2.circle(frame, point, 1, (0, 165, 255), -1)
            # 信頼度0.2以下は点を描画しない
        
        # 現在のフレームでの検出点を強調表示
        current_frame_idx = i - 1
        if (current_frame_idx < len(final_trajectory_with_none) and 
            final_trajectory_with_none[current_frame_idx] is not None):
            
            current_point = final_trajectory_with_none[current_frame_idx]
            current_confidence = (confidence_scores[current_frame_idx] 
                                if current_frame_idx < len(confidence_scores) else 0)
            
            # 現在位置を大きな円で強調
            cv2.circle(frame, current_point, 12, (255, 255, 255), 2)  # 白い外枠
            
            # 信頼度に応じた内側の色
            if current_confidence > 0.7:
                inner_color = (0, 255, 0)  # 緑
            elif current_confidence > 0.4:
                inner_color = (0, 255, 255)  # 黄
            elif current_confidence > 0.2:
                inner_color = (0, 165, 255)  # オレンジ
            else:
                inner_color = (0, 0, 255)  # 赤
            
            cv2.circle(frame, current_point, 8, inner_color, -1)
            
            # 信頼度テキストを表示
            conf_text = f"Conf: {current_confidence:.2f}"
            cv2.putText(frame, conf_text, 
                       (current_point[0] + 15, current_point[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        writer.write(frame)

    cap.release()
    writer.release()
    return True

# =============================
# GUI部分（既存のものを拡張）
# =============================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("JetRacer 走行軌跡解析 (カルマンフィルター版)")
        self.root.geometry("500x320")
        
        self.video_file = ""
        self.track_file = ""

        main_frame = tk.Frame(root, padx=15, pady=15)
        main_frame.pack(expand=True, fill="both")

        # タイトル
        title_label = tk.Label(main_frame, text="高精度軌跡追跡システム", 
                              font=("Arial", 14, "bold"), fg="navy")
        title_label.pack(pady=(0, 10))

        # 機能説明
        features_text = ("✓ カルマンフィルターによる予測追跡\n"
                        "✓ 適応的パラメーター最適化\n"
                        "✓ 信頼度ベースのフィルタリング\n"
                        "✓ 詳細な統計情報出力")
        features_label = tk.Label(main_frame, text=features_text, 
                                 font=("Arial", 9), fg="darkgreen", justify=tk.LEFT)
        features_label.pack(pady=(0, 15))

        # UI Widgets
        tk.Button(main_frame, text="1. 走行動画を選択", command=self.select_video, 
                 bg="#e6f3ff", font=("", 10)).pack(pady=5, fill="x")
        self.lbl_video = tk.Label(main_frame, text="選択されていません", fg="gray")
        self.lbl_video.pack()

        tk.Button(main_frame, text="2. サーキット画像（台紙）を選択", command=self.select_track,
                 bg="#e6f3ff", font=("", 10)).pack(pady=5, fill="x")
        self.lbl_track = tk.Label(main_frame, text="選択されていません", fg="gray")
        self.lbl_track.pack()

        self.btn_run = tk.Button(main_frame, text="3. 高精度軌跡解析を実行", 
                                command=self.run_process, bg="#4CAF50", fg="white",
                                font=("", 11, "bold"))
        self.btn_run.pack(pady=15, fill="x")
        
        self.progress_label = tk.Label(main_frame, text="", font=("", 9))
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(5, 0))

        # 結果説明
        result_label = tk.Label(main_frame, 
                               text="出力: trajectory_enhanced.csv/.png, trajectory_video_enhanced.mp4, tracking_stats.csv",
                               font=("Arial", 8), fg="darkblue")
        result_label.pack(pady=(10, 0))

    def select_video(self):
        self.video_file = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")],
            title="走行動画を選択してください"
        )
        if self.video_file: 
            filename = os.path.basename(self.video_file)
            self.lbl_video.config(text=f"✓ {filename}", fg="darkgreen")

    def select_track(self):
        self.track_file = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg")],
            title="サーキット画像を選択してください"
        )
        if self.track_file: 
            filename = os.path.basename(self.track_file)
            self.lbl_track.config(text=f"✓ {filename}", fg="darkgreen")

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label['text'] = text
        self.root.update_idletasks()

    def run_process_thread(self):
        try:
            success = process_video_and_generate_outputs(self.video_file, self.track_file, self.update_progress)
            if success:
                messagebox.showinfo("処理完了", 
                    "高精度軌跡解析が完了しました！\n\n"
                    "生成されたファイル:\n"
                    "• trajectory_enhanced.csv (詳細軌跡データ)\n"
                    "• trajectory_enhanced.png (軌跡画像)\n"
                    "• trajectory_video_enhanced.mp4 (追跡動画)\n"
                    "• tracking_stats.csv (解析統計)\n\n"
                    "すべて動画と同じフォルダに保存されました。")
        except Exception as e:
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{str(e)}")
        finally:
            self.btn_run.config(state="normal")
            self.update_progress(0, "解析待機中...")

    def run_process(self):
        if not self.video_file or not self.track_file:
            messagebox.showerror("エラー", "走行動画とサーキット画像の両方を選択してください。")
            return
        
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.run_process_thread, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

