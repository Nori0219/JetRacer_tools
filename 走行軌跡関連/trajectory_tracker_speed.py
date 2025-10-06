import cv2
import numpy as np
import csv
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import os
import threading
import matplotlib.pyplot as plt

class KalmanTracker:
    """カルマンフィルターを使用した軌跡追跡クラス"""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        dt = 1.0
        self.kalman.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1000
        self.initialized = False
        self.prediction_confidence = 0.0
    
    def initialize(self, x, y):
        self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        self.initialized = True
    
    def predict(self):
        if not self.initialized: return None
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def update(self, x, y):
        if not self.initialized:
            self.initialize(x, y)
            return (x, y)
        measurement = np.array([x, y], dtype=np.float32)
        self.kalman.correct(measurement)
        predicted = self.predict()
        if predicted:
            distance = np.sqrt((x - predicted[0])**2 + (y - predicted[1])**2)
            self.prediction_confidence = max(0, 1.0 - distance / 100.0)
        return (int(self.kalman.statePost[0]), int(self.kalman.statePost[1]))

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        self.frame_brightness_history.append(brightness)
        if len(self.frame_brightness_history) > 30: self.frame_brightness_history.pop(0)
        
        motion_intensity = np.mean(frame_delta)
        self.motion_intensity_history.append(motion_intensity)
        if len(self.motion_intensity_history) > 30: self.motion_intensity_history.pop(0)
    
    def optimize_threshold(self):
        if len(self.motion_intensity_history) < 5: return self.adaptive_threshold
        avg_motion = np.mean(self.motion_intensity_history[-10:])
        if avg_motion < 5: self.adaptive_threshold = max(20, self.adaptive_threshold - 1)
        elif avg_motion > 15: self.adaptive_threshold = min(60, self.adaptive_threshold + 1)
        return self.adaptive_threshold
    
    def optimize_blur_kernel(self):
        if len(self.frame_brightness_history) < 5: return (self.blur_kernel_size, self.blur_kernel_size)
        avg_brightness = np.mean(self.frame_brightness_history[-10:])
        if avg_brightness < 100: self.blur_kernel_size = min(15, self.blur_kernel_size + 2)
        elif avg_brightness > 180: self.blur_kernel_size = max(7, self.blur_kernel_size - 1)
        if self.blur_kernel_size % 2 == 0: self.blur_kernel_size += 1
        return (self.blur_kernel_size, self.blur_kernel_size)
    
    def optimize_contour_area(self):
        if len(self.detection_confidence_history) < 10: return self.min_contour_area
        avg_confidence = np.mean(self.detection_confidence_history[-10:])
        if avg_confidence < 0.5: self.min_contour_area = max(200, self.min_contour_area - 50)
        elif avg_confidence > 0.8: self.min_contour_area = min(800, self.min_contour_area + 50)
        return self.min_contour_area
    
    def update_detection_confidence(self, confidence):
        self.detection_confidence_history.append(confidence)
        if len(self.detection_confidence_history) > 20: self.detection_confidence_history.pop(0)

def enhanced_motion_detection(prev_gray, current_gray, optimizer):
    blur_kernel = optimizer.optimize_blur_kernel()
    threshold_val = optimizer.optimize_threshold()
    min_area = optimizer.optimize_contour_area()
    
    prev_blurred = cv2.GaussianBlur(prev_gray, blur_kernel, 0)
    curr_blurred = cv2.GaussianBlur(current_gray, blur_kernel, 0)
    
    frame_delta = cv2.absdiff(prev_blurred, curr_blurred)
    thresh = cv2.threshold(frame_delta, threshold_val, 255, cv2.THRESH_BINARY)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_point, confidence = None, 0.0
    if contours:
        valid_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > min_area]
        if valid_contours:
            largest_contour, area = max(valid_contours, key=lambda x: x[1])
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                confidence = min(1.0, circularity * 2)
                if confidence > 0.3:
                    M = cv2.moments(largest_contour)
                    if M["m00"] > 0:
                        detected_point = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return detected_point, confidence, frame_delta

def process_video_and_generate_outputs(video_path, track_path, progress_callback):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"動画ファイルを開けませんでした: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 

    track_image_for_scale = cv2.imread(track_path)
    if track_image_for_scale is None: raise IOError(f"サーキット画像を開けませんでした: {track_path}")

    h_pixel, w_pixel, _ = track_image_for_scale.shape
    real_height_cm, real_width_cm = 8.20 * 100, 5.36 * 100
    scale_cm_per_pixel = ((real_height_cm / h_pixel) + (real_width_cm / w_pixel)) / 2.0

    kalman_tracker, optimizer = KalmanTracker(), AdaptiveParameterOptimizer()
    raw_trajectory, kalman_trajectory, confidence_scores = [], [], []
    
    ret, prev_frame = cap.read()
    if not ret: raise IOError("動画からフレームを読み込めませんでした。")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for frame_idx in range(1, total_frames):
        ret, frame = cap.read()
        if not ret: break
        progress_callback(int((frame_idx / total_frames) * 60), f"Step 1/3: 軌跡検出中... ({frame_idx}/{total_frames})")

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_point, confidence, frame_delta = enhanced_motion_detection(prev_gray, current_gray, optimizer)
        
        optimizer.analyze_frame_conditions(frame, frame_delta)
        optimizer.update_detection_confidence(confidence)
        
        if detected_point:
            kalman_trajectory.append(kalman_tracker.update(detected_point[0], detected_point[1]))
        else:
            predicted = kalman_tracker.predict()
            if predicted and kalman_tracker.prediction_confidence > 0.3:
                kalman_trajectory.append(predicted)
                confidence = kalman_tracker.prediction_confidence * 0.5
            else:
                kalman_trajectory.append(None)
        
        raw_trajectory.append(detected_point)
        confidence_scores.append(confidence)
        prev_gray = current_gray
    
    progress_callback(60, "Step 2/3: 軌跡の後処理中...")
    filtered_trajectory = []
    for point, conf in zip(kalman_trajectory, confidence_scores):
        if point is None or conf < 0.2:
            filtered_trajectory.append(None)
            continue
        if filtered_trajectory and filtered_trajectory[-1] is not None:
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(point, filtered_trajectory[-1])))
            if dist > optimizer.max_distance_threshold:
                filtered_trajectory.append(None)
                continue
        filtered_trajectory.append(point)

    final_trajectory_with_none = []
    window_size = 7
    for i in range(len(filtered_trajectory)):
        start, end = max(0, i - window_size // 2), min(len(filtered_trajectory), i + window_size // 2 + 1)
        points = [filtered_trajectory[j] for j in range(start, end) if filtered_trajectory[j] is not None]
        weights = [confidence_scores[j] for j in range(start, end) if filtered_trajectory[j] is not None]
        if points and sum(weights) > 0:
            total_weight = sum(weights)
            wx = sum(p[0] * w for p, w in zip(points, weights)) / total_weight
            wy = sum(p[1] * w for p, w in zip(points, weights)) / total_weight
            final_trajectory_with_none.append((int(wx), int(wy)))
        else:
            final_trajectory_with_none.append(None)

    final_trajectory = [p for p in final_trajectory_with_none if p]

    instant_speeds_kmh = [None] * len(final_trajectory_with_none)
    for i in range(1, len(final_trajectory_with_none)):
        p1, p2 = final_trajectory_with_none[i-1], final_trajectory_with_none[i]
        if p1 and p2:
            dist_px = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            instant_speeds_kmh[i] = (dist_px * scale_cm_per_pixel * fps) * 0.036

    smoothed_speeds_kmh = [None] * len(instant_speeds_kmh)
    speed_window_size = 7
    for i in range(len(instant_speeds_kmh)):
        start, end = max(0, i - speed_window_size // 2), min(len(instant_speeds_kmh), i + speed_window_size // 2 + 1)
        speeds = [s for s in instant_speeds_kmh[start:end] if s is not None]
        if speeds: smoothed_speeds_kmh[i] = np.mean(speeds)

    base_path = os.path.dirname(video_path)
    csv_path = os.path.join(base_path, "trajectory_enhanced.csv")
    img_path = os.path.join(base_path, "trajectory_enhanced.png")
    video_out_path = os.path.join(base_path, "trajectory_video_enhanced.mp4")
    stats_path = os.path.join(base_path, "tracking_stats.csv")
    graph_path = os.path.join(base_path, "speed_graph.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "confidence", "raw_x", "raw_y", "speed_kmh"])
        for i, (p, c, r_p, s) in enumerate(zip(final_trajectory_with_none, confidence_scores, raw_trajectory, smoothed_speeds_kmh)):
            row = [i+1]
            row.extend([p[0], p[1], f"{c:.3f}"] if p else [None, None, f"{c:.3f}"])
            row.extend([r_p[0], r_p[1]] if r_p else [None, None])
            row.append(f"{s:.2f}" if s is not None else None)
            writer.writerow(row)

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_frames", total_frames-1])
        writer.writerow(["detected_points", sum(1 for p in raw_trajectory if p)])
        writer.writerow(["final_points", len(final_trajectory)])
        writer.writerow(["avg_confidence", f"{np.mean(confidence_scores):.3f}"])
        writer.writerow([f"detection_rate", f"{len(final_trajectory)/(total_frames-1)*100:.1f}%"])
        valid_speeds = [s for s in smoothed_speeds_kmh if s]
        if valid_speeds:
            writer.writerow(["max_speed_kmh", f"{max(valid_speeds):.2f}"])
            writer.writerow(["avg_speed_kmh", f"{np.mean(valid_speeds):.2f}"])

    track_image = cv2.imread(track_path)
    if len(final_trajectory) > 1:
        cv2.polylines(track_image, [np.array(final_trajectory)], isClosed=False, color=(255, 0, 0), thickness=3)
    cv2.imwrite(img_path, track_image)

    graph_speeds = [s for s in smoothed_speeds_kmh if s]
    if graph_speeds:
        plt.figure(figsize=(15, 7))
        plt.plot([i for i, s in enumerate(smoothed_speeds_kmh) if s], graph_speeds)
        plt.title("JetRacer Speed Over Time", fontsize=16)
        plt.xlabel("Frame Number", fontsize=12)
        plt.ylabel("Speed (km/h)", fontsize=12)
        plt.grid(True); plt.savefig(graph_path); plt.close()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    height, width, _ = prev_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    for i in range(1, len(final_trajectory_with_none) + 1):
        ret, frame = cap.read()
        if not ret: break
        progress_callback(60 + int((i / len(final_trajectory_with_none)) * 40), f"Step 3/3: 追跡ビデオ生成中... ({i}/{len(final_trajectory_with_none)})")
        
        points = [p for p in final_trajectory_with_none[:i] if p]
        if len(points) > 1:
            cv2.polylines(frame, [np.array(points)], isClosed=False, color=(255, 100, 0), thickness=3)
        
        current_point = final_trajectory_with_none[i-1]
        if current_point:
            cv2.circle(frame, current_point, 12, (255, 255, 255), 2)
            cv2.circle(frame, current_point, 8, (0, 0, 255), -1)
        
        current_speed = smoothed_speeds_kmh[i-1]
        if current_speed:
            cv2.putText(frame, f"Speed: {current_speed:.2f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    return True

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("JetRacer 走行軌跡・速度解析")
        self.root.geometry("500x320")
        
        self.video_file, self.track_file = "", ""

        main_frame = tk.Frame(root, padx=15, pady=15)
        main_frame.pack(expand=True, fill="both")

        tk.Label(main_frame, text="高精度軌跡追跡＆速度解析システム", font=("Arial", 14, "bold"), fg="navy").pack(pady=(0, 10))
        
        tk.Button(main_frame, text="1. 走行動画を選択", command=self.select_video).pack(pady=5, fill="x")
        self.lbl_video = tk.Label(main_frame, text="選択されていません", fg="gray")
        self.lbl_video.pack()

        tk.Button(main_frame, text="2. サーキット画像を選択", command=self.select_track).pack(pady=5, fill="x")
        self.lbl_track = tk.Label(main_frame, text="選択されていません", fg="gray")
        self.lbl_track.pack()

        self.btn_run = tk.Button(main_frame, text="3. 解析を実行", command=self.run_process, bg="#4CAF50", fg="white", font=("", 11, "bold"))
        self.btn_run.pack(pady=15, fill="x")
        
        self.progress_label = tk.Label(main_frame, text="", font=("", 9))
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill="x", pady=(5, 0))

    def select_video(self):
        self.video_file = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_file: self.lbl_video.config(text=f"✓ {os.path.basename(self.video_file)}", fg="darkgreen")

    def select_track(self):
        self.track_file = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.track_file: self.lbl_track.config(text=f"✓ {os.path.basename(self.track_file)}", fg="darkgreen")

    def update_progress(self, value, text):
        self.progress_bar['value'] = value
        self.progress_label['text'] = text
        self.root.update_idletasks()

    def run_process_thread(self):
        try:
            success = process_video_and_generate_outputs(self.video_file, self.track_file, self.update_progress)
            if success:
                messagebox.showinfo("処理完了", 
                    "解析が完了しました。\n動画と同じフォルダに以下のファイルが出力されました：\n\n"
                    "• trajectory_enhanced.csv\n"
                    "• trajectory_enhanced.png\n"
                    "• trajectory_video_enhanced.mp4\n"
                    "• tracking_stats.csv\n"
                    "• speed_graph.png")
        except Exception as e:
            messagebox.showerror("エラー", f"処理中にエラーが発生しました:\n{e}")
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
