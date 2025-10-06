import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import re
from PIL import Image, ImageTk
import threading

class HomographyTransformerApp:
    """
    ホモグラフィー変換を行うためのGUIアプリケーションクラス。
    ビュー調整機能（ズーム・移動）と保存バグを修正した最終版。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("鳥瞰図変換ツール (最終修正版)")
        self.root.geometry("1200x850")

        self.input_file_path = None
        self.original_image = None
        self.transformed_image = None

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._create_control_panel(main_frame)
        self._create_preview_panel(main_frame)
        self._create_status_bar(main_frame)

        main_frame.columnconfigure(0, weight=1); main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

    def _create_control_panel(self, parent_frame):
        control_frame = ttk.LabelFrame(parent_frame, text="設定 & 操作", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        control_frame.columnconfigure(1, weight=1)

        # --- 入力ウィジェット ---
        ttk.Button(control_frame, text="画像/動画ファイルを選択", command=self.load_file).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.file_label = ttk.Label(control_frame, text="ファイルが選択されていません")
        self.file_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(control_frame, text="ホモグラフィー行列 (H):").grid(row=1, column=0, padx=5, pady=5, sticky=tk.NW)
        self.matrix_text = tk.Text(control_frame, height=5, width=40)
        self.matrix_text.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.matrix_text.insert("1.0", "[[1.0, 0.0, 0.0],\n [0.0, 1.0, 0.0],\n [0.0, 0.0, 1.0]]")
        
        calib_size_frame = ttk.Frame(control_frame)
        calib_size_frame.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(control_frame, text="キャリブレーション画像サイズ:", foreground="blue").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.calib_width = tk.StringVar(value="3024"); self.calib_height = tk.StringVar(value="4032")
        ttk.Label(calib_size_frame, text="幅:").pack(side=tk.LEFT); ttk.Entry(calib_size_frame, textvariable=self.calib_width, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(calib_size_frame, text="x 高さ:").pack(side=tk.LEFT, padx=5); ttk.Entry(calib_size_frame, textvariable=self.calib_height, width=10).pack(side=tk.LEFT)
        
        size_frame = ttk.Frame(control_frame)
        size_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(control_frame, text="出力サイズ (幅 x 高さ):").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_width = tk.StringVar(value="800"); self.output_height = tk.StringVar(value="1164")
        ttk.Entry(size_frame, textvariable=self.output_width, width=10).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(size_frame, text="x").pack(side=tk.LEFT, padx=5); ttk.Entry(size_frame, textvariable=self.output_height, width=10).pack(side=tk.LEFT)
        
        # --- ビュー調整パネル ---
        adj_frame = ttk.LabelFrame(control_frame, text="ビュー調整", padding="10")
        adj_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        adj_frame.columnconfigure(1, weight=1)
        
        ttk.Label(adj_frame, text="ズーム:").grid(row=0, column=0, sticky="w"); self.zoom_var = tk.DoubleVar(value=1.0)
        ttk.Scale(adj_frame, from_=0.5, to=2.0, variable=self.zoom_var, orient="horizontal").grid(row=0, column=1, sticky="ew", padx=5)
        self.zoom_label = ttk.Label(adj_frame, text="100%"); self.zoom_label.grid(row=0, column=2)
        self.zoom_var.trace_add("write", lambda *args: self.zoom_label.config(text=f"{self.zoom_var.get():.0%}"))
        
        ttk.Label(adj_frame, text="X移動:").grid(row=1, column=0, sticky="w"); self.pan_x_var = tk.DoubleVar(value=0)
        ttk.Scale(adj_frame, from_=-0.5, to=0.5, variable=self.pan_x_var, orient="horizontal").grid(row=1, column=1, sticky="ew", padx=5)
        self.pan_x_label = ttk.Label(adj_frame, text="0%"); self.pan_x_label.grid(row=1, column=2)
        self.pan_x_var.trace_add("write", lambda *args: self.pan_x_label.config(text=f"{self.pan_x_var.get():.0%}"))

        ttk.Label(adj_frame, text="Y移動:").grid(row=2, column=0, sticky="w"); self.pan_y_var = tk.DoubleVar(value=0)
        ttk.Scale(adj_frame, from_=-0.5, to=0.5, variable=self.pan_y_var, orient="horizontal").grid(row=2, column=1, sticky="ew", padx=5)
        self.pan_y_label = ttk.Label(adj_frame, text="0%"); self.pan_y_label.grid(row=2, column=2)
        self.pan_y_var.trace_add("write", lambda *args: self.pan_y_label.config(text=f"{self.pan_y_var.get():.0%}"))
        
        # --- ボタン ---
        button_container = ttk.Frame(control_frame)
        button_container.grid(row=5, column=1, sticky=tk.E, padx=5, pady=10)
        ttk.Button(button_container, text="ビューを更新", command=self.update_preview).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_container, text="テスト情報", command=self.run_test_transformation).pack(side=tk.LEFT, padx=(0, 5))
        self.process_button = ttk.Button(button_container, text="変換実行 & 保存", command=self.process_and_save)
        self.process_button.pack(side=tk.LEFT)

    def _create_preview_panel(self, parent):
        preview_frame = ttk.Frame(parent)
        preview_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1); preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        ttk.Label(preview_frame, text="変換元プレビュー", font=("Helvetica", 14)).grid(row=0, column=0, pady=5)
        ttk.Label(preview_frame, text="変換後プレビュー", font=("Helvetica", 14)).grid(row=0, column=1, pady=5)
        self.source_panel = ttk.Label(preview_frame, background="gray80", anchor=tk.CENTER)
        self.dest_panel = ttk.Label(preview_frame, background="gray80", anchor=tk.CENTER)
        self.source_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        self.dest_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))

    def _create_status_bar(self, parent):
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        status_frame.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(status_frame, text="準備完了")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        self.progress_bar = ttk.Progressbar(status_frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=0, column=1, sticky=tk.E)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Media Files", "*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"), ("All files", "*.*")])
        if not file_path: return
        self.input_file_path = file_path
        self.file_label.config(text=file_path.split('/')[-1])
        try:
            if self.input_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.original_image = cv2.imread(self.input_file_path)
            else:
                cap = cv2.VideoCapture(self.input_file_path)
                if not cap.isOpened(): raise IOError("動画ファイルを開けませんでした。")
                ret, frame = cap.read(); cap.release()
                if not ret: raise IOError("動画からフレームを読み込めませんでした。")
                self.original_image = frame
            self.display_image(self.original_image, self.source_panel)
            self.set_status(f"ファイル '{self.file_label.cget('text')}' を読み込みました。")
        except Exception as e:
            messagebox.showerror("読み込みエラー", f"ファイルの読み込み中にエラーが発生しました:\n{e}")

    def display_image(self, cv_image, panel):
        if cv_image is None: return
        panel_w, panel_h = panel.winfo_width(), panel.winfo_height()
        if panel_w < 2 or panel_h < 2: panel_w, panel_h = 600, 400
        h, w, _ = cv_image.shape
        aspect_ratio = w / h
        if w > panel_w or h > panel_h:
            if w/panel_w > h/panel_h: new_w, new_h = panel_w, int(panel_w / aspect_ratio)
            else: new_h, new_w = panel_h, int(panel_h * aspect_ratio)
            resized_image = cv2.resize(cv_image, (new_w, new_h))
        else: resized_image = cv_image
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        photo_image = ImageTk.PhotoImage(image_pil)
        panel.config(image=photo_image); panel.image = photo_image

    def parse_inputs(self):
        try:
            matrix_str = self.matrix_text.get("1.0", tk.END)
            numbers = re.findall(r"[-+]?\d*\.\d+e?[-+]?\d*|[-+]?\d+", matrix_str)
            if len(numbers) != 9: raise ValueError(f"行列には9個の数値が必要です。")
            H = np.array(numbers, dtype=np.float32).reshape(3, 3)
            calib_w, calib_h = int(self.calib_width.get()), int(self.calib_height.get())
            out_w, out_h = int(self.output_width.get()), int(self.output_height.get())
            if any(x <= 0 for x in [calib_w, calib_h, out_w, out_h]): raise ValueError("サイズは正の整数である必要があります。")
            return H, (calib_w, calib_h), (out_w, out_h)
        except Exception as e:
            messagebox.showerror("入力エラー", f"入力値のフォーマットが正しくありません。\n詳細: {e}")
            return None, None, None

    def get_final_homography(self, base_matrix, calib_size, current_size, out_dims):
        matrix_scaled = self.scale_homography_matrix(base_matrix, calib_size, current_size)
        zoom, pan_x, pan_y = self.zoom_var.get(), self.pan_x_var.get() * out_dims[0], self.pan_y_var.get() * out_dims[1]
        cx, cy = out_dims[0] / 2, out_dims[1] / 2
        M_adjust = np.array([[zoom, 0, pan_x + cx*(1-zoom)], [0, zoom, pan_y + cy*(1-zoom)], [0, 0, 1]], dtype=np.float32)
        return M_adjust @ matrix_scaled
    
    def scale_homography_matrix(self, H, original_size, new_size):
        orig_w, orig_h = original_size; new_w, new_h = new_size
        if orig_w == 0 or orig_h == 0 or (original_size == new_size): return H
        scale_x, scale_y = new_w / orig_w, new_h / orig_h
        S_inv = np.array([[1/scale_x, 0, 0], [0, 1/scale_y, 0], [0, 0, 1]], dtype=np.float32)
        return H @ S_inv

    def set_status(self, message):
        self.status_label.config(text=message); self.root.update_idletasks()

    def update_preview(self):
        if self.original_image is None: messagebox.showwarning("ファイル未選択", "ファイルを選択してください。"); return
        base_matrix, calib_size, out_dims = self.parse_inputs()
        if base_matrix is None: return
        self.set_status("プレビューを更新中...")
        current_size = (self.original_image.shape[1], self.original_image.shape[0])
        final_matrix = self.get_final_homography(base_matrix, calib_size, current_size, out_dims)
        transformed_image = cv2.warpPerspective(self.original_image, final_matrix, out_dims)
        self.display_image(transformed_image, self.dest_panel)
        self.set_status("プレビューを更新しました。")

    def run_test_transformation(self):
        if self.original_image is None: messagebox.showwarning("ファイル未選択", "ファイルを選択してください。"); return
        base_matrix, calib_size, out_dims = self.parse_inputs()
        if base_matrix is None: return
        current_size = (self.original_image.shape[1], self.original_image.shape[0])
        final_matrix = self.get_final_homography(base_matrix, calib_size, current_size, out_dims)
        transformed_image = cv2.warpPerspective(self.original_image, final_matrix, out_dims)
        gray = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        non_black, total = np.count_nonzero(gray), out_dims[0] * out_dims[1]
        coverage = non_black / total if total > 0 else 0
        messagebox.showinfo("テスト情報", f"キャリブレーションサイズ: {calib_size}\n現在の画像サイズ: {current_size}\n出力サイズ: {out_dims}\n画像カバレッジ: {coverage:.1%}")

    def process_and_save(self):
        if not self.input_file_path: messagebox.showwarning("ファイル未選択", "ファイルを選択してください。"); return
        base_matrix, calib_size, out_dims = self.parse_inputs()
        if base_matrix is None: return
        is_image = self.input_file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        default_ext = "." + self.input_file_path.split('.')[-1] if is_image else ".mp4"
        file_types = [("Image file", "*.jpg *.png")] if is_image else [("MP4 video", "*.mp4")]
        output_path = filedialog.asksaveasfilename(defaultextension=default_ext, filetypes=file_types)
        if not output_path: self.set_status("保存がキャンセルされました。"); return
        self.process_button.config(state=tk.DISABLED)
        threading.Thread(target=self._run_transformation, args=(base_matrix, calib_size, out_dims, output_path, is_image)).start()

    def _run_transformation(self, base_matrix, calib_size, out_dims, output_path, is_image):
        try:
            if is_image:
                self._transform_image(base_matrix, calib_size, out_dims, output_path)
            else:
                self._transform_video(base_matrix, calib_size, out_dims, output_path)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("変換エラー", f"処理中にエラーが発生しました:\n{e}"))
        finally:
            self.set_status("準備完了"); self.progress_bar['value'] = 0
            self.root.after(0, lambda: self.process_button.config(state=tk.NORMAL))

    def _transform_image(self, base_matrix, calib_size, out_dims, output_path):
        self.set_status("画像を変換中...")
        current_size = (self.original_image.shape[1], self.original_image.shape[0])
        final_matrix = self.get_final_homography(base_matrix, calib_size, current_size, out_dims)
        self.transformed_image = cv2.warpPerspective(self.original_image, final_matrix, out_dims)
        self.display_image(self.transformed_image, self.dest_panel)
        cv2.imwrite(output_path, self.transformed_image)
        self.set_status(f"変換完了！ {output_path} に保存しました。")
        self.root.after(0, lambda: messagebox.showinfo("成功", "画像を正常に変換し、保存しました。"))

    def _transform_video(self, base_matrix, calib_size, out_dims, output_path):
        self.set_status("動画を変換中...")
        cap = cv2.VideoCapture(self.input_file_path)
        if not cap.isOpened(): raise IOError("入力動画ファイルを開けませんでした。")
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        current_video_size = (video_width, video_height)
        final_matrix = self.get_final_homography(base_matrix, calib_size, current_video_size, out_dims)
        if calib_size != current_video_size: self.set_status(f"行列をスケーリング: {calib_size} -> {current_video_size}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: total_frames = 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, out_dims)
        if not writer.isOpened(): cap.release(); raise IOError(f"動画ファイル '{output_path}' の作成に失敗しました。")
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                transformed_frame = cv2.warpPerspective(frame, final_matrix, out_dims)
                writer.write(transformed_frame)
                frame_count += 1
                self.progress_bar['value'] = (frame_count / total_frames) * 100
                if frame_count == 1: self.display_image(transformed_frame, self.dest_panel)
        finally:
            cap.release(); writer.release()
        self.set_status(f"変換完了！ {output_path} に保存しました。")
        self.root.after(0, lambda: messagebox.showinfo("成功", f"動画を正常に変換し、保存しました。\nフレーム数: {frame_count}"))

if __name__ == "__main__":
    root = tk.Tk()
    app = HomographyTransformerApp(root)
    root.mainloop()