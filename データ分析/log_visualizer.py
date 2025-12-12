import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

# --- ハードコード設定 (ここで閾値を変更してください) ---
D_SAFE_FRONT = 100   # 前方安全距離閾値 [cm] ($d_{safe, front}$)
D_SAFE_SIDE = 60    # 側方安全距離閾値 [cm] ($d_{safe, side}$)
D_CRIT = 30         # 限界距離閾値 [cm] ($d_{crit}$)

class LogVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JetRacer Log Visualizer V2")
        self.root.geometry("1000x800")

        # Data storage
        self.df = None
        self.file_path = None
        self.x_axis_mode = "frame"  # Default to frame for detail check

        # --- GUI Layout ---
        
        # Top Frame for Controls
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # File Select Button
        self.btn_load = tk.Button(control_frame, text="Load CSV File", command=self.load_csv, bg="#DDDDDD")
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # X-Axis Toggle Button
        self.btn_toggle_x = tk.Button(control_frame, text="X-Axis: Frame", command=self.toggle_x_axis, state=tk.DISABLED)
        self.btn_toggle_x.pack(side=tk.LEFT, padx=5)

        # File Name Label
        self.lbl_filename = tk.Label(control_frame, text="No file loaded", fg="gray")
        self.lbl_filename.pack(side=tk.LEFT, padx=10)

        # Save Plot Button
        self.btn_save = tk.Button(control_frame, text="Save Graph as PNG", command=self.save_graph, state=tk.DISABLED)
        self.btn_save.pack(side=tk.RIGHT, padx=5)

        # Slider Frame (for Range)
        slider_frame = tk.Frame(root)
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        self.lbl_slider = tk.Label(slider_frame, text="Frame Range:")
        self.lbl_slider.pack(side=tk.LEFT)
        
        self.slider_min = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Start", length=300, command=self.update_plot_event)
        self.slider_min.pack(side=tk.LEFT, padx=10)
        
        self.slider_max = tk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="End", length=300, command=self.update_plot_event)
        self.slider_max.pack(side=tk.LEFT, padx=10)

        # Matplotlib Figure
        self.fig = Figure(figsize=(8, 8), dpi=100)
        # Add subplots (3 rows, 1 column)
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312, sharex=self.ax1)
        self.ax3 = self.fig.add_subplot(313, sharex=self.ax1)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initial dummy plot
        self.plot_empty()

    def plot_empty(self):
        """Show empty plot initially."""
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.text(0.5, 0.5, "Load CSV to visualize", transform=ax.transAxes, ha='center')
        self.canvas.draw()

    def load_csv(self):
        """Open file dialog and load CSV."""
        filetypes = [("CSV Files", "*.csv"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(title="Select Log CSV", filetypes=filetypes)
        
        if not path:
            return

        try:
            # Load CSV
            self.df = pd.read_csv(path)
            
            # Check basic columns (flexible check)
            required_columns = ['frame_index', 'tof_front', 'alpha', 'steer_final']
            if not all(col in self.df.columns for col in required_columns):
                messagebox.showerror("Error", "CSV format mismatch. Missing required columns.")
                return

            self.file_path = path
            self.lbl_filename.config(text=f"Loaded: {path.split('/')[-1]}")
            self.btn_save.config(state=tk.NORMAL)
            self.btn_toggle_x.config(state=tk.NORMAL)

            # Set X-Axis mode to Frame initially
            self.x_axis_mode = "frame"
            self.btn_toggle_x.config(text="X-Axis: Frame")
            self.lbl_slider.config(text="Frame Range:")
            
            self.reset_sliders()
            self.update_plot()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def toggle_x_axis(self):
        """Toggle between Time and Frame Index for X-Axis, maintaining zoom range."""
        if self.df is None:
            return

        # 1. 現在のスライダー位置（範囲）を取得
        current_min = self.slider_min.get()
        current_max = self.slider_max.get()
        
        new_min_val = 0
        new_max_val = 0

        # 2. 現在の軸モードに基づいて、対応するもう一方の値をデータから検索して変換
        if self.x_axis_mode == "time":
            # Time -> Frame へ切り替え
            # 現在の time に最も近い行を探す
            idx_start = (self.df['time'] - current_min).abs().idxmin()
            idx_end = (self.df['time'] - current_max).abs().idxmin()
            
            new_min_val = self.df.loc[idx_start, 'frame_index']
            new_max_val = self.df.loc[idx_end, 'frame_index']
            
            # モード変更
            self.x_axis_mode = "frame"
            self.btn_toggle_x.config(text="X-Axis: Frame")
            self.lbl_slider.config(text="Frame Range:")
            
        else: # Current is Frame
            # Frame -> Time へ切り替え
            if 'time' not in self.df.columns:
                messagebox.showerror("Error", "'time' column not found in CSV.")
                return
            
            # 現在の frame_index に最も近い行を探す
            idx_start = (self.df['frame_index'] - current_min).abs().idxmin()
            idx_end = (self.df['frame_index'] - current_max).abs().idxmin()
            
            new_min_val = self.df.loc[idx_start, 'time']
            new_max_val = self.df.loc[idx_end, 'time']

            # モード変更
            self.x_axis_mode = "time"
            self.btn_toggle_x.config(text="X-Axis: Time")
            self.lbl_slider.config(text="Time Range [s]:")

        # 3. スライダーの全体範囲設定 (Resolutionの設定)
        target_col = 'time' if self.x_axis_mode == 'time' else 'frame_index'
        max_limit = self.df[target_col].max()
        res = 0.1 if self.x_axis_mode == 'time' else 1
        
        self.slider_min.config(from_=0, to=max_limit, resolution=res)
        self.slider_max.config(from_=0, to=max_limit, resolution=res)
        
        # 4. 変換した値をスライダーにセット（これで位置が維持される）
        self.slider_min.set(new_min_val)
        self.slider_max.set(new_max_val)
            
        self.update_plot()

    def reset_sliders(self):
        """Reset sliders to full range."""
        if self.df is None:
            return
            
        target_col = 'time' if self.x_axis_mode == 'time' else 'frame_index'
        max_val = self.df[target_col].max()
        
        self.slider_min.config(from_=0, to=max_val, resolution=0.1 if self.x_axis_mode == 'time' else 1)
        self.slider_max.config(from_=0, to=max_val, resolution=0.1 if self.x_axis_mode == 'time' else 1)
        
        self.slider_min.set(0)
        self.slider_max.set(max_val)

    def update_plot_event(self, val):
        if self.df is not None:
            self.update_plot()

    def update_plot(self):
        """Draw plots based on current data and slider values."""
        if self.df is None:
            return

        # Determine X-Axis Column
        x_col = 'time' if self.x_axis_mode == 'time' else 'frame_index'

        # Filter data by range
        v_start = self.slider_min.get()
        v_end = self.slider_max.get()
        
        if v_start >= v_end:
            return

        mask = (self.df[x_col] >= v_start) & (self.df[x_col] <= v_end)
        data = self.df.loc[mask]

        if data.empty:
            return

        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        x_data = data[x_col]

        # --- 1. Sensors (Top) ---
        self.ax1.plot(x_data, data['tof_front'], label='Front', color='#1f77b4', linewidth=1.5)
        self.ax1.plot(x_data, data['tof_left'], label='Left', color="#f32bfd", linewidth=1.0)
        self.ax1.plot(x_data, data['tof_right'], label='Right', color='#ff7f0e',  linewidth=1.0)
        
        # Threshold Lines
        self.ax1.axhline(y=D_SAFE_FRONT, color='red', linestyle=':', alpha=0.8, label=f'$d_{{safe,front}}$={D_SAFE_FRONT}')
        self.ax1.axhline(y=D_SAFE_SIDE, color='green', linestyle='--', alpha=0.8, label=f'$d_{{safe,side}}$={D_SAFE_SIDE}')
        self.ax1.axhline(y=D_CRIT, color='black', linestyle='-.', alpha=0.6, label=f'$d_{{crit}}$={D_CRIT}')
        
        self.ax1.set_ylabel('ToF Distance [cm]')
        self.ax1.set_ylim(0, 150)  # Y-Axis Limited to 0-200
        self.ax1.set_title('(a) Sensor Measurements', loc='left', fontsize=10, fontweight='bold')
        self.ax1.legend(loc='upper right', fontsize='small', framealpha=0.9, ncol=2)
        self.ax1.grid(True, linestyle=':', alpha=0.6)

        # --- 2. Risks & Alpha (Middle) ---
        self.ax2.plot(x_data, data['risk_static'], label='Static Risk ($D_S$)', color='gray', linestyle=':', linewidth=1.0)
        
        # Fill Dynamic Risk
        self.ax2.fill_between(x_data, 0, data['risk_dynamic'], color='#ff7f0e', alpha=0.3, label='Dynamic Risk ($D_D$)')
        
        # Alpha
        self.ax2.plot(x_data, data['alpha'], label='Intervention Rate ($\\alpha$)', color='#d62728', linewidth=1.0)
        
        self.ax2.set_ylabel('Rate / Risk')
        self.ax2.set_ylim(-0.1, 1.1)
        self.ax2.set_title('(b) Risk & Intervention Rate', loc='left', fontsize=10, fontweight='bold')
        self.ax2.legend(loc='upper right', fontsize='small', framealpha=0.9)
        self.ax2.grid(True, linestyle=':', alpha=0.6)

        # --- 3. Steering (Bottom) ---
        self.ax3.plot(x_data, data['steer_ai'], label='AI ($\\theta_{AI}$)', color='#1f77b4', linestyle='--', linewidth=1.5)
        self.ax3.plot(x_data, data['steer_final'], label='Final ($\\theta_{final}$)', color='#2ca02c', linewidth=2.0)
        
        if 'steer_avoid' in data.columns:
             self.ax3.plot(x_data, data['steer_avoid'], label='Avoid ($\\theta_{Avoid}$)', color='gray', linestyle=':', alpha=0.5)

        self.ax3.set_ylabel('Steering Angle\n(-1.0 to 1.0)')
        
        if self.x_axis_mode == 'time':
            self.ax3.set_xlabel('Time [s]')
        else:
            self.ax3.set_xlabel('Frame Index')

        self.ax3.set_ylim(-1.2, 1.2)
        self.ax3.set_title('(c) Steering Control', loc='left', fontsize=10, fontweight='bold')
        self.ax3.legend(loc='lower right', fontsize='small', framealpha=0.9)
        self.ax3.grid(True, linestyle=':', alpha=0.6)

        self.fig.tight_layout()
        self.canvas.draw()

    def save_graph(self):
        if self.df is None:
            return
        filetypes = [("PNG Image", "*.png"), ("PDF Document", "*.pdf")]
        path = filedialog.asksaveasfilename(title="Save Graph", filetypes=filetypes, defaultextension=".png")
        if path:
            self.fig.savefig(path, dpi=300)
            messagebox.showinfo("Success", f"Graph saved to:\n{path}")

if __name__ == "__main__":
    try:
        import pandas
        import matplotlib
    except ImportError:
        print("Required libraries missing. Run: pip install pandas matplotlib")
        exit()

    root = tk.Tk()
    app = LogVisualizerApp(root)
    root.mainloop()