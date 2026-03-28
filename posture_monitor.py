"""
PostureCoach — Real-Time Desktop Monitor
=========================================
High-performance Tkinter GUI with direct YOLO inference.
No web server needed — runs inference directly on camera frames.

Usage:
    python posture_monitor.py
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    print("Error: tkinter is required. It should come with Python.")
    sys.exit(1)

from ultralytics import YOLO

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

# ── Configuration ────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONF_THRESHOLD = 0.25
INFERENCE_SIZE = 640


class PostureMonitorApp:
    """Real-time posture monitoring with Tkinter + YOLOv8 pose."""

    def __init__(self, root):
        self.root = root
        self.root.title("PostureCoach — Real-Time Monitor")
        self.root.configure(bg="#0d0d14")
        self.root.minsize(1100, 700)

        # ── State ────────────────────────────────────────────────
        self.running = False
        self.camera = None
        self.model = None
        self.model_loaded = False
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        self.last_fps_count = 0

        # Session stats
        self.total_detections = 0
        self.good_count = 0
        self.bad_count = 0
        self.confidences = []
        self.log_entries = []

        # ── Colors ───────────────────────────────────────────────
        self.BG = "#0d0d14"
        self.BG_CARD = "#161622"
        self.BG_CARD2 = "#1a1a2e"
        self.BORDER = "#2a2a40"
        self.TEXT = "#e8e8ef"
        self.TEXT_DIM = "#6b6b80"
        self.GOOD = "#34d399"
        self.BAD = "#f87171"
        self.ACCENT = "#8b5cf6"
        self.ACCENT2 = "#6366f1"

        # ── Style ────────────────────────────────────────────────
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.BG_CARD)
        style.configure("Card2.TFrame", background=self.BG_CARD2)

        # ── Build UI ─────────────────────────────────────────────
        self._build_ui()

        # ── Load model in background ─────────────────────────────
        self._log("System", "Starting PostureCoach Monitor...")
        self._log("System", f"Model: {MODEL_PATH}")
        threading.Thread(target=self._load_model, daemon=True).start()

        # ── Window close handler ─────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════

    def _build_ui(self):
        """Build the complete user interface."""
        # Main container
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ── Header ───────────────────────────────────────────────
        header = tk.Frame(main, bg=self.BG)
        header.pack(fill=tk.X, pady=(0, 12))

        tk.Label(header, text="⦿ PostureCoach", font=("Segoe UI", 18, "bold"),
                 fg=self.GOOD, bg=self.BG).pack(side=tk.LEFT)

        self.status_label = tk.Label(header, text="● Loading Model...",
                                     font=("Segoe UI", 10), fg="#f59e0b", bg=self.BG)
        self.status_label.pack(side=tk.RIGHT)

        self.fps_label = tk.Label(header, text="FPS: --",
                                  font=("Consolas", 10), fg=self.TEXT_DIM, bg=self.BG)
        self.fps_label.pack(side=tk.RIGHT, padx=(0, 20))

        # ── Content area (video + sidebar) ───────────────────────
        content = tk.Frame(main, bg=self.BG)
        content.pack(fill=tk.BOTH, expand=True)

        # LEFT: Video feed
        video_frame = tk.Frame(content, bg=self.BG_CARD, highlightbackground=self.BORDER,
                               highlightthickness=1, padx=2, pady=2)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self.video_label = tk.Label(video_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Video controls bar
        controls = tk.Frame(video_frame, bg=self.BG_CARD2, padx=12, pady=8)
        controls.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_start = tk.Button(
            controls, text="▶  Start Camera", font=("Segoe UI", 10, "bold"),
            bg=self.ACCENT, fg="white", activebackground=self.ACCENT2,
            activeforeground="white", relief=tk.FLAT, padx=16, pady=6,
            cursor="hand2", command=self._toggle_camera
        )
        self.btn_start.pack(side=tk.LEFT)

        # Confidence threshold slider
        tk.Label(controls, text="Confidence:", font=("Segoe UI", 9),
                 fg=self.TEXT_DIM, bg=self.BG_CARD2).pack(side=tk.LEFT, padx=(20, 5))

        self.conf_var = tk.DoubleVar(value=CONF_THRESHOLD)
        self.conf_slider = tk.Scale(
            controls, from_=0.1, to=0.9, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.conf_var,
            bg=self.BG_CARD2, fg=self.TEXT, troughcolor=self.BG,
            highlightthickness=0, sliderrelief=tk.FLAT, length=120,
            font=("Consolas", 8)
        )
        self.conf_slider.pack(side=tk.LEFT)

        # RIGHT: Sidebar
        sidebar = tk.Frame(content, bg=self.BG, width=320)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # ── Posture Status Card ──────────────────────────────────
        status_card = tk.Frame(sidebar, bg=self.BG_CARD, highlightbackground=self.BORDER,
                               highlightthickness=1, padx=16, pady=16)
        status_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(status_card, text="POSTURE STATUS", font=("Segoe UI", 8, "bold"),
                 fg=self.TEXT_DIM, bg=self.BG_CARD, anchor="w").pack(fill=tk.X)

        self.posture_label = tk.Label(
            status_card, text="--", font=("Segoe UI", 28, "bold"),
            fg=self.TEXT_DIM, bg=self.BG_CARD
        )
        self.posture_label.pack(pady=(8, 4))

        self.confidence_label = tk.Label(
            status_card, text="Confidence: --", font=("Segoe UI", 12),
            fg=self.TEXT_DIM, bg=self.BG_CARD
        )
        self.confidence_label.pack()

        # Confidence bar
        bar_frame = tk.Frame(status_card, bg=self.BG, height=8)
        bar_frame.pack(fill=tk.X, pady=(10, 0))
        bar_frame.pack_propagate(False)

        self.conf_bar = tk.Frame(bar_frame, bg=self.TEXT_DIM, width=0, height=8)
        self.conf_bar.pack(side=tk.LEFT, fill=tk.Y)

        # ── Session Stats ────────────────────────────────────────
        stats_card = tk.Frame(sidebar, bg=self.BG_CARD, highlightbackground=self.BORDER,
                              highlightthickness=1, padx=16, pady=12)
        stats_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(stats_card, text="SESSION STATS", font=("Segoe UI", 8, "bold"),
                 fg=self.TEXT_DIM, bg=self.BG_CARD, anchor="w").pack(fill=tk.X, pady=(0, 8))

        stats_grid = tk.Frame(stats_card, bg=self.BG_CARD)
        stats_grid.pack(fill=tk.X)

        self.stat_labels = {}
        stats_data = [
            ("Total:", "total", self.TEXT),
            ("Good:", "good", self.GOOD),
            ("Bad:", "bad", self.BAD),
            ("Avg Conf:", "avgconf", self.ACCENT),
        ]

        for i, (label, key, color) in enumerate(stats_data):
            row = i // 2
            col = i % 2
            frame = tk.Frame(stats_grid, bg=self.BG_CARD)
            frame.grid(row=row, column=col, sticky="ew", padx=4, pady=3)
            stats_grid.columnconfigure(col, weight=1)

            tk.Label(frame, text=label, font=("Segoe UI", 9),
                     fg=self.TEXT_DIM, bg=self.BG_CARD).pack(side=tk.LEFT)
            val_label = tk.Label(frame, text="0", font=("Segoe UI", 11, "bold"),
                                 fg=color, bg=self.BG_CARD)
            val_label.pack(side=tk.RIGHT)
            self.stat_labels[key] = val_label

        # ── Detection Log (Serial Monitor) ───────────────────────
        log_card = tk.Frame(sidebar, bg=self.BG_CARD, highlightbackground=self.BORDER,
                            highlightthickness=1, padx=12, pady=10)
        log_card.pack(fill=tk.BOTH, expand=True)

        log_header = tk.Frame(log_card, bg=self.BG_CARD)
        log_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(log_header, text="DETECTION LOG", font=("Segoe UI", 8, "bold"),
                 fg=self.TEXT_DIM, bg=self.BG_CARD).pack(side=tk.LEFT)

        tk.Button(log_header, text="Clear", font=("Segoe UI", 7),
                  bg=self.BG_CARD2, fg=self.TEXT_DIM, relief=tk.FLAT,
                  cursor="hand2", command=self._clear_log).pack(side=tk.RIGHT)

        self.log_text = tk.Text(
            log_card, bg="#0a0a12", fg=self.TEXT, font=("Consolas", 9),
            relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED,
            insertbackground=self.TEXT, selectbackground=self.ACCENT,
            padx=8, pady=6, height=12
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for colored log entries
        self.log_text.tag_configure("time", foreground=self.TEXT_DIM)
        self.log_text.tag_configure("system", foreground="#f59e0b")
        self.log_text.tag_configure("good", foreground=self.GOOD)
        self.log_text.tag_configure("bad", foreground=self.BAD)
        self.log_text.tag_configure("info", foreground=self.ACCENT)

    # ═══════════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ═══════════════════════════════════════════════════════════════

    def _load_model(self):
        """Load the YOLO model in a background thread."""
        try:
            self._log("System", "Loading YOLOv8 pose model...")
            self.model = YOLO(str(MODEL_PATH), task="pose")

            # Warm up with a dummy frame
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=INFERENCE_SIZE, conf=0.25, verbose=False)

            self.model_loaded = True
            self._log("System", f"✓ Model loaded successfully")
            self._log("Info", f"Classes: {self.model.names}")
            self.root.after(0, lambda: self.status_label.config(
                text="● Model Ready", fg=self.GOOD
            ))
        except Exception as e:
            self._log("System", f"✗ Model load failed: {e}")
            self.root.after(0, lambda: self.status_label.config(
                text="● Model Error", fg=self.BAD
            ))

    # ═══════════════════════════════════════════════════════════════
    #  CAMERA CONTROL
    # ═══════════════════════════════════════════════════════════════

    def _toggle_camera(self):
        """Start or stop the camera feed."""
        if self.running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        """Initialize camera and begin frame processing."""
        if not self.model_loaded:
            self._log("System", "Model not loaded yet. Please wait...")
            return

        self._log("System", "Starting camera...")
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for less latency

        if not self.camera.isOpened():
            self._log("System", "✗ Failed to open camera!")
            return

        self.running = True
        self.btn_start.config(text="■  Stop Camera", bg=self.BAD)
        self.status_label.config(text="● LIVE", fg=self.BAD)
        self.last_fps_time = time.time()
        self.last_fps_count = 0

        self._log("System", "✓ Camera started — analyzing posture in real-time")
        self._process_frame()

    def _stop_camera(self):
        """Stop the camera feed."""
        self.running = False
        if self.camera:
            self.camera.release()
            self.camera = None

        self.btn_start.config(text="▶  Start Camera", bg=self.ACCENT)
        self.status_label.config(text="● Model Ready", fg=self.GOOD)
        self.fps_label.config(text="FPS: --")
        self._log("System", "Camera stopped")

    def _process_frame(self):
        """Capture a frame, run inference, and display results."""
        if not self.running or not self.camera:
            return

        ret, frame = self.camera.read()
        if not ret:
            self._log("System", "Frame capture failed")
            self.root.after(30, self._process_frame)
            return

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # Run YOLO inference
        conf = self.conf_var.get()
        results = self.model.predict(
            frame,
            imgsz=INFERENCE_SIZE,
            conf=conf,
            verbose=False,
        )

        # Process results
        result = results[0]
        annotated = result.plot()  # Get frame with skeleton drawn

        # Extract posture classification
        if result.boxes and len(result.boxes) > 0:
            best_idx = result.boxes.conf.argmax().item()
            best_conf = result.boxes.conf[best_idx].item()
            best_cls = int(result.boxes.cls[best_idx].item())
            class_name = self.model.names[best_cls]

            self._update_posture(class_name, best_conf)
        else:
            self._update_posture(None, 0)

        # Calculate FPS
        self.frame_count += 1
        self.last_fps_count += 1
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 0.5:
            self.fps = self.last_fps_count / elapsed
            self.last_fps_count = 0
            self.last_fps_time = now
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")

        # Display the annotated frame in the Tkinter label
        self._display_frame(annotated)

        # Schedule next frame (aim for ~30fps but let inference be the bottleneck)
        self.root.after(1, self._process_frame)

    def _display_frame(self, frame):
        """Convert OpenCV frame to Tkinter-compatible image and display."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the label size and resize frame to fit
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w > 1 and label_h > 1:
            h, w = frame_rgb.shape[:2]
            scale = min(label_w / w, label_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self.video_label.imgtk = imgtk  # Prevent garbage collection

    # ═══════════════════════════════════════════════════════════════
    #  POSTURE UPDATES
    # ═══════════════════════════════════════════════════════════════

    def _update_posture(self, class_name, confidence):
        """Update the posture status display and log."""
        if class_name is None:
            self.posture_label.config(text="No Person", fg=self.TEXT_DIM)
            self.confidence_label.config(text="Confidence: --")
            self.conf_bar.config(width=0)
            return

        is_good = class_name.lower() == "good"
        conf_pct = confidence * 100

        # Update posture label
        if is_good:
            self.posture_label.config(text="✓ GOOD", fg=self.GOOD)
        else:
            self.posture_label.config(text="✗ BAD", fg=self.BAD)

        self.confidence_label.config(
            text=f"Confidence: {conf_pct:.1f}%",
            fg=self.GOOD if is_good else self.BAD
        )

        # Update confidence bar
        bar_width = int(3.0 * conf_pct)  # Scale to max ~300px
        self.conf_bar.config(
            width=min(bar_width, 300),
            bg=self.GOOD if is_good else self.BAD
        )

        # Update session stats
        self.total_detections += 1
        if is_good:
            self.good_count += 1
        else:
            self.bad_count += 1
        self.confidences.append(confidence)

        self.stat_labels["total"].config(text=str(self.total_detections))
        self.stat_labels["good"].config(text=str(self.good_count))
        self.stat_labels["bad"].config(text=str(self.bad_count))

        avg = sum(self.confidences[-100:]) / len(self.confidences[-100:])
        self.stat_labels["avgconf"].config(text=f"{avg * 100:.1f}%")

        # Log every 5th detection to avoid flooding the log
        if self.total_detections % 5 == 1:
            tag = "good" if is_good else "bad"
            self._log(tag.capitalize(), f"{class_name} posture ({conf_pct:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    #  LOGGING
    # ═══════════════════════════════════════════════════════════════

    def _log(self, level, message):
        """Add a timestamped entry to the detection log."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        tag_map = {
            "System": "system",
            "Info": "info",
            "Good": "good",
            "Bad": "bad",
        }
        tag = tag_map.get(level, "info")

        def _append():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{timestamp}] ", "time")
            self.log_text.insert(tk.END, f"{message}\n", tag)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        # Thread-safe UI update
        self.root.after(0, _append)

    def _clear_log(self):
        """Clear the detection log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)
        self._log("System", "Log cleared")

    # ═══════════════════════════════════════════════════════════════
    #  CLEANUP
    # ═══════════════════════════════════════════════════════════════

    def _on_close(self):
        """Handle window close."""
        self.running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Make sure the trained model is in the 'models/' directory.")
        sys.exit(1)

    root = tk.Tk()
    root.state("zoomed")  # Start maximized on Windows
    app = PostureMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
