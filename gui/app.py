# gui/app.py
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import *
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
from core.face_swapper import get_swapper
from utils.model_downloader import ensure_models

# Constants
VERSION = "1.0.0"
WINDOW_TITLE = f"FaceReenact-Pro v{VERSION} - Advanced Face Reenactment"

class FaceReenactApp(Tk):
    def __init__(self):
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry("1100x700")
        self.configure(bg="#1e1e1e")
        self.source_path = None
        self.target_path = None
        self.output_path = None
        self.swap_thread = None

        # Ensure models are ready
        if not os.path.exists("models/inswapper_128.onnx"):
            messagebox.showinfo("First Run", "Models are being downloaded... This happens only once.")
            threading.Thread(target=ensure_models, daemon=True).start()

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=10, font=('Helvetica', 10, 'bold'))

        # Header
        header = tk.Label(self, text="FaceReenact-Pro", font=("Helvetica", 28, "bold"),
                          fg="#00ff88", bg="#1e1e1e")
        header.pack(pady=15)

        # Main frame
        main_frame = tk.Frame(self, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left panel - Source
        left = tk.LabelFrame(main_frame, text=" Source Face (Drag & Drop)", fg="white", bg="#2d2d2d",
                             font=("Helvetica", 12, "bold"), padx=10, pady=10)
        left.grid(row=0, column=0, padx=15, pady=10, sticky="nsew")

        self.source_label = tk.Label(left, text="Drop image here\nor click to select", bg="#3d3d3d",
                                     fg="#cccccc", font=("Helvetica", 14), width=35, height=18,
                                     relief="sunken", bd=3)
        self.source_label.pack(pady=10)
        self.source_label.drop_target_register(DND_FILES)
        self.source_label.dnd_bind('<<Drop>>', lambda e: self.load_source(e.data))

        self.source_label.bind("<Button-1>", lambda e: self.browse_source())

        # Right panel - Target
        right = tk.LabelFrame(main_frame, text=" Target Image/Video (Drag & Drop)", fg="white", bg="#2d2d2d",
                              font=("Helvetica", 12, "bold"), padx=10, pady=10)
        right.grid(row=0, column=1, padx=15, pady=10, sticky="nsew")

        self.target_label = tk.Label(right, text="Drop file here\nor click to select", bg="#3d3d3d",
                                     fg="#cccccc", font=("Helvetica", 14), width=35, height=18,
                                     relief="sunken", bd=3)
        self.target_label.pack(pady=10)
        self.target_label.drop_target_register(DND_FILES)
        self.target_label.dnd_bind('<<Drop>>', lambda e: self.load_target(e.data))

        self.target_label.bind("<Button-1>", lambda e: self.browse_target())

        # Preview area
        preview_frame = tk.LabelFrame(self, text=" Preview", fg="#00ff88", bg="#1e1e1e", font=("Helvetica", 12, "bold"))
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.preview_canvas = tk.Label(preview_frame, bg="#000000")
        self.preview_canvas.pack(pady=10)

        # Progress + Buttons
        bottom = tk.Frame(self, bg="#1e1e1e")
        bottom.pack(fill="x", padx=20, pady=15)

        self.progress = ttk.Progressbar(bottom, mode='determinate', length=600)
        self.progress.pack(pady=10)

        self.status_label = tk.Label(bottom, text="Ready", fg="#00ff88", bg="#1e1e1e", font=("Helvetica", 11))
        self.status_label.pack(pady=5)

        btn_frame = tk.Frame(bottom, bg="#1e1e1e")
        btn_frame.pack(pady=10)

        self.swap_btn = ttk.Button(btn_frame, text="START FACE REENACTMENT", command=self.start_swap)
        self.swap_btn.pack(side="left", padx=15)

        clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_all)
        clear_btn.pack(side="left", padx=15)

        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

    def load_image_preview(self, path, label):
        if not os.path.exists(path):
            return
        img = Image.open(path)
        img = img.resize((280, 280), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.configure(image=photo, text="")
        label.image = photo

    def load_source(self, data):
        path = self.clean_path(data)
        if path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'webp')):
            self.source_path = path
            self.load_image_preview(path, self.source_label)
            self.status_label.config(text=f"Source loaded: {os.path.basename(path)}")

    def load_target(self, data):
        path = self.clean_path(data)
        if path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'webp', 'mp4', 'mov', 'avi', 'mkv')):
            self.target_path = path
            if path.lower().endswith(('mp4', 'mov', 'avi', 'mkv')):
                self.target_label.config(text="VIDEO LOADED\n" + os.path.basename(path), image="")
            else:
                self.load_image_preview(path, self.target_label)
            self.status_label.config(text=f"Target loaded: {os.path.basename(path)}")

    def clean_path(self, raw):
        path = raw.strip('{}')
        if path.startswith("file://"):
            path = path[7:]
        return path.replace("\\", "/")

    def browse_source(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if path:
            self.load_source(path)

    def browse_target(self):
        path = filedialog.askopenfilename(filetypes=[("Images & Videos", "*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.mov *.avi *.mkv")])
        if path:
            self.load_target(path)

    def clear_all(self):
        self.source_path = self.target_path = None
        self.source_label.config(image="", text="Drop image here\nor click to select")
        self.target_label.config(image="", text="Drop file here\nor click to select")
        self.preview_canvas.config(image="")
        self.progress['value'] = 0
        self.status_label.config(text="Cleared")

    def start_swap(self):
        if not self.source_path or not self.target_path:
            messagebox.showwarning("Missing Files", "Please select both source face and target file!")
            return

        self.output_path = filedialog.asksaveasfilename(
                        defaultextension=".mp4" if self.target_path.lower().endswith(('mp4', 'mov', 'avi', 'mkv')) else ".png",
            initialfile="output.png" if not self.target_path.lower().endswith(('mp4', 'mov', 'avi', 'mkv')) else "output_video.mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("JPEG Image", "*.jpg"), ("PNG Image", "*.png")])

        if not self.output_path:
            return

        self.swap_btn.config(state="disabled")
        self.status_label.config(text="Processing... Please wait")
        self.progress['value'] = 0

        self.swap_thread = threading.Thread(target=self.process_swap, daemon=True)
        self.swap_thread.start()

    def process_swap(self):
        try:
            swapper = get_swapper()

            if self.target_path.lower().endswith(('mp4', 'mov', 'avi', 'mkv')):
                # Video processing
                self.after(0, lambda: self.status_label.config(text="Processing video frames..."))
                swapper.process_video(
                    self.source_path,
                    self.target_path,
                    self.output_path,
                    callback=lambda cur, total: self.after(0, lambda: self.progress.config(value=cur/total*100))
                )
            else:
                # Image processing
                source_img = cv2.imread(self.source_path)
                target_img = cv2.imread(self.target_path)
                result = swapper.swap_faces(source_img, target_img)

                # Force correct extension (OpenCV is picky on Windows)
                ext = os.path.splitext(self.output_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    save_path = os.path.splitext(self.output_path)[0] + '.jpg'
                elif ext == '.png':
                    save_path = self.output_path
                else:
                    save_path = os.path.splitext(self.output_path)[0] + '.jpg'  # default to jpg

                success = cv2.imwrite(save_path, result)
                if not success:
                    # Fallback: use PIL (100% reliable)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(result_rgb)
                    img.save(save_path)
                self.output_path = save_path  # update path for preview

            self.after(0, self.on_complete)
        except Exception as e:
            error_msg = str(e)
            self.after(0, lambda: messagebox.showerror("Error", f"Failed: {error_msg}"))
            self.after(0, lambda: self.swap_btn.config(state="normal"))
            self.after(0, lambda: self.status_label.config(text="Error occurred", fg="red"))

    def on_complete(self):
        self.progress['value'] = 100
        self.status_label.config(text=f"Completed! Saved to:\n{os.path.basename(self.output_path)}", fg="#00ff88")
        self.swap_btn.config(state="normal")
        messagebox.showinfo("Success!", f"Face reenactment completed!\nSaved: {self.output_path}")

        # Show preview
        if self.output_path.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'webp')):
            img = Image.open(self.output_path)
            img = img.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.preview_canvas.config(image=photo)
            self.preview_canvas.image = photo


if __name__ == "__main__":
    # Required for tkinterdnd2
    app = FaceReenactApp()
    app.mainloop()
