import tkinter as tk
from tkinter import filedialog, colorchooser, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import threading


class BackgroundRemoverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Remover")
        self.root.minsize(900, 700)

        self.source_image = None
        self.result_image = None
        self.preview_photo = None
        self.bg_color = (255, 255, 255)

        # Eyedropper state
        self.eyedropper_active = False

        # Preview mapping (to translate canvas coords → image coords)
        self._preview_offset = (0, 0)
        self._preview_scale = 1.0
        self._preview_img = None  # which image is currently shown

        self._build_ui()

    def _build_ui(self):
        # --- Mode tabs ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

        # == Tab 1: Manual color-based removal ==
        manual_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(manual_frame, text="  Manual (Color)  ")

        # File selection
        file_row = ttk.Frame(manual_frame)
        file_row.pack(fill=tk.X, pady=2)
        ttk.Button(file_row, text="Open Image…", command=self._open_file).pack(side=tk.LEFT, padx=4)
        self.file_label = ttk.Label(file_row, text="No file selected", foreground="gray")
        self.file_label.pack(side=tk.LEFT, padx=4)

        # Color row
        color_row = ttk.Frame(manual_frame)
        color_row.pack(fill=tk.X, pady=6)
        ttk.Label(color_row, text="BG Color:").pack(side=tk.LEFT, padx=4)
        self.color_swatch = tk.Canvas(color_row, width=28, height=28, bg="#ffffff",
                                      highlightthickness=1, highlightbackground="gray")
        self.color_swatch.pack(side=tk.LEFT, padx=4)
        ttk.Button(color_row, text="Pick…", command=self._pick_color).pack(side=tk.LEFT, padx=4)

        self.eyedropper_btn = ttk.Button(color_row, text="Eyedropper", command=self._toggle_eyedropper)
        self.eyedropper_btn.pack(side=tk.LEFT, padx=4)
        self.eyedropper_status = ttk.Label(color_row, text="", foreground="blue")
        self.eyedropper_status.pack(side=tk.LEFT, padx=4)

        self.color_value_label = ttk.Label(color_row, text="RGB(255, 255, 255)")
        self.color_value_label.pack(side=tk.LEFT, padx=8)

        # Tolerance
        tol_row = ttk.Frame(manual_frame)
        tol_row.pack(fill=tk.X, pady=2)
        ttk.Label(tol_row, text="Tolerance:").pack(side=tk.LEFT, padx=4)
        self.tolerance_var = tk.IntVar(value=30)
        ttk.Scale(tol_row, from_=0, to=255, variable=self.tolerance_var,
                  orient=tk.HORIZONTAL, length=300, command=self._on_slider_change).pack(side=tk.LEFT, padx=4)
        self.tolerance_label = ttk.Label(tol_row, text="30", width=4)
        self.tolerance_label.pack(side=tk.LEFT, padx=4)

        # Edge softness
        soft_row = ttk.Frame(manual_frame)
        soft_row.pack(fill=tk.X, pady=2)
        ttk.Label(soft_row, text="Edge Softness:").pack(side=tk.LEFT, padx=4)
        self.softness_var = tk.IntVar(value=0)
        ttk.Scale(soft_row, from_=0, to=50, variable=self.softness_var,
                  orient=tk.HORIZONTAL, length=300, command=self._on_slider_change).pack(side=tk.LEFT, padx=4)
        self.softness_label = ttk.Label(soft_row, text="0", width=4)
        self.softness_label.pack(side=tk.LEFT, padx=4)

        # Action buttons
        action_row = ttk.Frame(manual_frame)
        action_row.pack(fill=tk.X, pady=8)
        ttk.Button(action_row, text="Remove Background", command=self._process_manual).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_row, text="Save Result…", command=self._save_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_row, text="Reset", command=self._reset_preview).pack(side=tk.LEFT, padx=4)

        # == Tab 2: Auto AI-based detection ==
        auto_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(auto_frame, text="  Auto Detect (AI)  ")

        # File selection (shared state, separate button)
        file_row2 = ttk.Frame(auto_frame)
        file_row2.pack(fill=tk.X, pady=2)
        ttk.Button(file_row2, text="Open Image…", command=self._open_file).pack(side=tk.LEFT, padx=4)
        self.file_label2 = ttk.Label(file_row2, text="No file selected", foreground="gray")
        self.file_label2.pack(side=tk.LEFT, padx=4)

        # Model selection
        model_row = ttk.Frame(auto_frame)
        model_row.pack(fill=tk.X, pady=6)
        ttk.Label(model_row, text="Model:").pack(side=tk.LEFT, padx=4)
        self.model_var = tk.StringVar(value="u2net")
        model_combo = ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", width=20,
                                   values=["u2net", "u2netp", "u2net_human_seg", "isnet-general-use", "silueta"])
        model_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(model_row, text="(u2net = best quality, u2netp = faster)", foreground="gray").pack(
            side=tk.LEFT, padx=4)

        # Alpha matting
        matting_row = ttk.Frame(auto_frame)
        matting_row.pack(fill=tk.X, pady=2)
        self.alpha_matting_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(matting_row, text="Alpha matting (smoother edges, slower)",
                        variable=self.alpha_matting_var).pack(side=tk.LEFT, padx=4)

        # Foreground / background threshold (for alpha matting)
        mat_sliders = ttk.Frame(auto_frame)
        mat_sliders.pack(fill=tk.X, pady=2)
        ttk.Label(mat_sliders, text="FG threshold:").pack(side=tk.LEFT, padx=4)
        self.fg_thresh_var = tk.IntVar(value=240)
        ttk.Scale(mat_sliders, from_=0, to=255, variable=self.fg_thresh_var,
                  orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT, padx=4)
        self.fg_thresh_label = ttk.Label(mat_sliders, text="240", width=4)
        self.fg_thresh_label.pack(side=tk.LEFT)

        ttk.Label(mat_sliders, text="BG threshold:").pack(side=tk.LEFT, padx=(16, 4))
        self.bg_thresh_var = tk.IntVar(value=10)
        ttk.Scale(mat_sliders, from_=0, to=255, variable=self.bg_thresh_var,
                  orient=tk.HORIZONTAL, length=140).pack(side=tk.LEFT, padx=4)
        self.bg_thresh_label = ttk.Label(mat_sliders, text="10", width=4)
        self.bg_thresh_label.pack(side=tk.LEFT)

        # Action buttons
        action_row2 = ttk.Frame(auto_frame)
        action_row2.pack(fill=tk.X, pady=8)
        self.auto_btn = ttk.Button(action_row2, text="Auto Detect Subject", command=self._process_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(action_row2, text="Save Result…", command=self._save_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_row2, text="Reset", command=self._reset_preview).pack(side=tk.LEFT, padx=4)

        # Status / progress
        self.status_label = ttk.Label(auto_frame, text="", foreground="gray")
        self.status_label.pack(fill=tk.X, pady=4)

        # --- Preview area (shared) ---
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

        self.canvas = tk.Canvas(preview_frame, bg="#cccccc", cursor="arrow")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)

    # ─── File I/O ────────────────────────────────────────────

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.source_image = Image.open(path).convert("RGBA")
        self.result_image = None
        short = path if len(path) < 60 else "…" + path[-55:]
        self.file_label.config(text=short, foreground="black")
        self.file_label2.config(text=short, foreground="black")
        self._show_preview(self.source_image)

    def _save_file(self):
        img = self.result_image or self.source_image
        if img is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")]
        )
        if path:
            img.save(path)

    def _reset_preview(self):
        self.result_image = None
        if self.source_image:
            self._show_preview(self.source_image)

    # ─── Eyedropper ──────────────────────────────────────────

    def _toggle_eyedropper(self):
        self.eyedropper_active = not self.eyedropper_active
        if self.eyedropper_active:
            self.canvas.config(cursor="crosshair")
            self.eyedropper_status.config(text="Click on the image to sample a color")
            self.eyedropper_btn.config(text="Cancel Eyedropper")
        else:
            self.canvas.config(cursor="arrow")
            self.eyedropper_status.config(text="")
            self.eyedropper_btn.config(text="Eyedropper")

    def _canvas_to_image_coords(self, cx, cy):
        """Convert canvas pixel position to source image pixel coords."""
        if self.source_image is None:
            return None
        ox, oy = self._preview_offset
        scale = self._preview_scale
        ix = int((cx - ox) / scale)
        iy = int((cy - oy) / scale)
        if 0 <= ix < self.source_image.width and 0 <= iy < self.source_image.height:
            return ix, iy
        return None

    def _on_canvas_click(self, event):
        if not self.eyedropper_active or self.source_image is None:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return
        ix, iy = coords
        # Sample from whichever image is currently displayed
        img = self._preview_img or self.source_image
        pixel = img.getpixel((ix, iy))
        self.bg_color = (pixel[0], pixel[1], pixel[2])
        hex_color = "#{:02x}{:02x}{:02x}".format(*self.bg_color)
        self.color_swatch.config(bg=hex_color)
        self.color_value_label.config(text=f"RGB({pixel[0]}, {pixel[1]}, {pixel[2]})")
        # Deactivate eyedropper after sampling
        self._toggle_eyedropper()

    def _on_canvas_motion(self, event):
        if not self.eyedropper_active or self.source_image is None:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            self.eyedropper_status.config(text="(outside image)")
            return
        ix, iy = coords
        img = self._preview_img or self.source_image
        pixel = img.getpixel((ix, iy))
        self.eyedropper_status.config(
            text=f"RGB({pixel[0]}, {pixel[1]}, {pixel[2]}) @ ({ix}, {iy}) — click to select"
        )

    # ─── Color picker dialog ────────────────────────────────

    def _pick_color(self):
        color = colorchooser.askcolor(initialcolor=self.bg_color, title="Pick background color")
        if color and color[0]:
            self.bg_color = tuple(int(c) for c in color[0])
            hex_color = "#{:02x}{:02x}{:02x}".format(*self.bg_color)
            self.color_swatch.config(bg=hex_color)
            self.color_value_label.config(text=f"RGB{self.bg_color}")

    # ─── Slider feedback ────────────────────────────────────

    def _on_slider_change(self, _event=None):
        self.tolerance_label.config(text=str(self.tolerance_var.get()))
        self.softness_label.config(text=str(self.softness_var.get()))
        self.fg_thresh_label.config(text=str(self.fg_thresh_var.get()))
        self.bg_thresh_label.config(text=str(self.bg_thresh_var.get()))

    # ─── Manual processing ──────────────────────────────────

    def _process_manual(self):
        if self.source_image is None:
            return

        data = np.array(self.source_image, dtype=np.float64)
        r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

        br, bg_, bb = self.bg_color
        distance = np.sqrt((r - br) ** 2 + (g - bg_) ** 2 + (b - bb) ** 2)

        tolerance = self.tolerance_var.get()
        softness = self.softness_var.get()

        if softness == 0:
            alpha = np.where(distance <= tolerance, 0.0, data[:, :, 3])
        else:
            inner = float(tolerance)
            outer = float(tolerance + softness)
            alpha = np.clip((distance - inner) / (outer - inner), 0.0, 1.0) * data[:, :, 3]

        result = data.copy()
        result[:, :, 3] = alpha
        self.result_image = Image.fromarray(result.astype(np.uint8))
        self._show_preview(self.result_image)

    # ─── Auto AI processing ─────────────────────────────────

    def _process_auto(self):
        if self.source_image is None:
            messagebox.showwarning("No image", "Open an image first.")
            return

        self.auto_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Loading AI model… (first run downloads ~170 MB)")

        def run():
            try:
                from rembg import remove, new_session

                model = self.model_var.get()
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Running {model} model…"))

                session = new_session(model)

                use_matting = self.alpha_matting_var.get()
                kwargs = {}
                if use_matting:
                    kwargs["alpha_matting"] = True
                    kwargs["alpha_matting_foreground_threshold"] = self.fg_thresh_var.get()
                    kwargs["alpha_matting_background_threshold"] = self.bg_thresh_var.get()

                result = remove(
                    self.source_image,
                    session=session,
                    **kwargs,
                )
                self.result_image = result.convert("RGBA")

                def finish():
                    self.status_label.config(text="Done!")
                    self.auto_btn.config(state=tk.NORMAL)
                    self._show_preview(self.result_image)

                self.root.after(0, finish)

            except ImportError:
                def show_err():
                    self.status_label.config(text="")
                    self.auto_btn.config(state=tk.NORMAL)
                    messagebox.showerror(
                        "Missing dependency",
                        "The 'rembg' package is required for auto detection.\n\n"
                        "Install it with:\n"
                        "  pip install rembg[cpu]\n\n"
                        "Or for GPU acceleration:\n"
                        "  pip install rembg[gpu]"
                    )
                self.root.after(0, show_err)

            except Exception as e:
                def show_err():
                    self.status_label.config(text="")
                    self.auto_btn.config(state=tk.NORMAL)
                    messagebox.showerror("Error", str(e))
                self.root.after(0, show_err)

        threading.Thread(target=run, daemon=True).start()

    # ─── Preview ─────────────────────────────────────────────

    def _show_preview(self, img):
        self._preview_img = img
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2:
            canvas_w, canvas_h = 880, 400

        checker = self._make_checkerboard(canvas_w, canvas_h, square=10)

        scale = min(canvas_w / img.width, canvas_h / img.height, 1.0)
        display_w = max(1, int(img.width * scale))
        display_h = max(1, int(img.height * scale))
        resized = img.resize((display_w, display_h), Image.LANCZOS)

        offset_x = (canvas_w - display_w) // 2
        offset_y = (canvas_h - display_h) // 2

        self._preview_offset = (offset_x, offset_y)
        self._preview_scale = scale

        checker.paste(resized, (offset_x, offset_y), resized)

        self.preview_photo = ImageTk.PhotoImage(checker)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.preview_photo)

    @staticmethod
    def _make_checkerboard(w, h, square=10):
        ys = np.arange(h) // square
        xs = np.arange(w) // square
        grid = (xs[None, :] + ys[:, None]) % 2
        arr = np.where(grid[:, :, None] == 0, 200, 255).astype(np.uint8)
        arr = np.broadcast_to(arr, (h, w, 3)).copy()
        return Image.fromarray(arr, "RGB")


if __name__ == "__main__":
    root = tk.Tk()
    BackgroundRemoverApp(root)
    root.mainloop()
