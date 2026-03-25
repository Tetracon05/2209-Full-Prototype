"""
data_tab.py — Phase 1: Data Management & Decomposition panel.

Responsibilities:
  • Load CSV, display dataset summary
  • Show correlation bar chart (FigureCanvasTkAgg)
  • Select & apply decomposition method (VMD/EMD/EEMD/CEEMDAN)
  • Configure and execute 70/15/15 split
  • All heavy work runs in background threads to keep UI responsive
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.data_processor import DataProcessor
from core.decomposition import decompose


class DataTab(ctk.CTkFrame):
    """Phase 1 panel: dataset management and signal decomposition."""

    def __init__(self, master, state: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state          # shared app-level state dict
        self.processor = DataProcessor()
        self.state["processor"] = self.processor

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left control panel ──────────────────────────────────────────
        ctrl = ctk.CTkScrollableFrame(self, width=260)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self._build_controls(ctrl)

        # ── Right display area ──────────────────────────────────────────
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # Dataset info text box
        self.info_box = ctk.CTkTextbox(right, height=140, state="disabled",
                                        font=("Courier New", 11))
        self.info_box.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        # Correlation chart placeholder
        chart_frame = ctk.CTkFrame(right)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 3.5), dpi=96)
        self.fig.patch.set_facecolor("#1e1e2e")
        self.ax = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Status label
        self.status_var = ctk.StringVar(value="Ready — load a CSV file to begin.")
        ctk.CTkLabel(right, textvariable=self.status_var,
                     font=("Segoe UI", 11), anchor="w").grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 6)
        )

    def _build_controls(self, parent):
        """Populate the left sidebar with controls."""
        lbl = lambda text, pad_top=12: ctk.CTkLabel(
            parent, text=text, font=("Segoe UI Bold", 12), anchor="w"
        ).pack(fill="x", padx=4, pady=(pad_top, 2))

        # Section: Load
        lbl("📂  Load Dataset", pad_top=4)
        ctk.CTkButton(parent, text="Browse CSV…",
                      command=self._load_csv).pack(fill="x", padx=4, pady=2)
        self.lbl_file = ctk.CTkLabel(parent, text="No file loaded",
                                      font=("Segoe UI", 10),
                                      text_color="gray", wraplength=230, anchor="w")
        self.lbl_file.pack(fill="x", padx=4)

        # Section: Cleaning
        lbl("🧹  Missing Values Strategy")
        self.clean_var = ctk.StringVar(value="drop")
        for opt in ("drop", "mean", "ffill"):
            ctk.CTkRadioButton(parent, text=opt, variable=self.clean_var,
                               value=opt).pack(anchor="w", padx=16, pady=1)

        # Section: Correlation top_n
        lbl("📊  Top-N Correlation Features")
        self.topn_var = ctk.IntVar(value=7)
        ctk.CTkSlider(parent, from_=3, to=15, number_of_steps=12,
                      variable=self.topn_var).pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(parent, textvariable=self.topn_var).pack()

        # Section: Lag features
        lbl("⏱  Lag Window (steps)")
        self.lag_var = ctk.IntVar(value=3)
        ctk.CTkSlider(parent, from_=0, to=10, number_of_steps=10,
                      variable=self.lag_var).pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(parent, textvariable=self.lag_var).pack()

        # Section: Decomposition
        lbl("🌊  Decomposition Method")
        self.decomp_var = ctk.StringVar(value="EMD")
        ctk.CTkOptionMenu(parent, values=["None", "EMD", "EEMD", "CEEMDAN", "VMD"],
                          variable=self.decomp_var).pack(fill="x", padx=4, pady=2)

        lbl("   # IMF Components")
        self.imf_var = ctk.IntVar(value=5)
        ctk.CTkSlider(parent, from_=2, to=12, number_of_steps=10,
                      variable=self.imf_var).pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(parent, textvariable=self.imf_var).pack()

        # Action button
        ctk.CTkButton(parent, text="⚡  Process Data",
                      fg_color="#1a73e8", hover_color="#1558b0",
                      font=("Segoe UI Bold", 13),
                      command=self._process_data).pack(fill="x", padx=4, pady=(18, 4))

        self.progress = ctk.CTkProgressBar(parent)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=4, pady=4)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _load_csv(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            summary = self.processor.load_csv(path)
            self.lbl_file.configure(text=path.split("/")[-1].split("\\")[-1])
            self._update_info(summary)
            self.state["data_loaded"] = True
            self.status_var.set(
                f"Loaded {summary['rows']:,} rows × {len(summary['columns'])} columns "
                f"({summary['missing']} missing values)"
            )
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    def _process_data(self):
        if not self.state.get("data_loaded"):
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        """Full processing pipeline executed in a background thread."""
        self._set_progress(0.05)
        self.status_var.set("Cleaning data…")

        # 1. Clean
        n = self.processor.clean(self.clean_var.get())
        self._set_progress(0.15)

        # 2. Correlation + chart
        self.status_var.set("Computing correlations…")
        try:
            corr = self.processor.compute_correlation(self.topn_var.get())
            self.state["corr"] = corr
            self.after(0, self._draw_correlation, corr)
        except Exception as exc:
            self.after(0, lambda: messagebox.showerror("Correlation Error", str(exc)))
        self._set_progress(0.30)

        # 3. Lag features
        if self.lag_var.get() > 0:
            self.status_var.set("Adding lag features…")
            lags = list(range(1, self.lag_var.get() + 1))
            top_cols = list(self.state["corr"].index) if "corr" in self.state else []
            self.processor.add_lag_features(top_cols, lags)
        self._set_progress(0.45)

        # 4. Decomposition
        method = self.decomp_var.get()
        if method != "None":
            self.status_var.set(f"Decomposing signal with {method}…")
            try:
                signal = self.processor.df["Active_Power"].values
                imfs = decompose(signal, method=method, n_components=self.imf_var.get())
                self.processor.add_imf_features(imfs, prefix=method)
            except Exception as exc:
                self.after(0, lambda e=exc: messagebox.showerror(
                    "Decomposition Error", str(e)))
        self._set_progress(0.70)

        # 5. Split
        self.status_var.set("Splitting dataset (70/15/15)…")
        info = self.processor.split()
        self.state["split_info"] = info
        self.state["data_ready"] = True
        self._set_progress(1.0)

        # 6. Update info
        lines = [
            f"After cleaning: {n:,} rows",
            f"Features used : {len(info['features'])}",
            f"Train samples : {info['train']:,}",
            f"Val   samples : {info['val']:,}",
            f"Test  samples : {info['test']:,}",
        ]
        self.after(0, lambda: self._append_info("\n".join(lines)))
        self.status_var.set(
            f"✅ Done — {info['train']+info['val']+info['test']:,} samples ready."
        )

    def _draw_correlation(self, corr):
        """Redraw the correlation bar chart on the embedded figure."""
        self.ax.clear()
        self._style_axes(self.ax)
        colors_ = ["#1a73e8" if v >= 0 else "#ea4335" for v in corr.values]
        self.ax.barh(corr.index[::-1], corr.values[::-1], color=colors_[::-1])
        self.ax.set_xlabel("Absolute Correlation with Active_Power",
                           color="#c9d1d9", fontsize=9)
        self.ax.set_title("Feature Correlation", color="#c9d1d9", fontsize=10, pad=8)
        self.fig.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_progress(self, val: float):
        self.after(0, lambda: self.progress.set(val))

    def _update_info(self, summary: dict):
        text = (
            f"Rows     : {summary['rows']:>10,}\n"
            f"Columns  : {summary['columns']}\n"
            f"Missing  : {summary['missing']:>10,}\n"
        )
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", "end")
        self.info_box.insert("end", text)
        self.info_box.configure(state="disabled")

    def _append_info(self, text: str):
        self.info_box.configure(state="normal")
        self.info_box.insert("end", "\n" + text)
        self.info_box.configure(state="disabled")

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor("#12121f")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
