"""
pretrained_tab.py — Phase 2: Pre-trained Model Training panel.

Responsibilities:
  • Select from 6 pre-defined 1-D CNN architectures
  • Configure hyperparameters (epochs, batch size, learning rate)
  • Start/stop training in a background thread
  • Show live progress bar and loss/val_loss chart
"""

import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.model_builder import MODEL_REGISTRY, get_model, Trainer


class PretrainedTab(ctk.CTkFrame):
    """Phase 2 panel: pre-defined CNN model training."""

    def __init__(self, master, state: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.trainer = Trainer()
        self._model = None
        self._epochs_done = 0
        self._train_loss = []
        self._val_loss   = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Left controls
        ctrl = ctk.CTkScrollableFrame(self, width=270)
        ctrl.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self._build_controls(ctrl)

        # Right: chart panel
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 4), dpi=96)
        self.fig.patch.set_facecolor("#1e1e2e")
        self.ax  = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                         padx=8, pady=8)

        self.status_var = ctk.StringVar(value="Ready — configure and start training.")
        ctk.CTkLabel(right, textvariable=self.status_var,
                     font=("Segoe UI", 11), anchor="w").grid(
            row=1, column=0, sticky="ew", padx=10, pady=(0, 6)
        )

    def _build_controls(self, parent):
        lbl = lambda t, pt=12: ctk.CTkLabel(
            parent, text=t, font=("Segoe UI Bold", 12), anchor="w"
        ).pack(fill="x", padx=4, pady=(pt, 2))

        # Model selection
        lbl("🤖  Model Architecture", pt=4)
        self.model_var = ctk.StringVar(value="ResNet")
        ctk.CTkOptionMenu(parent, values=list(MODEL_REGISTRY.keys()),
                          variable=self.model_var).pack(fill="x", padx=4, pady=2)

        # Model summary display
        lbl("📋  Model Summary")
        self.summary_box = ctk.CTkTextbox(parent, height=120, state="disabled",
                                           font=("Courier New", 9))
        self.summary_box.pack(fill="x", padx=4, pady=2)
        ctk.CTkButton(parent, text="Show Summary",
                      command=self._show_summary).pack(fill="x", padx=4, pady=2)

        # Hyperparameters
        lbl("⚙️  Hyperparameters")

        def _row(label, var, lo, hi, steps):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(f, text=label, width=80, anchor="w").pack(side="left")
            ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps,
                          variable=var).pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(f, textvariable=var, width=40).pack(side="left")

        self.epochs_var = ctk.IntVar(value=30)
        self.batch_var  = ctk.IntVar(value=32)
        _row("Epochs  ", self.epochs_var, 5, 200, 39)
        _row("Batch   ", self.batch_var,  8, 256, 30)

        lbl("  Learning Rate (×1e-4)")
        self.lr_var = ctk.IntVar(value=10)   # stored as int × 1e-4
        ctk.CTkSlider(parent, from_=1, to=100, number_of_steps=99,
                      variable=self.lr_var).pack(fill="x", padx=4, pady=2)
        self.lr_label = ctk.CTkLabel(parent, text="LR = 0.0010")
        self.lr_label.pack()
        self.lr_var.trace_add("write", self._update_lr_label)

        # Buttons
        self.btn_train = ctk.CTkButton(parent, text="▶  Train",
                                        fg_color="#188038", hover_color="#0d652d",
                                        font=("Segoe UI Bold", 13),
                                        command=self._start_training)
        self.btn_train.pack(fill="x", padx=4, pady=(16, 4))

        self.btn_stop = ctk.CTkButton(parent, text="⏹  Stop",
                                       fg_color="#c5221f", hover_color="#a50e0e",
                                       state="disabled",
                                       command=self._stop_training)
        self.btn_stop.pack(fill="x", padx=4, pady=2)

        self.progress = ctk.CTkProgressBar(parent)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=4, pady=6)

        lbl("📈  Epoch Stats")
        self.stats_box = ctk.CTkTextbox(parent, height=80, state="disabled",
                                         font=("Courier New", 9))
        self.stats_box.pack(fill="x", padx=4, pady=2)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _show_summary(self):
        """Build the model silently and display its summary in the textbox."""
        if not self.state.get("data_ready"):
            messagebox.showwarning("No Data", "Run Phase 1 first.")
            return
        try:
            proc = self.state["processor"]
            n_features = len(proc.feature_cols)
            model = get_model(self.model_var.get(), (1, n_features))
            lines = []
            model.summary(print_fn=lambda s: lines.append(s))
            text = "\n".join(lines)
        except Exception as exc:
            text = str(exc)
        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("end", text)
        self.summary_box.configure(state="disabled")

    def _start_training(self):
        if not self.state.get("data_ready"):
            messagebox.showwarning("No Data", "Run Phase 1 first.")
            return

        proc = self.state["processor"]
        X_tr, y_tr, X_v, y_v, X_te, y_te = proc.get_scaled_splits()
        self.state["X_test_scaled"] = X_te
        self.state["y_test_scaled"] = y_te

        n_features = X_tr.shape[1]
        lr = self.lr_var.get() * 1e-4

        try:
            self._model = get_model(self.model_var.get(), (1, n_features))
        except Exception as exc:
            messagebox.showerror("Model Error", str(exc))
            return

        self.state["model"] = self._model
        self.state["active_model_name"] = self.model_var.get()

        self._train_loss.clear()
        self._val_loss.clear()
        self._epochs_done = 0
        total = self.epochs_var.get()

        self.btn_train.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.status_var.set("Training…")

        def on_epoch(epoch, logs):
            self._train_loss.append(logs.get("loss", 0))
            self._val_loss.append(logs.get("val_loss", 0))
            self._epochs_done = epoch + 1
            frac = self._epochs_done / total
            stats = (f"Epoch {self._epochs_done}/{total}\n"
                     f"  loss    = {logs.get('loss', 0):.6f}\n"
                     f"  val_loss= {logs.get('val_loss', 0):.6f}")
            self.after(0, self._update_chart)
            self.after(0, lambda: self.progress.set(frac))
            self.after(0, self._update_stats, stats)

        def on_done(history):
            self.after(0, self._training_finished)

        self.trainer.train(
            self._model, X_tr, y_tr, X_v, y_v,
            epochs=total,
            batch_size=self.batch_var.get(),
            on_epoch_end=on_epoch,
            on_done=on_done,
        )

    def _stop_training(self):
        self.trainer.stop()
        self.status_var.set("Stopped by user.")
        self.btn_train.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _training_finished(self):
        self.progress.set(1.0)
        self.status_var.set(
            f"✅ Training complete — {self._epochs_done} epochs. "
            "Go to Phase 4 to evaluate."
        )
        self.btn_train.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_chart(self):
        self.ax.clear()
        self._style_axes(self.ax)
        e = range(1, len(self._train_loss) + 1)
        self.ax.plot(e, self._train_loss, color="#1a73e8", label="Train Loss")
        self.ax.plot(e, self._val_loss,   color="#fbbc04", label="Val Loss")
        self.ax.set_xlabel("Epoch", color="#c9d1d9", fontsize=9)
        self.ax.set_ylabel("MSE Loss", color="#c9d1d9", fontsize=9)
        self.ax.set_title("Training / Validation Loss", color="#c9d1d9", fontsize=10)
        self.ax.legend(facecolor="#1e1e2e", edgecolor="#30363d",
                       labelcolor="#c9d1d9", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def _update_stats(self, text: str):
        self.stats_box.configure(state="normal")
        self.stats_box.delete("1.0", "end")
        self.stats_box.insert("end", text)
        self.stats_box.configure(state="disabled")

    def _update_lr_label(self, *_):
        self.lr_label.configure(text=f"LR = {self.lr_var.get() * 1e-4:.4f}")

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor("#12121f")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
