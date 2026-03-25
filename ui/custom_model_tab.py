"""
custom_model_tab.py — Phase 3: Custom Model Designer panel.

Responsibilities:
  • Let users build a neural network by adding layers one-by-one
  • Display a scrollable layer list
  • Compile the model and start training with the same Trainer backend
  • Show live loss chart during training
"""

import threading
from tkinter import messagebox

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.model_builder import CustomModelBuilder, Trainer


class CustomModelTab(ctk.CTkFrame):
    """Phase 3 panel: interactive neural network layer designer."""

    def __init__(self, master, state: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state
        self.builder = CustomModelBuilder()
        self.trainer = Trainer()
        self._model = None
        self._train_loss = []
        self._val_loss   = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Left: Layer designer panel ──────────────────────────────────
        left = ctk.CTkScrollableFrame(self, width=290)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self._build_designer(left)

        # ── Right: Layer list + chart ───────────────────────────────────
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        # Layer list box
        list_frame = ctk.CTkFrame(right, height=160)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        ctk.CTkLabel(list_frame, text="Layer Stack",
                     font=("Segoe UI Bold", 12)).grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        self.layer_box = ctk.CTkTextbox(list_frame, height=120,
                                         font=("Courier New", 10), state="disabled")
        self.layer_box.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        # Loss chart
        chart_frame = ctk.CTkFrame(right)
        chart_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 4))
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(5, 3.5), dpi=96)
        self.fig.patch.set_facecolor("#1e1e2e")
        self.ax  = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Status + progress
        self.status_var = ctk.StringVar(value="Add layers and click 'Build & Train'.")
        ctk.CTkLabel(right, textvariable=self.status_var,
                     font=("Segoe UI", 11), anchor="w").grid(
            row=2, column=0, sticky="ew", padx=10, pady=(0, 2)
        )
        self.progress = ctk.CTkProgressBar(right)
        self.progress.set(0)
        self.progress.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))

    def _build_designer(self, parent):
        lbl = lambda t, pt=10: ctk.CTkLabel(
            parent, text=t, font=("Segoe UI Bold", 12), anchor="w"
        ).pack(fill="x", padx=4, pady=(pt, 2))

        lbl("➕  Add Layer", pt=4)

        # Layer type dropdown
        self.layer_type_var = ctk.StringVar(value="Conv1D")
        ctk.CTkOptionMenu(
            parent,
            values=CustomModelBuilder.SUPPORTED_LAYERS,
            variable=self.layer_type_var,
            command=self._on_layer_type_change
        ).pack(fill="x", padx=4, pady=2)

        # Dynamic parameter entries (shown/hidden per layer type)
        self.param_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.param_frame.pack(fill="x", padx=4, pady=4)
        self._param_widgets = {}
        self._build_conv1d_params()   # default view

        ctk.CTkButton(parent, text="➕ Add Layer",
                      command=self._add_layer).pack(fill="x", padx=4, pady=4)
        ctk.CTkButton(parent, text="❌ Remove Last",
                      fg_color="#c5221f", hover_color="#a50e0e",
                      command=self._remove_last).pack(fill="x", padx=4, pady=2)
        ctk.CTkButton(parent, text="🗑  Clear All",
                      fg_color="#5f6368", hover_color="#3c4043",
                      command=self._clear_all).pack(fill="x", padx=4, pady=2)

        lbl("⚙️  Training Settings")
        self.epochs_var = ctk.IntVar(value=20)
        self.batch_var  = ctk.IntVar(value=32)
        self.lr_var     = ctk.IntVar(value=10)

        def _row(label, var, lo, hi, steps):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(f, text=label, width=80, anchor="w").pack(side="left")
            ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps,
                          variable=var).pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(f, textvariable=var, width=40).pack(side="left")

        _row("Epochs", self.epochs_var, 5, 100, 19)
        _row("Batch ", self.batch_var,  8, 256, 30)
        _row("LR×1e4", self.lr_var,     1, 100, 99)

        ctk.CTkButton(parent, text="🔨  Build & Train",
                      fg_color="#1a73e8", hover_color="#1558b0",
                      font=("Segoe UI Bold", 13),
                      command=self._build_and_train).pack(fill="x", padx=4, pady=(16, 2))

        ctk.CTkButton(parent, text="⏹  Stop",
                      fg_color="#5f6368", hover_color="#3c4043",
                      command=self._stop).pack(fill="x", padx=4, pady=2)

    # ------------------------------------------------------------------
    # Parameter widget builder helpers
    # ------------------------------------------------------------------
    def _clear_param_frame(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self._param_widgets.clear()

    def _add_entry(self, label: str, key: str, default: str):
        f = ctk.CTkFrame(self.param_frame, fg_color="transparent")
        f.pack(fill="x", pady=1)
        ctk.CTkLabel(f, text=label, width=120, anchor="w").pack(side="left")
        v = ctk.StringVar(value=default)
        ctk.CTkEntry(f, textvariable=v, width=80).pack(side="left")
        self._param_widgets[key] = v

    def _build_conv1d_params(self):
        self._clear_param_frame()
        self._add_entry("Filters",         "filters",     "64")
        self._add_entry("Kernel Size",     "kernel_size", "3")
        self._add_entry("Activation",      "activation",  "relu")
        self._add_entry("Padding",         "padding",     "same")

    def _build_dense_params(self):
        self._clear_param_frame()
        self._add_entry("Units",      "units",      "128")
        self._add_entry("Activation", "activation", "relu")

    def _build_dropout_params(self):
        self._clear_param_frame()
        self._add_entry("Rate (0-1)", "rate", "0.3")

    def _build_pool_params(self):
        self._clear_param_frame()
        self._add_entry("Pool Size", "pool_size", "2")
        self._add_entry("Strides",   "strides",   "2")

    def _build_activation_params(self):
        self._clear_param_frame()
        self._add_entry("Activation", "activation", "relu")

    def _build_rnn_params(self, default_units=64):
        self._clear_param_frame()
        self._add_entry("Units",           "units",              str(default_units))
        self._add_entry("Return Seq",      "return_sequences",   "False")

    def _on_layer_type_change(self, value):
        builders = {
            "Conv1D":            self._build_conv1d_params,
            "Dense":             self._build_dense_params,
            "Dropout":           self._build_dropout_params,
            "MaxPooling1D":      self._build_pool_params,
            "Activation":        self._build_activation_params,
            "BatchNormalization": self._clear_param_frame,
            "Flatten":           self._clear_param_frame,
            "LSTM":              self._build_rnn_params,
            "GRU":               self._build_rnn_params,
        }
        builders.get(value, self._clear_param_frame)()

    # ------------------------------------------------------------------
    # Layer actions
    # ------------------------------------------------------------------
    def _add_layer(self):
        lt = self.layer_type_var.get()
        raw = {k: v.get() for k, v in self._param_widgets.items()}

        # Convert numeric strings to int/float
        params = {}
        for k, v in raw.items():
            if v.lower() in ("true", "false"):
                params[k] = v.lower() == "true"
            else:
                try:
                    params[k] = int(v)
                except ValueError:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v

        self.builder.add_layer(lt, params)
        self._refresh_layer_box()

    def _remove_last(self):
        if self.builder.layer_specs:
            self.builder.remove_layer(len(self.builder.layer_specs) - 1)
        self._refresh_layer_box()

    def _clear_all(self):
        self.builder.clear()
        self._refresh_layer_box()

    def _refresh_layer_box(self):
        text = self.builder.summary_str()
        self.layer_box.configure(state="normal")
        self.layer_box.delete("1.0", "end")
        self.layer_box.insert("end", text)
        self.layer_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # Build + Train
    # ------------------------------------------------------------------
    def _build_and_train(self):
        if not self.state.get("data_ready"):
            messagebox.showwarning("No Data", "Run Phase 1 first.")
            return
        if not self.builder.layer_specs:
            messagebox.showwarning("No Layers", "Add at least one layer.")
            return

        proc = self.state["processor"]
        X_tr, y_tr, X_v, y_v, X_te, y_te = proc.get_scaled_splits()
        self.state["X_test_scaled"] = X_te
        self.state["y_test_scaled"] = y_te

        n_features = X_tr.shape[1]
        lr = self.lr_var.get() * 1e-4

        try:
            self._model = self.builder.build((1, n_features), lr=lr)
        except Exception as exc:
            messagebox.showerror("Build Error", str(exc))
            return

        self.state["model"] = self._model
        self.state["active_model_name"] = "Custom Model"

        self._train_loss.clear()
        self._val_loss.clear()
        total = self.epochs_var.get()

        self.status_var.set("Training custom model…")

        def on_epoch(epoch, logs):
            self._train_loss.append(logs.get("loss", 0))
            self._val_loss.append(logs.get("val_loss", 0))
            frac = (epoch + 1) / total
            self.after(0, self._update_chart)
            self.after(0, lambda: self.progress.set(frac))
            self.after(0, lambda: self.status_var.set(
                f"Epoch {epoch+1}/{total} — loss={logs.get('loss',0):.5f}"
            ))

        def on_done(_):
            self.after(0, lambda: self.status_var.set(
                "✅ Custom model trained. Go to Phase 4 to evaluate."
            ))
            self.after(0, lambda: self.progress.set(1.0))

        self.trainer.train(
            self._model, X_tr, y_tr, X_v, y_v,
            epochs=total,
            batch_size=self.batch_var.get(),
            on_epoch_end=on_epoch,
            on_done=on_done,
        )

    def _stop(self):
        self.trainer.stop()
        self.status_var.set("Training stopped.")

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
        self.ax.set_title("Custom Model — Training Loss", color="#c9d1d9", fontsize=10)
        self.ax.legend(facecolor="#1e1e2e", edgecolor="#30363d",
                       labelcolor="#c9d1d9", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor("#12121f")
        ax.tick_params(colors="#c9d1d9", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
