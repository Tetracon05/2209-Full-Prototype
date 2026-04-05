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
        self.trainer = Trainer()
        self._train_loss = []
        self._val_loss   = []
        self._inputs = []
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
        self.ax  = self.fig.add_subplot(111)
        self._style_axes(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                         padx=8, pady=8)
        self._set_canvas_bg()

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
        lbl("Model Architecture")
        model_names = list(MODEL_REGISTRY.keys())
        initial_value = model_names[0] if model_names else "Custom"
        self.model_var = ctk.StringVar(value=initial_value)
        menu_model = ctk.CTkOptionMenu(parent, values=model_names,
                          variable=self.model_var)
        menu_model.pack(fill="x", padx=4, pady=2)
        self._inputs.append(menu_model)

        # Model summary display
        lbl("Model Summary")
        self.summary_box = ctk.CTkTextbox(parent, height=120, state="disabled",
                                           font=("Courier New", 9))
        self.summary_box.pack(fill="x", padx=4, pady=2)
        ctk.CTkButton(parent, text="Show Summary",
                      command=self._show_summary).pack(fill="x", padx=4, pady=2)

        # Hyperparameters
        lbl("Hyperparameters")

        def _row(label, var, lo, hi, steps):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(f, text=label, width=80, anchor="w").pack(side="left")
            
            entry_str = ctk.StringVar(value=str(var.get()))
            entry = ctk.CTkEntry(f, textvariable=entry_str, width=45)
            entry.pack(side="right", padx=(4, 0))
            self._inputs.append(entry)
            
            slider = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=steps, variable=var)
            slider.pack(side="left", fill="x", expand=True)
            self._inputs.append(slider)

            def on_slide(val):
                entry_str.set(str(int(float(val))))
            slider.configure(command=on_slide)

            def on_entry(event=None):
                try:
                    v = int(entry_str.get())
                    v = max(lo, min(v, hi))
                    entry_str.set(str(v))
                    var.set(v)
                    slider.set(v)
                except ValueError:
                    entry_str.set(str(var.get()))
                    
            entry.bind("<Return>", on_entry)
            entry.bind("<FocusOut>", on_entry)

        self.epochs_var = ctk.IntVar(value=30)
        self.batch_var  = ctk.IntVar(value=32)
        _row("Epochs  ", self.epochs_var, 5, 200, 39)
        _row("Batch   ", self.batch_var,  8, 256, 30)

        self.lr_var = ctk.IntVar(value=10)   # stored as int × 1e-4
        _row("LR(×1e-4)", self.lr_var, 1, 100, 99)

        # Buttons
        self.btn_train = ctk.CTkButton(parent, text="Train",
                      fg_color=("gray60", "gray40"), hover_color=("gray45", "gray25"),
                      font=("Segoe UI Bold", 13),
                      command=self._start_training)
        self.btn_train.pack(fill="x", padx=4, pady=8)
        self._inputs.append(self.btn_train)

        self.btn_stop = ctk.CTkButton(parent, text="Stop",
                      fg_color=("gray40", "gray30"), hover_color=("gray25", "gray15"),
                      command=self._stop_training)
        self.btn_stop.pack(fill="x", padx=4)

        self.progress = ctk.CTkProgressBar(parent)
        self.progress.set(0)
        self.progress.pack(fill="x", padx=4, pady=6)

        lbl("Epoch Stats")
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
        
        app = self.winfo_toplevel()
        if hasattr(app, "set_tabs_locked"):
            app.set_tabs_locked(True)
        self.set_locked(True)
            
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            self._do_run_training()
        finally:
            self.after(0, lambda: self.set_locked(False))
            app = self.winfo_toplevel()
            if hasattr(app, "set_tabs_locked"):
                self.after(0, lambda: app.set_tabs_locked(False))

    def _do_run_training(self):
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
        if hasattr(self.trainer, "error") and self.trainer.error:
            self.status_var.set(f"Training failed: {type(self.trainer.error).__name__}")
            messagebox.showerror("Training Error", str(self.trainer.error))
        else:
            self.status_var.set(
                f"Training complete — {self._epochs_done} epochs. "
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
        self.ax.plot(e, self._train_loss, color="#9e9e9e", label="Train Loss")
        if self._val_loss:
            self.ax.plot(e, self._val_loss, label="Val Loss", color="#757575", linewidth=1.5)
            
        self.ax.set_xlabel("Epoch", color="#7f7f7f", fontsize=9)
        self.ax.set_ylabel("Loss (MSE)", color="#7f7f7f", fontsize=9)
        self.ax.set_title("Training Progress", color="#7f7f7f", fontsize=10, pad=8)
        self.ax.legend(facecolor="#2b2b2b", edgecolor="#7f7f7f",
                       labelcolor="#e0e0e0", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def _update_stats(self, text: str):
        self.stats_box.configure(state="normal")
        self.stats_box.delete("1.0", "end")
        self.stats_box.insert("end", text)
        self.stats_box.configure(state="disabled")

    def _update_lr_label(self, *_):
        pass

    def set_locked(self, locked: bool):
        state = "disabled" if locked else "normal"
        for w in self._inputs:
            try: w.configure(state=state)
            except Exception: pass

    def _set_canvas_bg(self, mode=None):
        if mode is None:
            import customtkinter as ctk
            mode = ctk.get_appearance_mode()
        bg = "#1e1e2e" if mode == "Dark" else "#e5e5e5"
        self.fig.patch.set_facecolor(bg)
        self.canvas.get_tk_widget().configure(bg=bg)
        self.canvas.draw()

    def update_theme(self, mode):
        self._set_canvas_bg(mode)

    @staticmethod
    def _style_axes(ax):
        ax.set_facecolor("none")
        ax.tick_params(colors="#7f7f7f", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#7f7f7f")
