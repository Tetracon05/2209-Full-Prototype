"""
evaluation_tab.py — Phase 4: Evaluation & Reporting panel.

Responsibilities:
  • Run inference on the test set using the trained model
  • Plot Actual vs Predicted line chart (FigureCanvasTkAgg)
  • Display R, RMSE, MAE, MAPE metric cards
  • Export results as CSV or PDF report
"""

import os
from tkinter import messagebox, filedialog

import customtkinter as ctk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.metrics import compute_metrics
from core.model_builder import Trainer
from core import report_generator as rg


class EvaluationTab(ctk.CTkFrame):
    """Phase 4 panel: evaluation metrics, prediction chart, and report export."""

    def __init__(self, master, state: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.state  = state
        self.trainer = Trainer()
        self._y_true = None
        self._y_pred = None
        self._metrics = {}
        self._fig = None
        self._inputs = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ── Top control bar ──────────────────────────────────────────────
        bar = ctk.CTkFrame(self, height=52)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        bar.grid_columnconfigure(5, weight=1)

        btn_eval = ctk.CTkButton(bar, text="Evaluate on Test Set",
                      fg_color=("gray60", "gray40"), hover_color=("gray45", "gray25"),
                      font=("Segoe UI Bold", 13),
                      command=self._run_evaluation)
        btn_eval.pack(side="left", padx=8, pady=8)
        self._inputs.append(btn_eval)

        btn_csv = ctk.CTkButton(bar, text="Export CSV",
                      fg_color=("gray50", "gray40"), hover_color=("gray35", "gray25"),
                      command=self._export_csv)
        btn_csv.pack(side="left", padx=4, pady=8)
        self._inputs.append(btn_csv)

        btn_pdf = ctk.CTkButton(bar, text="Export PDF",
                      fg_color=("gray40", "gray30"), hover_color=("gray25", "gray15"),
                      command=self._export_pdf)
        btn_pdf.pack(side="left", padx=4, pady=8)
        self._inputs.append(btn_pdf)

        self.status_var = ctk.StringVar(value="Train a model in Phase 2 or 3 first.")
        ctk.CTkLabel(bar, textvariable=self.status_var,
                     font=("Segoe UI", 11)).pack(side="left", padx=12)

        # ── Metric cards ────────────────────────────────────────────────
        cards = ctk.CTkFrame(self, height=90)
        cards.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        cards.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._metric_labels = {}
        for col, name in enumerate(["R", "RMSE", "MAE", "MAPE"]):
            card = ctk.CTkFrame(cards, corner_radius=10,
                                fg_color=("gray90", "#242424"),
                                border_color=("gray70", "#4d4d4d"), border_width=1)
            card.grid(row=0, column=col, sticky="nsew", padx=6, pady=6)
            ctk.CTkLabel(card, text=name, font=("Segoe UI Bold", 13),
                         text_color=("black", "white")).pack(pady=(10, 2))
            val_lbl = ctk.CTkLabel(card, text="—",
                                   font=("Segoe UI Bold", 20), text_color=("black", "white"))
            val_lbl.pack(pady=(0, 10))
            self._metric_labels[name] = val_lbl

        # ── Prediction chart ─────────────────────────────────────────────
        chart_frame = ctk.CTkFrame(self)
        chart_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._fig = Figure(figsize=(8, 4), dpi=96)
        self._ax = self._fig.add_subplot(111)
        self._style_axes(self._ax)
        self._ax.set_title("Actual vs Predicted Power Output",
                           color="#7f7f7f", fontsize=11)

        self.canvas = FigureCanvasTkAgg(self._fig, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew",
                                          padx=4, pady=4)
        self._set_canvas_bg()

        # ── Zoom controls ────────────────────────────────────────────────
        zoom_bar = ctk.CTkFrame(chart_frame, fg_color="transparent")
        zoom_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        ctk.CTkLabel(zoom_bar, text="Show last N samples:").pack(side="left", padx=4)
        self.zoom_var = ctk.IntVar(value=500)
        
        entry_str = ctk.StringVar(value=str(self.zoom_var.get()))
        entry = ctk.CTkEntry(zoom_bar, textvariable=entry_str, width=60)
        entry.pack(side="right", padx=(4, 8))
        self._inputs.append(entry)
        
        slider = ctk.CTkSlider(zoom_bar, from_=50, to=5000, number_of_steps=99, variable=self.zoom_var)
        slider.pack(side="left", fill="x", expand=True, padx=4)
        self._inputs.append(slider)
        
        def on_slide(val):
            entry_str.set(str(int(float(val))))
            self._redraw_chart()
        slider.configure(command=on_slide)
        
        def on_entry(event=None):
            try:
                v = int(entry_str.get())
                v = max(50, min(v, 5000))
                entry_str.set(str(v))
                self.zoom_var.set(v)
                slider.set(v)
                self._redraw_chart()
            except ValueError:
                entry_str.set(str(self.zoom_var.get()))
                
        entry.bind("<Return>", on_entry)
        entry.bind("<FocusOut>", on_entry)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _run_evaluation(self):
        model = self.state.get("model")
        proc  = self.state.get("processor")

        if model is None:
            messagebox.showwarning("No Model", "Train a model in Phase 2 or 3 first.")
            return
        if proc is None or not self.state.get("data_ready"):
            messagebox.showwarning("No Data", "Run Phase 1 first.")
            return
            
        app = self.winfo_toplevel()
        if hasattr(app, "set_tabs_locked"):
            app.set_tabs_locked(True)
        self.set_locked(True)
        try:
            self._do_run_evaluation(model, proc)
        finally:
            app = self.winfo_toplevel()
            if hasattr(app, "set_tabs_locked"):
                app.set_tabs_locked(False)
            self.set_locked(False)
            
    def _do_run_evaluation(self, model, proc):
        self.status_var.set("Running inference…")

        # Get scaled test split
        _, _, _, _, X_te, y_te = proc.get_scaled_splits()

        # Predict (reshape handled inside Trainer.predict)
        y_pred_scaled = self.trainer.predict(model, X_te)
        y_pred = proc.inverse_scale_y(y_pred_scaled)
        y_true = proc.inverse_scale_y(y_te.ravel())

        self._y_true = y_true
        self._y_pred = y_pred
        self._metrics = compute_metrics(y_true, y_pred)

        # Update metric cards
        for name, val in self._metrics.items():
            suffix = "%" if name == "MAPE" else ""
            self._metric_labels[name].configure(text=f"{val:.4f}{suffix}")

        # Draw chart
        self._redraw_chart()

        model_name = self.state.get("active_model_name", "Model")
        self.status_var.set(
            f"{model_name} — R={self._metrics['R']:.4f}  "
            f"RMSE={self._metrics['RMSE']:.4f}  "
            f"MAE={self._metrics['MAE']:.4f}  "
            f"MAPE={self._metrics['MAPE']:.2f}%"
        )

    def _redraw_chart(self):
        if self._y_true is None:
            return
        n = min(self.zoom_var.get(), len(self._y_true))
        yt = self._y_true[-n:]
        yp = self._y_pred[-n:]
        xs = range(len(yt))

        self._ax.clear()
        self._style_axes(self._ax)
        self._ax.plot(xs, yt, color="#c0c0c0", linewidth=1.0,
                      label="Actual",    alpha=0.9)
        self._ax.plot(xs, yp, color="#757575", linewidth=1.0,
                      label="Predicted", alpha=0.9, linestyle="--")
        self._ax.set_xlabel("Time Step", color="#7f7f7f", fontsize=9)
        self._ax.set_ylabel("Active Power (W)", color="#7f7f7f", fontsize=9)
        self._ax.set_title("Actual vs Predicted Power Output",
                           color="#7f7f7f", fontsize=11)
        self._ax.legend(facecolor="#2b2b2b", edgecolor="#7f7f7f",
                        labelcolor="#e0e0e0", fontsize=9)
        self._fig.tight_layout()
        self.canvas.draw()

    def _export_csv(self):
        if self._y_true is None:
            messagebox.showwarning("No Results", "Evaluate the model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save CSV Report"
        )
        if not path:
            return
        try:
            saved = rg.export_csv(self._y_true, self._y_pred, self._metrics, path)
            messagebox.showinfo("Saved", f"CSV saved to:\n{saved}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_pdf(self):
        if self._y_true is None:
            messagebox.showwarning("No Results", "Evaluate the model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save PDF Report"
        )
        if not path:
            return
        proc = self.state.get("processor")
        params = {
            "Model":    self.state.get("active_model_name", "N/A"),
            "Features": len(proc.feature_cols) if proc else "N/A",
            "Train":    self.state.get("split_info", {}).get("train", "N/A"),
            "Val":      self.state.get("split_info", {}).get("val",   "N/A"),
            "Test":     self.state.get("split_info", {}).get("test",  "N/A"),
        }
        try:
            saved = rg.export_pdf(
                self._y_true, self._y_pred,
                self._metrics, params, path, self._fig
            )
            messagebox.showinfo("Saved", f"Report saved to:\n{saved}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
        self._fig.patch.set_facecolor(bg)
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
