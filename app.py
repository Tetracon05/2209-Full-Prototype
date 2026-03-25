"""
app.py — Main application window.

Builds a sidebar-based navigation CTk window that hosts the four
phase panels.  A shared `state` dict is passed to every tab so they
can exchange processed data (DataProcessor, trained model, etc.).
"""

import customtkinter as ctk

from ui.data_tab        import DataTab
from ui.pretrained_tab  import PretrainedTab
from ui.custom_model_tab import CustomModelTab
from ui.evaluation_tab  import EvaluationTab


# Sidebar navigation item definitions
_NAV_ITEMS = [
    ("1 — Data Management",     "📂"),
    ("2 — Pre-trained Models",  "🤖"),
    ("3 — Custom Model",        "🔧"),
    ("4 — Evaluation",          "📊"),
]

# Colour palette
_SIDEBAR_BG   = "#0d0d1a"
_ACCENT       = "#1a73e8"
_ACCENT_HOVER = "#1558b0"
_FG_DARK      = "#1e1e2e"


class SolarPowerApp(ctk.CTk):
    """
    Root CTk window for the Solar Power Prediction application.
    Left sidebar provides navigation; right area swaps between phase frames.
    """

    def __init__(self):
        super().__init__()
        self.title("Solar Power Prediction — Deep Learning Interface")
        self.geometry("1280x760")
        self.minsize(900, 600)

        # Shared data store passed to every tab
        self._state: dict = {}

        self._setup_layout()
        self._build_sidebar()
        self._build_tabs()

        # Start on the first tab
        self._show_tab(0)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _setup_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self):
        """Build the fixed left navigation panel."""
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0,
                               fg_color=_SIDEBAR_BG)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)   # pushes theme toggle to bottom

        # ── App logo / title ──────────────────────────────────────────
        logo = ctk.CTkLabel(
            sidebar,
            text="☀️  SolarDL",
            font=("Segoe UI Bold", 20),
            text_color="#f8f9fa",
        )
        logo.grid(row=0, column=0, padx=18, pady=(22, 4), sticky="w")

        sub = ctk.CTkLabel(
            sidebar,
            text="Power Prediction Suite",
            font=("Segoe UI", 11),
            text_color="#8b949e",
        )
        sub.grid(row=1, column=0, padx=18, pady=(0, 18), sticky="w")

        ctk.CTkLabel(sidebar, text="NAVIGATION",
                     font=("Segoe UI Bold", 9),
                     text_color="#30363d").grid(
            row=2, column=0, padx=20, pady=(0, 4), sticky="w"
        )

        # ── Nav buttons ───────────────────────────────────────────────
        self._nav_buttons: list[ctk.CTkButton] = []
        for i, (label, icon) in enumerate(_NAV_ITEMS):
            btn = ctk.CTkButton(
                sidebar,
                text=f"  {icon}  {label}",
                font=("Segoe UI", 12),
                anchor="w",
                fg_color="transparent",
                hover_color="#1a1a2e",
                text_color="#c9d1d9",
                corner_radius=8,
                height=40,
                command=lambda idx=i: self._show_tab(idx),
            )
            btn.grid(row=3 + i, column=0, padx=10, pady=2, sticky="ew")
            self._nav_buttons.append(btn)

        # ── Separator ─────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="", height=1,
                     fg_color="#21262d").grid(
            row=8, column=0, sticky="ew", padx=10, pady=8
        )

        # ── Theme toggle ─────────────────────────────────────────────
        self._theme_var = ctk.StringVar(value="Dark")
        theme_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        theme_frame.grid(row=11, column=0, padx=12, pady=(0, 18), sticky="ew")
        ctk.CTkLabel(theme_frame, text="Theme:", font=("Segoe UI", 11),
                     text_color="#8b949e").pack(side="left")
        ctk.CTkSwitch(
            theme_frame,
            text="Light",
            variable=self._theme_var,
            onvalue="Light",
            offvalue="Dark",
            command=self._toggle_theme,
            width=44,
        ).pack(side="left", padx=8)

    def _build_tabs(self):
        """Instantiate all four phase frames (hidden initially)."""
        self._tabs: list[ctk.CTkFrame] = [
            DataTab(self,        self._state, fg_color=_FG_DARK),
            PretrainedTab(self,  self._state, fg_color=_FG_DARK),
            CustomModelTab(self, self._state, fg_color=_FG_DARK),
            EvaluationTab(self,  self._state, fg_color=_FG_DARK),
        ]
        for tab in self._tabs:
            tab.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def _show_tab(self, index: int):
        """Raise the selected tab frame and highlight the matching button."""
        for i, (tab, btn) in enumerate(zip(self._tabs, self._nav_buttons)):
            if i == index:
                tab.tkraise()
                btn.configure(fg_color=_ACCENT, text_color="#ffffff",
                              hover_color=_ACCENT_HOVER)
            else:
                btn.configure(fg_color="transparent", text_color="#c9d1d9",
                              hover_color="#1a1a2e")

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def _toggle_theme(self):
        mode = self._theme_var.get()
        ctk.set_appearance_mode(mode)
