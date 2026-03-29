"""
models_tab.py - Phase 2: Consolidated Modeling panel.

A CTkTabview container that hosts:
  1. Pre-trained Models
  2. Custom Architecture
"""

import customtkinter as ctk

from ui.pretrained_tab import PretrainedTab
from ui.custom_model_tab import CustomModelTab

class ModelsTab(ctk.CTkFrame):
    def __init__(self, master, state: dict, **kwargs):
        super().__init__(master, **kwargs)
        self.state = state

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.tabview.add("Pre-trained Models")
        self.tabview.add("Custom Architecture")

        self.pretrained_tab = PretrainedTab(self.tabview.tab("Pre-trained Models"), self.state, fg_color="transparent")
        self.pretrained_tab.pack(fill="both", expand=True)

        self.custom_tab = CustomModelTab(self.tabview.tab("Custom Architecture"), self.state, fg_color="transparent")
        self.custom_tab.pack(fill="both", expand=True)

    def update_theme(self, mode):
        if hasattr(self.pretrained_tab, "update_theme"):
            self.pretrained_tab.update_theme(mode)
        if hasattr(self.custom_tab, "update_theme"):
            self.custom_tab.update_theme(mode)
