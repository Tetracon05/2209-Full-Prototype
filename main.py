"""
main.py — Entry point for the Solar Power Prediction application.

Run with:
    python main.py
"""

import customtkinter as ctk
from app import SolarPowerApp

if __name__ == "__main__":
    # Set global appearance before creating any widgets
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = SolarPowerApp()
    app.mainloop()
