import customtkinter as ctk

class ProgressWindow(ctk.CTkToplevel):
    def __init__(self, master, title="Processing...", text="Please wait while the task completes.", abort_callback=None, **kwargs):
        super().__init__(master, **kwargs)
        self.title(title)
        self.geometry("400x160")
        self.resizable(False, False)
        
        # Make the window modal
        self.transient(master)

        self.abort_callback = abort_callback

        # Center the window relative to its master
        self.update_idletasks()
        if master.winfo_viewable():
            x = master.winfo_rootx() + (master.winfo_width() // 2) - (400 // 2)
            y = master.winfo_rooty() + (master.winfo_height() // 2) - (160 // 2)
            self.geometry(f"+{x}+{y}")
        
        # Handle window close (X button) as abort
        self.protocol("WM_DELETE_WINDOW", self._on_abort)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.label = ctk.CTkLabel(self, text=text, font=("Segoe UI", 12))
        self.label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.progress = ctk.CTkProgressBar(self)
        self.progress.set(0)
        self.progress.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

        self.btn_abort = ctk.CTkButton(self, text="Abort",
                                       fg_color=("gray40", "gray30"),
                                       hover_color=("gray25", "gray15"),
                                       command=self._on_abort)
        self.btn_abort.grid(row=2, column=0, padx=20, pady=(0, 20))

        # Defer grab_set until the window is fully mapped and viewable
        self.after(100, self.grab_set)

    def set_progress(self, val: float):
        self.progress.set(val)

    def set_text(self, text: str):
        self.label.configure(text=text)

    def _on_abort(self):
        self.btn_abort.configure(state="disabled", text="Aborting...")
        if self.abort_callback:
            self.abort_callback()
        self.destroy()

    def close(self):
        self.destroy()
