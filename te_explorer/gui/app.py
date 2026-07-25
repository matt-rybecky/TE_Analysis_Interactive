"""Application shell: root window, tab notebook, shared session state.

The GUI is a three-tab notebook (Data Preparation, TE Analysis, IAAFT
Significance). Each tab is its own class in its own module; this shell owns
the pieces they share: the application configuration, the data directory,
the loaded dataset, and the latest TE results. Tabs communicate only
through :class:`SessionState` and the callbacks exposed here, never by
reaching into each other's widgets.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import ttk
from typing import Optional

import pandas as pd

from te_explorer.config import AppConfig, DATA_DIR


@dataclass
class SessionState:
    """State shared across tabs.

    Attributes
    ----------
    config : AppConfig
        Application configuration (publication defaults).
    data_dir : Path
        Directory scanned for CSV data files.
    current_file : str or None
        Filename of the dataset selected in the analysis tab.
    current_data : pd.DataFrame or None
        The loaded dataset backing the current analysis.
    te_results : dict or None
        Latest rolling TE/JTE results from the analysis tab; read by the
        significance tab.
    """

    config: AppConfig = field(default_factory=AppConfig)
    data_dir: Path = DATA_DIR
    current_file: Optional[str] = None
    current_data: Optional[pd.DataFrame] = None
    te_results: Optional[dict] = None


class TEExplorerApp:
    """Root application: notebook shell wiring the three tabs together.

    Parameters
    ----------
    root : tk.Tk
        Root tkinter window.
    config : AppConfig, optional
        Configuration override; publication defaults when omitted.
    """

    def __init__(self, root: tk.Tk, config: Optional[AppConfig] = None):
        self.root = root
        self.root.title("TE Explorer 1.0.0")
        self.root.geometry("1200x800")

        self.state = SessionState(config=config or AppConfig())
        self.state.data_dir.mkdir(exist_ok=True)

        self._create_widgets()

    def _create_widgets(self) -> None:
        """Build the notebook and instantiate the three tabs."""
        # Imports deferred so each tab module can import this one freely.
        from te_explorer.gui.prep_tab import PrepTab
        from te_explorer.gui.analysis_tab import AnalysisTab
        from te_explorer.gui.significance_tab import SignificanceTab

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        prep_frame = ttk.Frame(self.notebook)
        analysis_frame = ttk.Frame(self.notebook)
        significance_frame = ttk.Frame(self.notebook)
        self.notebook.add(prep_frame, text="Data Preparation")
        self.notebook.add(analysis_frame, text="Transfer Entropy Analysis")
        self.notebook.add(significance_frame, text="IAAFT Significance")

        self.prep_tab = PrepTab(prep_frame, self)
        self.analysis_tab = AnalysisTab(analysis_frame, self)
        self.significance_tab = SignificanceTab(significance_frame, self)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def refresh_file_lists(self) -> None:
        """Rescan the data directory in every tab that lists files.

        Called by the preparation tab after it saves a new analysis-ready
        CSV so the analysis tab sees it immediately.
        """
        self.analysis_tab.load_available_files()


def run() -> None:
    """Launch the application."""
    root = tk.Tk()
    TEExplorerApp(root)
    root.mainloop()
