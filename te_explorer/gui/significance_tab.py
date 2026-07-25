"""IAAFT surrogate significance testing tab.

Provides a self-contained GUI panel that runs rolling-window IAAFT surrogate
significance testing for individual TE or joint TE (JTE) and displays the
result as a time series with a shaded significance band.  All parameter
controls live on this tab; the tab reads ``app.state`` only for defaults
and for the latest ``te_results`` produced by the analysis tab.

The surrogate type is fixed to ``'iaaft'`` throughout (``AppConfig`` enforces
this).  No noise-injection (MCMC) branch is present.

Rendering and export logic is in the companion module
``_significance_plot.py``; this file is a pure GUI controller.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from tkinter import messagebox, ttk

import te_explorer.config  # noqa: F401 - side effect: injects core/ into sys.path
from te_explorer.config import AppConfig, OUTPUT_DIR
from te_explorer.gui._significance_plot import (
    export_significance_csv,
    render_significance_plot,
    save_significance_plot,
)
from TE_Surrogate import SurrogateAnalyzer


class SignificanceTab:
    """IAAFT surrogate significance testing tab.

    Parameters
    ----------
    parent : ttk.Frame
        The notebook frame that owns this tab.
    app : TEExplorerApp
        Root application shell, used to reach ``app.root``, ``app.state``.

    Notes
    -----
    Parameter sourcing
    ------------------
    The original ``create_mcmc_tab`` in ``TE_Main.py`` pulled every analysis
    parameter (file, target, inputs, tau, window, history, worker count) from
    the *analysis tab's own tk.Variables* via ``self.file_var``,
    ``self.target_var``, etc., which were attributes of the same monolithic
    class.  In the new architecture those attributes are not accessible from
    here (the contract forbids reaching into ``app.analysis_tab`` widgets).
    This tab therefore owns its own self-contained parameter controls.
    Defaults come from ``app.state.config``; when ``app.state.te_results`` is
    present the controls are pre-filled from its ``metadata`` block via
    :meth:`prefill_from_te_results`.

    Public surface
    --------------
    ``SignificanceTab(parent, app)`` - constructor, builds the full UI.
    ``prefill_from_te_results()`` - populate controls from ``app.state.te_results``.
    """

    def __init__(self, parent: ttk.Frame, app: Any) -> None:
        self._app = app
        self._root: tk.Tk = app.root
        self._cfg: AppConfig = app.state.config

        self._sig_results: Optional[Dict] = None
        self._sig_thread: Optional[threading.Thread] = None
        self._sig_error: Optional[str] = None
        self._tau_dict_cache: Optional[Dict[str, int]] = None

        self._build_settings_frame(parent)
        self._build_results_frame(parent)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

    def _build_settings_frame(self, parent: ttk.Frame) -> None:
        """Build the parameter settings LabelFrame (top half of tab)."""
        sf = ttk.LabelFrame(parent, text="IAAFT Significance Settings", padding="10")
        sf.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        sf.columnconfigure(1, weight=1)

        self._build_variable_controls(sf)
        self._build_fixed_labels(sf)
        self._build_run_controls(sf)

    def _build_variable_controls(self, sf: ttk.LabelFrame) -> None:
        """Insert rows 0-6: file, target, inputs, tau, window, history, surrogates.

        Parameters
        ----------
        sf : ttk.LabelFrame
            Parent frame (settings frame).
        """
        cfg = self._cfg

        # Row 0: data file
        ttk.Label(sf, text="Data File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._file_var = tk.StringVar()
        self._file_combo = ttk.Combobox(sf, textvariable=self._file_var,
                                        width=35, state="readonly")
        self._file_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self._refresh_file_list()
        ttk.Button(sf, text="Refresh", command=self._refresh_file_list).grid(
            row=0, column=2, sticky=tk.W)

        # Row 1: target variable
        ttk.Label(sf, text="Target Variable:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self._target_var = tk.StringVar()
        ttk.Entry(sf, textvariable=self._target_var, width=20).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Label(sf, text="(column name in CSV)",
                  font=('TkDefaultFont', 8), foreground="darkblue").grid(
            row=1, column=2, sticky=tk.W)

        # Row 2: input variables
        ttk.Label(sf, text="Input Variables:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self._inputs_var = tk.StringVar()
        ttk.Entry(sf, textvariable=self._inputs_var, width=40).grid(
            row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Label(sf, text="comma-separated; 2+ = JTE mode",
                  font=('TkDefaultFont', 8), foreground="darkblue").grid(
            row=2, column=3, sticky=tk.W)

        # Rows 3-6: numeric spinboxes
        self._tau_var = tk.IntVar(value=cfg.tau_default)
        self._window_var = tk.IntVar(value=cfg.window_days)
        self._history_var = tk.IntVar(value=cfg.history_length)
        self._n_surr_var = tk.IntVar(value=cfg.n_surrogates)

        self._spinrow(sf, row=3, label="Time Lag (tau):",
                      var=self._tau_var, from_=0, to=24,
                      info="Time lag in data intervals")
        self._spinrow(sf, row=4, label="Window (days):",
                      var=self._window_var, from_=1, to=365,
                      info="Rolling window length")
        self._spinrow(sf, row=5, label="History Length (h):",
                      var=self._history_var, from_=1, to=10,
                      info="Past target steps to condition on")
        self._spinrow(sf, row=6, label="Surrogates per Window:",
                      var=self._n_surr_var, from_=100, to=10000, increment=100,
                      info="More surrogates = sharper null (slower)")

    def _build_fixed_labels(self, sf: ttk.LabelFrame) -> None:
        """Insert rows 7-8: fixed method and KSG-k informational labels.

        Parameters
        ----------
        sf : ttk.LabelFrame
            Parent settings frame.
        """
        cfg = self._cfg
        # Row 7: method label (fixed)
        ttk.Label(sf, text="Method:").grid(row=7, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(sf, text="IAAFT, p<0.05 (95th percentile)",
                  foreground="darkgreen").grid(row=7, column=1, sticky=tk.W)
        ttk.Label(sf, text="Preserves spectrum + amplitude distribution",
                  font=('TkDefaultFont', 8), foreground="darkblue").grid(
            row=7, column=2, sticky=tk.W)

        # Row 8: KSG label (fixed)
        ttk.Label(sf, text="KSG k:").grid(row=8, column=0, sticky=tk.W, padx=(0, 5))
        ttk.Label(sf, text=f"k={cfg.ksg_k} (fixed)",
                  foreground="darkgreen").grid(row=8, column=1, sticky=tk.W)

    def _build_run_controls(self, sf: ttk.LabelFrame) -> None:
        """Insert rows 9-11: run button, pre-fill button, progress label and bar.

        Parameters
        ----------
        sf : ttk.LabelFrame
            Parent settings frame.
        """
        # Row 9: run / pre-fill buttons
        self._run_btn = ttk.Button(sf, text="Run IAAFT Significance Test",
                                   command=self._start_analysis)
        self._run_btn.grid(row=9, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        ttk.Button(sf, text="Pre-fill from TE Results",
                   command=self.prefill_from_te_results).grid(
            row=9, column=2, pady=(10, 0), sticky=tk.W)

        # Rows 10-11: progress
        self._progress_var = tk.StringVar(value="Enter parameters above and click Run")
        ttk.Label(sf, textvariable=self._progress_var).grid(
            row=10, column=0, columnspan=4, pady=(5, 0), sticky=tk.W)
        self._progress_bar = ttk.Progressbar(sf, mode='determinate')
        self._progress_bar.grid(row=11, column=0, columnspan=4,
                                sticky=(tk.W, tk.E), pady=(5, 0))

    def _spinrow(self, parent: ttk.Frame, *, row: int, label: str,
                 var: tk.IntVar, from_: int, to: int,
                 increment: int = 1, info: str = "") -> None:
        """Insert one label + spinbox + info row into a grid frame.

        Parameters
        ----------
        parent : ttk.Frame
            Frame to grid into.
        row : int
            Grid row.
        label : str
            Left-column label text.
        var : tk.IntVar
            Control variable.
        from_, to : int
            Spinbox range.
        increment : int, optional
            Step size (default 1).
        info : str, optional
            Descriptive text for the third column.
        """
        ttk.Label(parent, text=label).grid(row=row, column=0,
                                           sticky=tk.W, padx=(0, 5))
        ttk.Spinbox(parent, textvariable=var, from_=from_, to=to,
                    increment=increment, width=8).grid(
            row=row, column=1, sticky=tk.W, padx=(0, 10))
        if info:
            ttk.Label(parent, text=info, font=('TkDefaultFont', 8),
                      foreground="darkblue").grid(row=row, column=2, sticky=tk.W)

    def _build_results_frame(self, parent: ttk.Frame) -> None:
        """Build the plot + export LabelFrame (bottom half of tab)."""
        rf = ttk.LabelFrame(parent, text="Significance Results", padding="5")
        rf.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(0, weight=1)

        self._fig = Figure(figsize=(12, 7), dpi=100)
        self._ax_placeholder = self._fig.add_subplot(111)
        self._ax_placeholder.text(0.5, 0.5, "Run analysis above to display results",
                                   ha='center', va='center',
                                   transform=self._ax_placeholder.transAxes,
                                   fontsize=12, color='gray')

        self._canvas = FigureCanvasTkAgg(self._fig, rf)
        self._canvas.get_tk_widget().grid(row=0, column=0,
                                          sticky=(tk.W, tk.E, tk.N, tk.S))
        toolbar_frame = ttk.Frame(rf)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        NavigationToolbar2Tk(self._canvas, toolbar_frame).update()

        btn_frame = ttk.Frame(rf)
        btn_frame.grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self._save_btn = ttk.Button(btn_frame, text="Save Plot",
                                    command=self._save_plot, state="disabled")
        self._save_btn.grid(row=0, column=0, padx=(0, 5))
        self._export_btn = ttk.Button(btn_frame, text="Export CSV",
                                      command=self._export_csv, state="disabled")
        self._export_btn.grid(row=0, column=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefill_from_te_results(self) -> None:
        """Pre-fill parameter controls from the latest TE analysis results.

        Reads ``app.state.te_results['metadata']`` (produced by the analysis
        tab) and populates file, target, inputs, tau, window, and history
        controls.  No-ops silently when ``te_results`` is ``None``.

        Notes
        -----
        The analysis tab must store a top-level ``'metadata'`` key in
        ``te_results`` containing at minimum: ``data_file``, ``target_var``,
        ``input_vars``, ``tau``, ``window_days``, ``history_length``.
        The optional ``tau_dict`` value is cached for use during dispatch.
        """
        te = self._app.state.te_results
        if te is None:
            messagebox.showinfo("Info", "No TE results available yet.")
            return
        meta = te.get('metadata', {})
        if not meta:
            messagebox.showinfo("Info", "TE results have no metadata block.")
            return

        if 'data_file' in meta:
            self._file_var.set(Path(meta['data_file']).name)
        if 'target_var' in meta:
            self._target_var.set(meta['target_var'])
        if 'input_vars' in meta:
            self._inputs_var.set(", ".join(meta['input_vars']))
        if 'tau' in meta:
            self._tau_var.set(int(meta['tau']))
        if 'window_days' in meta:
            self._window_var.set(int(meta['window_days']))
        if 'history_length' in meta:
            self._history_var.set(int(meta['history_length']))
        self._tau_dict_cache = meta.get('tau_dict')

    # ------------------------------------------------------------------
    # Parameter collection
    # ------------------------------------------------------------------

    def _collect_params(self) -> Optional[Dict]:
        """Validate GUI controls and return a params dict.

        Returns
        -------
        params : dict or None
            None when validation fails (an error dialog is shown to the user).
        """
        data_file_name = self._file_var.get().strip()
        if not data_file_name:
            messagebox.showerror("Error", "Select a data file.")
            return None

        data_file = self._app.state.data_dir / data_file_name
        if not data_file.exists():
            messagebox.showerror("Error", f"File not found:\n{data_file}")
            return None

        target_var = self._target_var.get().strip()
        if not target_var:
            messagebox.showerror("Error", "Enter a target variable name.")
            return None

        raw_inputs = self._inputs_var.get()
        input_vars: List[str] = [v.strip() for v in raw_inputs.split(",")
                                  if v.strip()]
        if not input_vars:
            messagebox.showerror("Error", "Enter at least one input variable.")
            return None

        return {
            'data_file': str(data_file),
            'target_var': target_var,
            'input_vars': input_vars,
            'tau': self._tau_var.get(),
            'window_days': self._window_var.get(),
            'history_length': self._history_var.get(),
            'n_surrogates': self._n_surr_var.get(),
            'surrogate_type': self._cfg.surrogate_type,   # always 'iaaft'
            'time_col': self._cfg.time_col,
            'jte_mode': len(input_vars) >= 2,
            'tau_dict': self._tau_dict_cache,
        }

    # ------------------------------------------------------------------
    # Analysis thread lifecycle
    # ------------------------------------------------------------------

    def _start_analysis(self) -> None:
        """Validate parameters, then launch the surrogate engine in a daemon thread."""
        params = self._collect_params()
        if params is None:
            return

        self._run_btn.config(state="disabled")
        self._save_btn.config(state="disabled")
        self._export_btn.config(state="disabled")
        self._progress_bar.config(mode='indeterminate')
        self._progress_bar.start()

        label = "JTE" if params['jte_mode'] else "TE"
        self._progress_var.set(
            f"Starting IAAFT surrogate significance test ({label} mode)...")

        self._sig_results = None
        self._sig_error = None

        self._sig_thread = threading.Thread(
            target=self._run_surrogate_engine,
            args=(params,),
            daemon=True,
        )
        self._sig_thread.start()
        self._root.after(1000, self._poll_thread)

    def _run_surrogate_engine(self, params: Dict) -> None:
        """Execute surrogate analysis in the background thread.

        Parameters
        ----------
        params : dict
            Validated parameter dict from :meth:`_collect_params`.
        """
        try:
            analyzer = SurrogateAnalyzer(
                confidence_level=self._cfg.significance_percentile / 100.0
            )

            def _progress(msg: str) -> None:
                try:
                    self._root.after(0, lambda m=msg: self._progress_var.set(m))
                except Exception:
                    pass

            _progress("Initializing IAAFT surrogate analysis...")

            if params['jte_mode']:
                # PHYSICS STEP 1: joint surrogate significance across all inputs
                results = analyzer.calculate_jte_surrogate_confidence_intervals(
                    data_file=params['data_file'],
                    target_var=params['target_var'],
                    input_vars=params['input_vars'],
                    window_days=params['window_days'],
                    tau=params['tau'],
                    time_col=params['time_col'],
                    n_surrogates=params['n_surrogates'],
                    surrogate_type=params['surrogate_type'],
                    tau_dict=params['tau_dict'],
                    progress_callback=_progress,
                    history_length=params['history_length'],
                )
            else:
                # PHYSICS STEP 1: per-pair surrogate significance
                results = analyzer.calculate_surrogate_confidence_intervals(
                    data_file=params['data_file'],
                    target_var=params['target_var'],
                    input_vars=params['input_vars'],
                    window_days=params['window_days'],
                    tau=params['tau'],
                    time_col=params['time_col'],
                    n_surrogates=params['n_surrogates'],
                    surrogate_type=params['surrogate_type'],
                    tau_dict=params['tau_dict'],
                    progress_callback=_progress,
                    history_length=params['history_length'],
                )

            self._sig_results = results

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._sig_error = str(exc)

    def _poll_thread(self) -> None:
        """Poll the background thread; reschedule or finalize via root.after()."""
        if self._sig_thread and self._sig_thread.is_alive():
            self._root.after(1000, self._poll_thread)
            return

        self._progress_bar.stop()
        self._progress_bar.config(mode='determinate')
        self._run_btn.config(state="normal")

        if self._sig_results:
            self._on_analysis_complete()
        else:
            err = self._sig_error or "Surrogate analysis failed (unknown error)"
            self._progress_var.set("Analysis failed.")
            messagebox.showerror("IAAFT Error", err)

    def _on_analysis_complete(self) -> None:
        """Handle successful completion: update progress and render plot."""
        self._progress_var.set("IAAFT surrogate significance test complete.")
        self._save_btn.config(state="normal")
        self._export_btn.config(state="normal")
        render_significance_plot(self._fig, self._sig_results)
        self._canvas.draw()

    def _refresh_file_list(self) -> None:
        """Populate the file dropdown from the current data directory."""
        data_dir: Path = self._app.state.data_dir
        csv_files = sorted(p.name for p in data_dir.glob("*.csv"))
        self._file_combo['values'] = csv_files
        if csv_files and not self._file_var.get():
            self._file_var.set(csv_files[0])

    def _save_plot(self) -> None:
        """Save the current significance plot to PNG and PDF in OUTPUT_DIR.

        Fixed canvas dimensions are preserved: ``bbox_inches`` is never set to
        ``'tight'`` (visualization standard).
        """
        if not self._sig_results:
            messagebox.showwarning("Warning", "No results to save.")
            return
        try:
            stem = save_significance_plot(self._fig, self._sig_results, OUTPUT_DIR)
            messagebox.showinfo("Saved",
                                f"Plot saved to output_plots/\n{stem}.png / .pdf")
        except Exception as exc:
            messagebox.showerror("Error", f"Save failed: {exc}")

    def _export_csv(self) -> None:
        """Export significance time series to CSV in OUTPUT_DIR."""
        if not self._sig_results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        try:
            out_path = export_significance_csv(self._sig_results, OUTPUT_DIR)
            messagebox.showinfo("Exported", f"Data saved to:\n{out_path.name}")
        except Exception as exc:
            messagebox.showerror("Error", f"Export failed: {exc}")
