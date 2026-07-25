"""Transfer Entropy Analysis tab.

Handles file selection, parameter controls, differential-lag variable
selection, background TE/JTE calculation (worker thread + ``root.after()``
polling), and an embedded matplotlib figure. Pure figure-building lives in
:mod:`te_explorer.gui.plots`; pure helpers in :mod:`te_explorer.gui.tab_helpers`.
This file exceeds 500 lines; the excess is justified by the irreducible count of
discrete tkinter widget groups (10 config rows, 2 scrollable lists, 3 mode
toggles, 2 worker threads, 2 plot dispatch paths, 2 entropy helpers).

Results written to ``app.state`` for the IAAFT Significance tab:
- ``te_results`` - ``{target: {input: [...]}, "metadata": {...}}``
- ``current_data`` - loaded ``pd.DataFrame``
- ``current_file`` - source CSV filename

Engine import note: ``import te_explorer.config`` injects ``core/`` and
``core/NPEET`` into ``sys.path``; the bare ``from TE_Calculator import ...``
then resolves correctly.
"""

from __future__ import annotations

import threading
import time
from multiprocessing import cpu_count
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import te_explorer.config  # side effect: NPEET path injection
from TE_Calculator import TECalculator, build_col_map, extract_source  # noqa: F401
from te_explorer.gui.plots import draw_te_stacked, draw_jte
from te_explorer.gui.tab_helpers import (
    load_dataframe, reshape_te_results, interp_to_length,
    infer_file_info, build_tau_description, build_history_description,
    build_workers_info, build_start_message, pack_jte_results,
)


class AnalysisTab:
    """Transfer Entropy analysis tab.

    Parameters
    ----------
    parent : tk.Frame
        Notebook frame that owns this tab.
    app : TEExplorerApp
        Application shell exposing ``app.root``, ``app.state``,
        and ``app.refresh_file_lists()``.
    """

    def __init__(self, parent: tk.Frame, app) -> None:
        self._app = app
        self._root = app.root
        self._state = app.state

        self._combo_thread: Optional[threading.Thread] = None
        self._combo_params: dict = {}
        self._combo_success: bool = False
        self._combo_error: str = ""
        self._entropy_cache: Dict[str, np.ndarray] = {}

        self._target_var: Optional[tk.StringVar] = None
        self._input_vars: Dict[str, tk.BooleanVar] = {}
        self._input_tau_vars: Dict[str, tk.IntVar] = {}
        self._input_tau_spinboxes: Dict[str, ttk.Spinbox] = {}
        self._input_tau_labels: Dict[str, ttk.Label] = {}
        self._available_variables: List[str] = []
        self._current_timestamps: Optional[pd.Series] = None
        self._current_te_matrix: dict = {}
        self._jte_results: dict = {}

        self._build_layout(parent)
        self.load_available_files()

    # ── Public API ─────────────────────────────────────────────────────────

    def load_available_files(self) -> None:
        """Rescan ``app.state.data_dir`` for CSV files and populate the combobox.

        Called by ``app.refresh_file_lists()`` and once at construction time.
        """
        names = [f.name for f in sorted(self._state.data_dir.glob("*.csv"))]
        self._file_combo["values"] = names
        print(f"[AnalysisTab] Found {len(names)} file(s) in {self._state.data_dir}")

    # ── Layout ────────────────────────────────────────────────────────────

    def _build_layout(self, parent: tk.Frame) -> None:
        self._build_config_frame(parent)
        self._build_variable_frame(parent)
        self._build_plot_frame(parent)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

    def _build_config_frame(self, parent: tk.Frame) -> None:
        cfg = ttk.LabelFrame(parent, text="Configuration", padding="10")
        cfg.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        c = self._state.config

        ttk.Label(cfg, text="Data File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self._file_var = tk.StringVar()
        self._file_combo = ttk.Combobox(cfg, textvariable=self._file_var,
                                        width=40, state="readonly")
        self._file_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self._file_combo.bind("<<ComboboxSelected>>", self._on_file_selected)
        self._file_info_var = tk.StringVar(value="Select a data file...")
        ttk.Label(cfg, textvariable=self._file_info_var,
                  foreground="blue").grid(row=0, column=2, sticky=tk.W)

        ttk.Label(cfg, text="Window Size (days):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self._window_var = tk.IntVar(value=c.window_days)
        ttk.Spinbox(cfg, from_=1, to=365, textvariable=self._window_var,
                    width=10).grid(row=1, column=1, sticky=tk.W, padx=(0, 10))

        ttk.Label(cfg, text="Time Column:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self._time_col_var = tk.StringVar(value=c.time_col)
        ttk.Entry(cfg, textvariable=self._time_col_var,
                  width=15).grid(row=2, column=1, sticky=tk.W, padx=(0, 10))

        self._build_config_params(cfg, c)

    def _build_config_params(self, cfg: ttk.LabelFrame, c) -> None:
        # Row 3 - tau.
        ttk.Label(cfg, text="Time Lag (τ):").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self._tau_var = tk.IntVar(value=c.tau_default)
        tau_cb = ttk.Combobox(cfg, textvariable=self._tau_var, width=10, state="readonly")
        tau_cb["values"] = c.tau_options
        tau_cb.set(c.tau_default)
        tau_cb.grid(row=3, column=1, sticky=tk.W, padx=(0, 10))
        self._tau_info_var = tk.StringVar(
            value=build_tau_description(c.tau_default, c.history_length))
        ttk.Label(cfg, textvariable=self._tau_info_var, foreground="darkgreen",
                  font=("TkDefaultFont", 8)).grid(row=3, column=2, sticky=tk.W)
        tau_cb.bind("<<ComboboxSelected>>", self._on_tau_changed)

        # Row 4 - parallel workers.
        n_default = max(1, cpu_count() - 2)
        ttk.Label(cfg, text="Parallel Workers:").grid(row=4, column=0, sticky=tk.W, padx=(0, 5))
        self._n_workers_var = tk.IntVar(value=n_default)
        w_spin = ttk.Spinbox(cfg, from_=1, to=cpu_count(),
                             textvariable=self._n_workers_var, width=10)
        w_spin.grid(row=4, column=1, sticky=tk.W, padx=(0, 10))
        self._workers_info_var = tk.StringVar(
            value=f"Cores available: {cpu_count()}, recommended: {n_default}")
        ttk.Label(cfg, textvariable=self._workers_info_var, foreground="darkblue",
                  font=("TkDefaultFont", 8)).grid(row=4, column=2, sticky=tk.W)
        w_spin.bind("<FocusOut>", self._on_workers_changed)
        w_spin.bind("<Return>", self._on_workers_changed)

        # Row 5 - history length.
        ttk.Label(cfg, text="History Length (h):").grid(row=5, column=0, sticky=tk.W, padx=(0, 5))
        self._history_var = tk.IntVar(value=c.history_length)
        h_spin = ttk.Spinbox(cfg, from_=0, to=50, textvariable=self._history_var, width=10)
        h_spin.grid(row=5, column=1, sticky=tk.W, padx=(0, 10))
        self._history_info_var = tk.StringVar(
            value=build_history_description(c.history_length))
        ttk.Label(cfg, textvariable=self._history_info_var, foreground="darkgreen",
                  font=("TkDefaultFont", 8)).grid(row=5, column=2, sticky=tk.W)
        h_spin.bind("<FocusOut>", self._on_history_changed)
        h_spin.bind("<Return>", self._on_history_changed)

        ttk.Label(cfg, text=f"KSG k = {c.ksg_k} (engine of record)",
                  foreground="gray", font=("TkDefaultFont", 8)).grid(
            row=6, column=0, columnspan=3, sticky=tk.W, pady=(2, 0))

        self._progress_var = tk.StringVar(value="Ready")
        ttk.Label(cfg, textvariable=self._progress_var).grid(
            row=7, column=0, columnspan=3, pady=(5, 0))
        self._progress_bar = ttk.Progressbar(cfg, mode="indeterminate")
        self._progress_bar.grid(row=8, column=0, columnspan=3,
                                sticky=(tk.W, tk.E), pady=(5, 0))

    def _build_variable_frame(self, parent: tk.Frame) -> None:
        vf = ttk.LabelFrame(parent, text="Variable Selection", padding="10")
        vf.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        self._build_variable_lists(vf)
        self._build_variable_buttons(vf)
        vf.columnconfigure(0, weight=1)
        vf.columnconfigure(1, weight=1)
        vf.rowconfigure(1, weight=1)

    def _build_variable_lists(self, vf: ttk.LabelFrame) -> None:
        # Target scrollable list.
        ttk.Label(vf, text="Target Variable:").grid(
            row=0, column=0, sticky=(tk.W, tk.N), padx=(0, 5))
        t_canvas = tk.Canvas(vf, height=200)
        t_scroll = ttk.Scrollbar(vf, orient="vertical", command=t_canvas.yview)
        self._target_frame = ttk.Frame(t_canvas)
        t_canvas.configure(yscrollcommand=t_scroll.set)
        t_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        t_scroll.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E), padx=(0, 5))
        t_canvas.create_window((0, 0), window=self._target_frame, anchor="nw")
        self._target_frame.bind(
            "<Configure>",
            lambda e: t_canvas.configure(scrollregion=t_canvas.bbox("all")))

        # Input scrollable list.
        ttk.Label(vf, text="Input Variables:").grid(
            row=0, column=1, sticky=(tk.W, tk.N), padx=(5, 0))
        i_canvas = tk.Canvas(vf, height=200)
        i_scroll = ttk.Scrollbar(vf, orient="vertical", command=i_canvas.yview)
        self._input_frame = ttk.Frame(i_canvas)
        i_canvas.configure(yscrollcommand=i_scroll.set)
        i_canvas.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        i_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E), padx=(5, 0))
        i_canvas.create_window((0, 0), window=self._input_frame, anchor="nw")
        self._input_frame.bind(
            "<Configure>",
            lambda e: i_canvas.configure(scrollregion=i_canvas.bbox("all")))

    def _build_variable_buttons(self, vf: ttk.LabelFrame) -> None:
        bf = ttk.Frame(vf)
        bf.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        self._update_btn = ttk.Button(bf, text="Update Plot",
                                      command=self._update_plot, state="disabled")
        self._update_btn.grid(row=0, column=0, padx=(0, 5))
        self._run_btn = ttk.Button(bf, text="Run Combination",
                                   command=self._start_calculation, state="disabled")
        self._run_btn.grid(row=0, column=1, padx=(0, 5))
        self._jte_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bf, text="JTE Mode", variable=self._jte_mode_var,
                        command=self._on_jte_mode_changed).grid(
            row=0, column=2, padx=(10, 0))
        self._show_entropy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bf, text="Show H(Y)", variable=self._show_entropy_var,
                        command=self._on_entropy_toggle).grid(
            row=0, column=3, padx=(10, 0))
        self._diff_lag_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bf, text="Diff. Lags", variable=self._diff_lag_var,
                        command=self._on_diff_lag_changed).grid(
            row=0, column=4, padx=(10, 0))

    def _build_plot_frame(self, parent: tk.Frame) -> None:
        pf = ttk.LabelFrame(parent, text="Transfer Entropy Plot", padding="5")
        pf.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        self._fig = Figure(figsize=(10, 6), dpi=100)
        ax = self._fig.add_subplot(111)
        ax.set_title("Select variables and click 'Run Combination'")
        ax.set_xlabel("Time")
        ax.set_ylabel("Transfer Entropy (bits)")
        self._canvas = FigureCanvasTkAgg(self._fig, pf)
        self._canvas.get_tk_widget().grid(row=0, column=0,
                                          sticky=(tk.W, tk.E, tk.N, tk.S))
        tb_frame = ttk.Frame(pf)
        tb_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        NavigationToolbar2Tk(self._canvas, tb_frame).update()
        pf.columnconfigure(0, weight=1)
        pf.rowconfigure(0, weight=1)
        pf.rowconfigure(1, weight=0)

    # ── File and variable handlers ─────────────────────────────────────────

    def _on_file_selected(self, event) -> None:
        file_name = self._file_var.get()
        if not file_name:
            return
        self._file_info_var.set(infer_file_info(file_name))
        self._entropy_cache.clear()
        self._populate_variables(file_name)

    def _populate_variables(self, file_name: str) -> None:
        """Read CSV headers and rebuild target/input widget lists.

        Parameters
        ----------
        file_name : str
            Name of the CSV file in ``app.state.data_dir``.
        """
        try:
            time_col = self._time_col_var.get()
            df_sample = pd.read_csv(self._state.data_dir / file_name, nrows=10)
            variables = [c for c in df_sample.columns
                         if c != time_col and pd.api.types.is_numeric_dtype(df_sample[c])]
            if not variables:
                self._progress_var.set("No numeric variables found in file.")
                return
        except Exception as exc:
            self._progress_var.set(f"Error reading file: {exc}")
            return

        for w in self._target_frame.winfo_children():
            w.destroy()
        for w in self._input_frame.winfo_children():
            w.destroy()
        self._input_vars.clear()
        self._input_tau_vars.clear()
        self._input_tau_spinboxes.clear()
        self._input_tau_labels.clear()

        self._target_var = tk.StringVar(value=variables[0])
        for i, var in enumerate(variables):
            ttk.Radiobutton(self._target_frame, text=var,
                            variable=self._target_var, value=var).grid(
                row=i, column=0, sticky=tk.W, pady=1)

        default_tau = self._tau_var.get()
        for i, var in enumerate(variables):
            self._add_input_row(var, i, default_tau)

        self._available_variables = variables
        self._run_btn.config(state="normal")
        self._update_btn.config(state="disabled")
        self._state.current_file = file_name
        self._progress_var.set(
            f"Ready: {len(variables)} variables. Select variables and run calculation.")

    def _add_input_row(self, var: str, row: int, default_tau: int) -> None:
        """Add one input variable checkbox + hidden τ spinbox to ``_input_frame``."""
        bv = tk.BooleanVar()
        ttk.Checkbutton(self._input_frame, text=var, variable=bv,
                        command=lambda v=var: self._on_input_toggled(v)).grid(
            row=row, column=0, sticky=tk.W, pady=1)
        self._input_vars[var] = bv
        lbl = ttk.Label(self._input_frame, text="τ:", font=("TkDefaultFont", 8))
        lbl.grid(row=row, column=1, sticky=tk.E, padx=(5, 2), pady=1)
        lbl.grid_remove()
        self._input_tau_labels[var] = lbl
        iv = tk.IntVar(value=default_tau)
        sp = ttk.Spinbox(self._input_frame, from_=0, to=24,
                         textvariable=iv, width=3, font=("TkDefaultFont", 8))
        sp.grid(row=row, column=2, sticky=tk.W, padx=(0, 5), pady=1)
        sp.grid_remove()
        self._input_tau_vars[var] = iv
        self._input_tau_spinboxes[var] = sp

    # ── Parameter change handlers ──────────────────────────────────────────

    def _on_tau_changed(self, event) -> None:
        self._tau_info_var.set(
            build_tau_description(self._tau_var.get(), self._history_var.get()))

    def _on_history_changed(self, event) -> None:
        self._history_info_var.set(build_history_description(self._history_var.get()))
        self._on_tau_changed(event)

    def _on_workers_changed(self, event) -> None:
        n_max = cpu_count()
        n = max(1, min(self._n_workers_var.get(), n_max))
        self._n_workers_var.set(n)
        self._workers_info_var.set(build_workers_info(n, n_max))

    def _on_jte_mode_changed(self) -> None:
        jte = self._jte_mode_var.get()
        self._progress_var.set(
            "JTE Mode: Select 2+ inputs to calculate joint transfer entropy."
            if jte else "Standard TE Mode: Calculate individual transfer entropies.")
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.set_title("Select variables and click 'Run Combination'")
        ax.set_xlabel("Time")
        ax.set_ylabel("Transfer Entropy (bits)")
        self._canvas.draw()

    def _on_entropy_toggle(self) -> None:
        if self._current_te_matrix or self._jte_results:
            self._update_plot()

    def _on_diff_lag_changed(self) -> None:
        enabled = self._diff_lag_var.get()
        self._progress_var.set(
            "Differential lagging: Set τ for each selected input variable."
            if enabled else "Using global τ for all variables.")
        self._update_tau_spinbox_visibility()

    def _on_input_toggled(self, var_name: str) -> None:
        self._update_tau_spinbox_visibility()

    def _update_tau_spinbox_visibility(self) -> None:
        diff_enabled = self._diff_lag_var.get()
        for var_name, sp in self._input_tau_spinboxes.items():
            lbl = self._input_tau_labels.get(var_name)
            selected = self._input_vars.get(var_name, tk.BooleanVar()).get()
            if diff_enabled and selected:
                if lbl:
                    lbl.grid()
                sp.grid()
            else:
                if lbl:
                    lbl.grid_remove()
                sp.grid_remove()

    # ── Calculation orchestration ──────────────────────────────────────────

    def _start_calculation(self) -> None:
        """Validate selections, package parameters, and launch the worker thread."""
        if not self._file_var.get():
            messagebox.showwarning("Warning", "Please select a data file.")
            return
        if self._target_var is None or not self._target_var.get():
            messagebox.showwarning("Warning", "Please select a target variable.")
            return
        target = self._target_var.get()
        selected_inputs = [v for v, s in self._input_vars.items()
                           if s.get() and v != target]
        if not selected_inputs:
            messagebox.showwarning(
                "Warning",
                "Please select at least one input variable different from the target.")
            return
        jte_mode = self._jte_mode_var.get()
        if jte_mode and len(selected_inputs) < 2:
            messagebox.showwarning("Warning", "JTE Mode requires at least 2 input variables.")
            return

        tau_dict = self._get_tau_dict()
        diff_lag_enabled = self._diff_lag_var.get()
        self._run_btn.config(state="disabled")
        self._progress_bar.start()
        self._progress_var.set(
            build_start_message(jte_mode, diff_lag_enabled, selected_inputs, target, tau_dict))
        self._combo_params = {
            "file_name": self._file_var.get(),
            "target_var": target,
            "input_vars": selected_inputs,
            "window_days": self._window_var.get(),
            "time_col": self._time_col_var.get(),
            "tau": self._tau_var.get(),
            "tau_dict": tau_dict,
            "n_workers": self._n_workers_var.get(),
            "jte_mode": jte_mode,
            "diff_lag_enabled": diff_lag_enabled,
            "history_length": self._history_var.get(),
        }
        worker = self._calc_jte_thread if jte_mode else self._calc_te_thread
        self._combo_thread = threading.Thread(target=worker, daemon=True)
        self._combo_thread.start()
        self._root.after(500, self._poll_calculation)

    def _poll_calculation(self) -> None:
        if self._combo_thread and self._combo_thread.is_alive():
            self._root.after(500, self._poll_calculation)
            return
        self._progress_bar.stop()
        self._run_btn.config(state="normal")
        if self._combo_success:
            self._on_calculation_success()
        else:
            self._progress_var.set("Calculation failed.")
            messagebox.showerror("Error", f"Calculation failed: {self._combo_error}")

    def _on_calculation_success(self) -> None:
        params = self._combo_params
        n = len(params.get("input_vars", []))
        jte_mode = params.get("jte_mode", False)
        diff_lag_enabled = params.get("diff_lag_enabled", False)
        tau_dict = params.get("tau_dict") or {}

        if jte_mode:
            self._progress_var.set(f"JTE calculation completed ({n} inputs).")
            msg = f"Calculated Joint TE for {n} inputs."
        elif diff_lag_enabled and tau_dict:
            tau_info = ", ".join(f"{v[:8]}:τ={t}" for v, t in list(tau_dict.items())[:3])
            if len(tau_dict) > 3:
                tau_info += "..."
            self._progress_var.set(f"Calculated {n} combinations with differential lags.")
            msg = f"Calculated {n} TE combinations with differential lags:\n{tau_info}"
        else:
            self._progress_var.set(f"Calculated {n} combinations successfully.")
            msg = f"Calculated {n} TE combinations."

        self._update_btn.config(state="normal")
        self._update_plot()
        messagebox.showinfo("Success", msg)

    # ── Background worker threads ──────────────────────────────────────────

    def _calc_te_thread(self) -> None:
        """Calculate rolling TE for selected combinations (daemon thread)."""
        try:
            params = self._combo_params
            df = load_dataframe(self._state.data_dir / params["file_name"],
                                params["time_col"])

            def _progress(msg: str) -> None:
                try:
                    self._root.after(0, lambda: self._progress_var.set(msg))
                except Exception:
                    pass

            calc = TECalculator(n_cores=params["n_workers"],
                                window_days=params["window_days"],
                                tau=params["tau"],
                                history_length=params["history_length"])
            t0 = time.time()
            raw = calc.run_partial_analysis(
                df, params["target_var"], params["input_vars"], params["time_col"],
                progress_callback=_progress, tau_dict=params.get("tau_dict"))
            print(f"[AnalysisTab] TE completed in {time.time() - t0:.1f}s")

            freq, window_points = calc.determine_data_frequency(df, params["time_col"])
            centers, _ = calc.create_rolling_windows(df, window_points)
            te_matrix = reshape_te_results(raw, params["target_var"])

            # PHYSICS STEP: write shared state (significance tab reads te_results).
            shared = dict(te_matrix)
            shared["metadata"] = {
                "data_file": params["file_name"],
                "target_var": params["target_var"],
                "input_vars": params["input_vars"],
                "tau": params["tau"],
                "window_days": params["window_days"],
                "history_length": params["history_length"],
                "tau_dict": params.get("tau_dict"),
            }
            self._state.current_data = df
            self._state.current_file = params["file_name"]
            self._state.te_results = shared
            self._current_timestamps = df[params["time_col"]].iloc[centers]
            self._current_te_matrix = te_matrix  # unwrapped, for plotting
            self._jte_results = {}
            self._combo_success = True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._combo_success = False
            self._combo_error = str(exc)

    def _calc_jte_thread(self) -> None:
        """Calculate Joint Transfer Entropy (daemon thread)."""
        try:
            params = self._combo_params
            df = load_dataframe(self._state.data_dir / params["file_name"],
                                params["time_col"])

            def _progress(msg: str) -> None:
                try:
                    self._root.after(0, lambda: self._progress_var.set(msg))
                except Exception:
                    pass

            calc = TECalculator(n_cores=params["n_workers"],
                                window_days=params["window_days"],
                                tau=params["tau"],
                                history_length=params["history_length"])
            jte_raw = calc.run_jte_rolling_analysis(
                df, params["target_var"], params["input_vars"], params["time_col"],
                progress_callback=_progress, tau_dict=params.get("tau_dict"))
            self._jte_results = pack_jte_results(jte_raw, params)
            self._state.current_data = df
            self._state.current_file = params["file_name"]
            self._state.te_results = None  # not applicable in JTE mode
            self._current_te_matrix = {}
            self._combo_success = True
            print(f"[AnalysisTab] JTE done. "
                  f"Mean JTE={jte_raw['synergy_stats']['mean_jte']:.4f} bits")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._combo_success = False
            self._combo_error = str(exc)

    # ── Plot dispatch ──────────────────────────────────────────────────────

    def _update_plot(self) -> None:
        if self._jte_mode_var.get():
            self._draw_jte_plot()
        else:
            self._draw_te_plot()

    def _draw_jte_plot(self) -> None:
        if not self._jte_results:
            messagebox.showwarning("Warning", "No JTE data. Run a JTE calculation first.")
            return
        try:
            meta = self._jte_results.get("metadata", {})
            entropy = self._maybe_get_entropy(
                meta.get("target_var", ""),
                len(self._jte_results.get("jte_timeseries", [])))
            draw_jte(self._fig, self._jte_results,
                     dataset_name=self._file_var.get(), entropy_values=entropy)
            self._canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to create JTE plot: {exc}")

    def _draw_te_plot(self) -> None:
        if not self._current_te_matrix or self._current_timestamps is None:
            messagebox.showwarning("Warning", "No TE data. Run a combination first.")
            return
        target = self._target_var.get() if self._target_var else ""
        inputs = [v for v, s in self._input_vars.items() if s.get() and v != target]
        if not inputs:
            messagebox.showwarning("Warning", "Please select at least one input variable.")
            return
        try:
            entropy = self._maybe_get_entropy(target, len(self._current_timestamps))
            tau_dict = self._combo_params.get("tau_dict") if self._combo_params else None
            draw_te_stacked(
                self._fig, pd.to_datetime(self._current_timestamps),
                self._current_te_matrix, target, inputs,
                tau=self._tau_var.get(), history_length=self._history_var.get(),
                window_days=self._window_var.get(), dataset_name=self._file_var.get(),
                tau_dict=tau_dict, entropy_values=entropy)
            self._canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to create TE plot: {exc}")

    # ── Entropy helpers ────────────────────────────────────────────────────

    def _maybe_get_entropy(self, target_var: str,
                           expected_len: int) -> Optional[np.ndarray]:
        """Return H(Y) when the Show H(Y) toggle is active, else ``None``.

        Parameters
        ----------
        target_var : str
            Target variable name.
        expected_len : int
            Required output array length.

        Returns
        -------
        entropy : np.ndarray or None
        """
        if not self._show_entropy_var.get() or not target_var:
            return None
        return self._get_target_entropy(target_var, expected_len)

    def _get_target_entropy(self, target_var: str,
                            expected_length: Optional[int] = None) -> Optional[np.ndarray]:
        """Get or compute rolling H(Y) with caching.

        Parameters
        ----------
        target_var : str
            Target variable name.
        expected_length : int, optional
            Interpolate to this length when the cached array differs.

        Returns
        -------
        entropy_values : np.ndarray or None
            Rolling H(Y) in bits per window, or ``None`` on error.
        """
        window_days = self._window_var.get()
        cache_key = f"{target_var}_{window_days}days"
        if cache_key in self._entropy_cache:
            cached = self._entropy_cache[cache_key]
            if expected_length is not None and len(cached) != expected_length:
                return interp_to_length(cached, expected_length)
            return cached
        try:
            df = load_dataframe(self._state.data_dir / self._file_var.get(),
                                self._time_col_var.get())
            calc = TECalculator(window_days=window_days)
            entropy_vals, _ = calc.calculate_entropy_rolling(
                df, target_var, self._time_col_var.get())
            self._entropy_cache[cache_key] = entropy_vals
            print(f"[AnalysisTab] H({target_var}): mean={np.mean(entropy_vals):.4f} bits")
            if expected_length is not None and len(entropy_vals) != expected_length:
                return interp_to_length(entropy_vals, expected_length)
            return entropy_vals
        except Exception as exc:
            print(f"[AnalysisTab] Entropy error: {exc}")
            return None

    # ── tau_dict accessor ──────────────────────────────────────────────────

    def _get_tau_dict(self) -> Optional[Dict[str, int]]:
        """Read per-variable tau values directly from spinbox widget text.

        Reads widget text rather than ``IntVar`` to capture edits not yet
        synced (user edits spinbox without pressing Enter).

        Returns
        -------
        tau_dict : dict or None
            ``{var_name: tau_int}`` for selected variables, or ``None`` when
            differential lagging is disabled.
        """
        if not self._diff_lag_var.get():
            return None
        result: Dict[str, int] = {}
        for var_name, bv in self._input_vars.items():
            if bv.get() and var_name in self._input_tau_spinboxes:
                try:
                    result[var_name] = int(self._input_tau_spinboxes[var_name].get())
                except (ValueError, TypeError):
                    result[var_name] = self._tau_var.get()
        return result or None
