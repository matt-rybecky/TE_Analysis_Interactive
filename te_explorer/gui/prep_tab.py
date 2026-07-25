"""Data Preparation tab: load any CSVs and make them analysis-ready.

The tab walks the user through the preparation chain of record, one button
per step, each enabled by the previous: load one or more CSVs (each with
its own native resolution and datetime column), align and merge them onto
one uniform time grid (Gaussian-weighted resampling), apply the beta
normalization (entropy-calibrated robust z-score), optionally encode
directional columns as shared-scale sin/cos pairs, and save the result
into ``data/`` where the analysis tab picks it up. A preview table and a
status log mirror every step. Computation runs in a background thread; the
GUI polls with ``root.after()``.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

import pandas as pd

from te_explorer.prep import (align_datasets, beta_normalize,
                              compute_beta_calibration, encode_circular)

_POLL_MS = 100
_PREVIEW_ROWS = 12


def detect_time_column(df: pd.DataFrame, preferred: str) -> Optional[str]:
    """Find the datetime column of a loaded CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Freshly loaded data.
    preferred : str
        Column name to try first (the configured default).

    Returns
    -------
    str or None
        Name of the first column that parses as datetimes, or None.
    """
    candidates = [preferred] + [c for c in df.columns if c != preferred]
    for col in candidates:
        if col not in df.columns:
            continue
        sample = df[col].dropna().head(20)
        if sample.empty or pd.api.types.is_numeric_dtype(sample):
            continue
        try:
            pd.to_datetime(sample)
            return col
        except (ValueError, TypeError):
            continue
    return None


class PrepTab:
    """Data preparation workflow tab.

    Parameters
    ----------
    parent : ttk.Frame
        Notebook frame this tab renders into.
    app : TEExplorerApp
        Application shell providing shared state and callbacks.
    """

    def __init__(self, parent: ttk.Frame, app) -> None:
        self.parent = parent
        self.app = app
        self.config = app.state.config

        # Loaded inputs: one (stem, data, time_col) triple per file.
        self._loaded: List[Tuple[str, pd.DataFrame, str]] = []
        self._aligned: Optional[pd.DataFrame] = None
        self._normalized: Optional[pd.DataFrame] = None
        self._calibration = None
        self._source_name = ''
        self._worker: Optional[threading.Thread] = None
        self._worker_result = None
        self._worker_error: Optional[Exception] = None

        self._build_layout()
        self._update_button_states()

    # ── UI construction ────────────────────────────────────────────────

    def _build_layout(self) -> None:
        """Two columns: step controls left, preview and log right."""
        controls = ttk.Frame(self.parent, padding="10")
        controls.grid(row=0, column=0, sticky=(tk.N, tk.W))
        display = ttk.Frame(self.parent, padding="10")
        display.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)

        self._build_step_controls(controls)
        self._build_preview(display)
        self._build_log(display)

    def _build_step_controls(self, parent: ttk.Frame) -> None:
        """Buttons and parameter widgets for the five preparation steps."""
        load_frame = ttk.LabelFrame(parent, text="1. Load", padding="8")
        load_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        self.load_button = ttk.Button(load_frame, text="Load CSV(s)...",
                                      command=self._on_load)
        self.load_button.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(load_frame, text="Time column:").grid(
            row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.time_col_var = tk.StringVar(value=self.config.time_col)
        self.time_col_combo = ttk.Combobox(
            load_frame, textvariable=self.time_col_var, width=18,
            state="disabled")
        self.time_col_combo.grid(row=2, column=0, sticky=tk.W)

        align_frame = ttk.LabelFrame(parent, text="2. Align and resample",
                                     padding="8")
        align_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        ttk.Label(align_frame, text="Interval (hours):").grid(
            row=0, column=0, sticky=tk.W)
        self.interval_var = tk.IntVar(value=self.config.interval_hours)
        ttk.Spinbox(align_frame, from_=1, to=48, width=6,
                    textvariable=self.interval_var).grid(
            row=0, column=1, sticky=tk.W, padx=(4, 0))
        ttk.Label(align_frame, text="Circular (deg) columns:").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        self.circular_list = self._make_column_list(align_frame, row=2)
        ttk.Label(align_frame, text="Flux (sum) columns:").grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        self.flux_list = self._make_column_list(align_frame, row=4)
        self.align_button = ttk.Button(align_frame, text="Align",
                                       command=self._on_align)
        self.align_button.grid(row=5, column=0, sticky=tk.W, pady=(6, 0))

        norm_frame = ttk.LabelFrame(parent, text="3. Beta normalize",
                                    padding="8")
        norm_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        self.cal_target_var = tk.StringVar()
        self.cal_input1_var = tk.StringVar()
        self.cal_input2_var = tk.StringVar()
        self._cal_vars = (self.cal_target_var, self.cal_input1_var,
                          self.cal_input2_var)
        self._cal_combos = []
        for row, (label, var) in enumerate(
                [("Calibration target:", self.cal_target_var),
                 ("Calibration input 1:", self.cal_input1_var),
                 ("Calibration input 2:", self.cal_input2_var)]):
            ttk.Label(norm_frame, text=label).grid(row=row, column=0,
                                                   sticky=tk.W)
            combo = ttk.Combobox(norm_frame, textvariable=var, width=18,
                                 state="readonly")
            combo.grid(row=row, column=1, sticky=tk.W, padx=(4, 0))
            self._cal_combos.append(combo)
        ttk.Label(norm_frame,
                  text=f"KSG k = {self.config.calibration_k} (unified with "
                       "engine)").grid(row=3, column=0, columnspan=2,
                                       sticky=tk.W, pady=(4, 0))
        self.normalize_button = ttk.Button(norm_frame, text="Normalize",
                                           command=self._on_normalize)
        self.normalize_button.grid(row=4, column=0, sticky=tk.W, pady=(6, 0))

        enc_frame = ttk.LabelFrame(parent,
                                   text="4. Circular encode (optional)",
                                   padding="8")
        enc_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 8))
        ttk.Label(enc_frame, text="Directional columns:").grid(
            row=0, column=0, sticky=tk.W)
        self.encode_list = self._make_column_list(enc_frame, row=1)
        self.encode_button = ttk.Button(enc_frame, text="Encode",
                                        command=self._on_encode)
        self.encode_button.grid(row=2, column=0, sticky=tk.W, pady=(6, 0))

        save_frame = ttk.LabelFrame(parent, text="5. Save", padding="8")
        save_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        self.save_button = ttk.Button(save_frame, text="Save to data/",
                                      command=self._on_save)
        self.save_button.grid(row=0, column=0, sticky=tk.W)

    def _make_column_list(self, parent: ttk.Frame, row: int) -> tk.Listbox:
        """Small multi-select listbox for column choices."""
        box = tk.Listbox(parent, selectmode=tk.MULTIPLE, height=4,
                         width=24, exportselection=False)
        box.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        return box

    def _build_preview(self, parent: ttk.Frame) -> None:
        """Scrollable table showing the head of the current dataset."""
        frame = ttk.LabelFrame(parent, text="Preview", padding="4")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=3)
        self.preview = ttk.Treeview(frame, show='headings', height=10)
        self.preview.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL,
                                command=self.preview.xview)
        xscroll.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.preview.configure(xscrollcommand=xscroll.set)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def _build_log(self, parent: ttk.Frame) -> None:
        """Read-only status log."""
        frame = ttk.LabelFrame(parent, text="Log", padding="4")
        frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S),
                   pady=(8, 0))
        parent.rowconfigure(1, weight=2)
        self.log_text = tk.Text(frame, height=8, width=80, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=yscroll.set)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    # ── step handlers ──────────────────────────────────────────────────

    def _on_load(self) -> None:
        """Step 1: load one or more CSVs, detecting each time column."""
        paths = filedialog.askopenfilenames(
            title="Select CSV data file(s)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")])
        if not paths:
            return

        loaded: List[Tuple[str, pd.DataFrame, str]] = []
        for path in paths:
            try:
                df = pd.read_csv(path)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                messagebox.showerror("Load failed", f"{path}:\n{exc}")
                return
            time_col = detect_time_column(df, self.config.time_col)
            if time_col is None:
                messagebox.showerror(
                    "Load failed",
                    f"No parseable datetime column found in {path}.")
                return
            loaded.append((Path(path).stem, df, time_col))

        self._loaded = loaded
        self._aligned = None
        self._normalized = None
        self._calibration = None
        self._source_name = (loaded[0][0] if len(loaded) == 1
                             else f"{loaded[0][0]}_merged")

        self._configure_time_override()
        self._fill_column_lists()
        self._show_preview(loaded[0][1])
        for stem, df, time_col in loaded:
            n_numeric = sum(pd.api.types.is_numeric_dtype(df[c])
                            for c in df.columns)
            self._log(f"Loaded {stem}: {len(df)} rows, "
                      f"{len(df.columns)} columns ({n_numeric} numeric), "
                      f"time column '{time_col}'.")
        if len(loaded) > 1:
            self._log(f"{len(loaded)} files will be merged onto one grid "
                      "at the align step.")
        self._update_button_states()

    def _configure_time_override(self) -> None:
        """Enable the time-column override for single-file loads only."""
        if len(self._loaded) == 1:
            _, df, time_col = self._loaded[0]
            self.time_col_combo.configure(values=list(df.columns),
                                          state="readonly")
            self.time_col_var.set(time_col)
        else:
            self.time_col_combo.configure(values=[], state="disabled")
            self.time_col_var.set("(auto, per file)")

    def _on_align(self) -> None:
        """Step 2: resample all loaded files onto one grid (background)."""
        interval = self.interval_var.get()
        circular = self._selected(self.circular_list)
        flux = self._selected(self.flux_list)
        out_time_col = self.config.time_col
        if len(self._loaded) == 1:
            # Single file: honor the user's time-column override.
            datasets = [(self._loaded[0][1], self.time_col_var.get())]
        else:
            datasets = [(df, tcol) for _, df, tcol in self._loaded]

        def task() -> pd.DataFrame:
            return align_datasets(
                datasets, interval_hours=interval, time_col=out_time_col,
                circular_cols=circular, flux_cols=flux,
                sigma_fraction=self.config.sigma_fraction,
                max_gap_hours=self.config.max_gap_hours)

        self._log(f"Aligning {len(datasets)} file(s) to a {interval}-hour "
                  f"grid (sigma = {self.config.sigma_fraction} x interval, "
                  f"gaps <= {self.config.max_gap_hours} h interpolated)...")
        self._run_async(task, self._align_done)

    def _align_done(self, result: pd.DataFrame) -> None:
        """Store the aligned dataset and prime the calibration choices."""
        self._aligned = result
        self._normalized = None
        self._calibration = None
        # The aligned output always carries the configured time column.
        self.time_col_combo.configure(state="disabled")
        self.time_col_var.set(self.config.time_col)
        self._show_preview(result)
        coverage = result.drop(columns=self.config.time_col).notna().mean()
        self._log(f"Aligned: {len(result)} grid points, "
                  f"{len(result.columns) - 1} columns, mean coverage "
                  f"{100 * coverage.mean():.1f}%.")
        numeric = [c for c in result.columns
                   if c != self.config.time_col]
        defaults = (self.config.calibration_target,
                    *self.config.calibration_inputs)
        for combo, var, default in zip(self._cal_combos, self._cal_vars,
                                       defaults):
            combo.configure(values=numeric)
            var.set(default if default in numeric else
                    (numeric[0] if numeric else ''))
        self.encode_list.delete(0, tk.END)
        for col in numeric:
            self.encode_list.insert(tk.END, col)
        self._update_button_states()

    def _on_normalize(self) -> None:
        """Step 3: entropy-calibrated beta normalization (background)."""
        target = self.cal_target_var.get()
        inputs = [self.cal_input1_var.get(), self.cal_input2_var.get()]
        aligned = self._aligned
        time_col = self.time_col_var.get()
        k = self.config.calibration_k
        base = self.config.entropy_base

        def task():
            calibration = compute_beta_calibration(
                aligned, target, inputs, k=k, base=base)
            normalized = beta_normalize(aligned, calibration.scale,
                                        time_col=time_col)
            return calibration, normalized

        self._log(f"Beta calibration: JMI({', '.join(inputs)} -> {target}), "
                  f"KSG k={k}, base {base:g}...")
        self._run_async(task, self._normalize_done)

    def _normalize_done(self, result) -> None:
        """Store the normalized dataset and report the calibration."""
        self._calibration, self._normalized = result
        cal = self._calibration
        self._show_preview(self._normalized)
        self._log(f"JMI = {cal.jmi_bits:.4f} bits, H = "
                  f"{cal.entropy_bits:.4f} bits, scale = 2^(JMI - H) = "
                  f"{cal.scale:.6f}. Applied to all numeric columns.")
        self._update_button_states()

    def _on_encode(self) -> None:
        """Step 4 (optional): encode directional columns as sin/cos."""
        cols = self._selected(self.encode_list)
        if not cols:
            messagebox.showinfo("Circular encode",
                                "Select at least one directional column.")
            return
        try:
            self._normalized = encode_circular(
                self._normalized, self._aligned, cols,
                self._calibration.scale)
        except ValueError as exc:
            messagebox.showerror("Circular encode failed", str(exc))
            return
        self._show_preview(self._normalized)
        self._log(f"Encoded {', '.join(cols)} as shared-scale sin/cos "
                  "pairs (originals dropped).")

    def _on_save(self) -> None:
        """Step 5: write the prepared dataset into data/."""
        out_path = (self.app.state.data_dir
                    / f"{self._source_name}_prepared.csv")
        self._normalized.to_csv(out_path, index=False)
        self._log(f"Saved {out_path} ({len(self._normalized)} rows). "
                  "Available in the analysis tab.")
        self.app.refresh_file_lists()

    # ── helpers ────────────────────────────────────────────────────────

    def _run_async(self, task: Callable, on_success: Callable) -> None:
        """Run ``task`` in a worker thread; deliver its result via after()."""
        self._set_all_buttons(tk.DISABLED)
        self._worker_result = None
        self._worker_error = None

        def worker() -> None:
            try:
                self._worker_result = task()
            except Exception as exc:  # surfaced to the user via messagebox
                self._worker_error = exc

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()
        self._poll_worker(on_success)

    def _poll_worker(self, on_success: Callable) -> None:
        """Poll the worker thread until it finishes."""
        if self._worker.is_alive():
            self.app.root.after(_POLL_MS, self._poll_worker, on_success)
            return
        if self._worker_error is not None:
            self._log(f"ERROR: {self._worker_error}")
            messagebox.showerror("Preparation step failed",
                                 str(self._worker_error))
            self._update_button_states()
            return
        on_success(self._worker_result)
        self._update_button_states()

    def _selected(self, box: tk.Listbox) -> List[str]:
        """Selected entries of a listbox."""
        return [box.get(i) for i in box.curselection()]

    def _fill_column_lists(self) -> None:
        """Populate the circular/flux lists from all loaded files."""
        numeric: List[str] = []
        for _, df, time_col in self._loaded:
            numeric.extend(c for c in df.columns if c != time_col
                           and pd.api.types.is_numeric_dtype(df[c])
                           and c not in numeric)
        for box in (self.circular_list, self.flux_list):
            box.delete(0, tk.END)
            for col in numeric:
                box.insert(tk.END, col)

    def _show_preview(self, df: pd.DataFrame) -> None:
        """Render the head of ``df`` in the preview table."""
        self.preview.delete(*self.preview.get_children())
        cols = list(df.columns)
        self.preview.configure(columns=cols)
        for col in cols:
            self.preview.heading(col, text=col)
            self.preview.column(col, width=110, stretch=False)
        for _, row in df.head(_PREVIEW_ROWS).iterrows():
            values = ['' if pd.isna(v) else
                      (f'{v:.4g}' if isinstance(v, float) else str(v))
                      for v in row]
            self.preview.insert('', tk.END, values=values)

    def _log(self, message: str) -> None:
        """Append a line to the status log."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _set_all_buttons(self, state: str) -> None:
        """Enable or disable every step button at once."""
        for button in (self.load_button, self.align_button,
                       self.normalize_button, self.encode_button,
                       self.save_button):
            button.configure(state=state)

    def _update_button_states(self) -> None:
        """Enable each step only when its prerequisite exists."""
        self.load_button.configure(state=tk.NORMAL)
        self.align_button.configure(
            state=tk.NORMAL if self._loaded else tk.DISABLED)
        self.normalize_button.configure(
            state=tk.NORMAL if self._aligned is not None else tk.DISABLED)
        encode_ready = (self._normalized is not None
                        and self._calibration is not None)
        self.encode_button.configure(
            state=tk.NORMAL if encode_ready else tk.DISABLED)
        self.save_button.configure(
            state=tk.NORMAL if self._normalized is not None else tk.DISABLED)
