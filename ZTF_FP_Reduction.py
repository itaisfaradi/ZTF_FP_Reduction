import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from io import StringIO
from typing import Optional

VALID_FILTERS = ['ZTF_g', 'ZTF_r', 'ZTF_i']
UNC_FACTOR = 1.0857
FILTER_STYLE: dict[str, dict] = {
    'ZTF_g': {'color': '#2ca02c', 'label': 'ZTF g'},
    'ZTF_r': {'color': '#d62728', 'label': 'ZTF r'},
    'ZTF_i': {'color': '#ff7f0e', 'label': 'ZTF i'},
}


def load_ztf_fp_file(filepath: str, quality_cut: bool = True) -> pd.DataFrame:
    """Load a ZTF forced photometry .txt file into a DataFrame.
    Skips '#' comment lines; first non-comment line is the column header.
    quality_cut: keep only diffimgstatus==1 and procstatus==0.
    """
    with open(filepath) as f:
        lines = [l for l in f if not l.strip().startswith('#')]
    columns = [c.strip() for c in lines[0].split(',')]
    df = pd.read_csv(StringIO(''.join(lines[1:])), sep=r'\s+', header=None, names=columns)
    if quality_cut:
        df = df[(df['diffimgstatus'] == 1) & (df['procstatus'] == 0)].reset_index(drop=True)
    return df


class ZTF_FP_Extractor():
    def __init__(self, name: str, df: pd.DataFrame):
        self.df = df
        self.name = name
        self._baseline_window: Optional[list] = None
        self._baseline_mag: Optional[dict[str, float]] = None
        self._span = None  # holds SpanSelector reference to prevent GC

    @property
    def baseline_window(self) -> Optional[list]:
        return self._baseline_window

    def _set_raw_mag_column(self,
                            mag_id: str = 'raw_mag',
                            flux_id: str = 'forcediffimflux',
                            diff_id: str = 'zpdiff') -> None:
        self.df[mag_id] = np.where(
            self.df[flux_id] > 0,
            self.df[diff_id] - 2.5 * np.log10(self.df[flux_id]),
            np.nan
        )

    def _set_raw_magerr_column(self,
                               magerr_id: str = 'raw_magerr',
                               flux_id: str = 'forcediffimflux',
                               unc_id: str = 'forcediffimfluxunc') -> None:
        self.df[magerr_id] = np.where(
            self.df[flux_id] > 0,
            UNC_FACTOR * self.df[unc_id] / self.df[flux_id],
            np.nan
        )

    def get_single_filter_df(self,
                             required_filter: str,
                             filter_id: str = 'filter') -> pd.DataFrame:
        assert required_filter in VALID_FILTERS, \
            f"Not a valid filter, must be one of {VALID_FILTERS}"
        return self.df[self.df[filter_id] == required_filter]

    def calculate_baseline(self, y: np.ndarray) -> tuple[float, float]:
        valid = y[~np.isnan(y)]
        return np.mean(valid), np.std(valid)

    def mag2flux_ujy(self,
                 mag: np.ndarray) -> np.ndarray:
        return 10**((23.9 - mag) / 2.5)

    
    def flux_ujy2mag(self,
                 flux: np.ndarray) -> np.ndarray:
        return 23.9 - 2.5 * np.log10(flux)

    def correct_baseline(self,
                         time_window: list,
                         corrected_mag_id: str = 'baseline_corrected_mag',
                         raw_mag_id: str = 'raw_mag',
                         time_id: str = 'jd',
                         filter_id: str = 'filter',
                         ccdid_id: str = 'ccdid',
                         field_id: str = 'field') -> None:
        assert time_window[0] < time_window[1], \
            "Window start date is not smaller than window end date"

        if corrected_mag_id not in self.df.columns:
            self.df[corrected_mag_id] = np.nan

        baseline_fluxes: dict[str, list[float]] = {}
        groups = self.df.groupby([filter_id, field_id, ccdid_id])
        for (filt, field, ccdid), group_idx in groups.groups.items():
            group = self.df.loc[group_idx]
            t = group[time_id].values
            time_mask = (t >= time_window[0]) & (t <= time_window[1])
            y_flux = self.mag2flux_ujy(self.df.loc[group_idx, raw_mag_id].values)
            valid_in_window = y_flux[time_mask]
            valid_in_window = valid_in_window[~np.isnan(valid_in_window)]
            if len(valid_in_window) < 1:
                print(f"Warning: no valid baseline data for {filt}, field {field}, ccdid {ccdid} — skipping")
                continue
            baseline_flux, _ = self.calculate_baseline(y=y_flux[time_mask])
            y_corrected = y_flux - baseline_flux
            self.df.loc[group_idx, corrected_mag_id] = self.flux_ujy2mag(y_corrected)
            baseline_fluxes.setdefault(filt, []).append(baseline_flux)

        self._baseline_mag = {
            filt: self.flux_ujy2mag(float(np.mean(fluxes)))
            for filt, fluxes in baseline_fluxes.items()
        }

    def get_lc_per_filter(self,
                          filter: str,
                          time_id: str = 'jd',
                          mag_id: str = None,
                          magerr_id: str = None,
                          use_corrected: bool = True,
                          use_err: bool = True):
        if use_corrected and mag_id is None:
            mag_id = 'baseline_corrected_mag'
        elif not use_corrected and mag_id is None:
            mag_id = 'raw_mag'
        if use_err and magerr_id is None:
            magerr_id = 'raw_magerr'
        filtered_df = self.get_single_filter_df(required_filter=filter)
        if use_err:
            return (filtered_df[time_id].values,
                    filtered_df[mag_id].values,
                    filtered_df[magerr_id].values)
        return filtered_df[time_id].values, filtered_df[mag_id].values

    def _available_filters(self, filter_id: str = 'filter') -> list[str]:
        return [f for f in VALID_FILTERS if f in self.df[filter_id].unique()]

    def plot_lc(self,
                ax: plt.Axes,
                use_corrected: bool = False,
                time_id: str = 'jd') -> None:
        if use_corrected and 'baseline_corrected_mag' not in self.df.columns:
            raise ValueError("Run correct_baseline() before plotting corrected magnitudes.")
        for filt in self._available_filters():
            style = FILTER_STYLE[filt]
            t, m, merr = self.get_lc_per_filter(
                filter=filt, time_id=time_id,
                use_corrected=use_corrected, use_err=True)
            valid = ~np.isnan(m)
            if valid.sum() == 0:
                continue
            ax.errorbar(t[valid], m[valid], yerr=merr[valid],
                        fmt='o', mec='k', color=style['color'],
                        label=style['label'], ms=4, lw=0.8,
                        capsize=2, elinewidth=0.8)
            if use_corrected and self._baseline_mag is not None and filt in self._baseline_mag:
                ax.axhline(self._baseline_mag[filt], color=style['color'],
                           ls='--', lw=0.8, zorder=0, alpha=0.6)
        ax.invert_yaxis()
        ax.set_xlabel(f'{time_id} [days]')
        ax.set_ylabel('Magnitude')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend()

    def get_baseline_window(self, time_id: str = 'jd') -> None:
        """Plot raw LC and let the user drag to select a pre-transient baseline window.
        Requires %matplotlib widget (ipympl). Selected range stored in self.baseline_window.
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        self.plot_lc(ax, use_corrected=False, time_id=time_id)

        _spans: list = []
        self._baseline_window = [None, None]

        def on_select(xmin: float, xmax: float) -> None:
            self._baseline_window = [xmin, xmax]
            for s in _spans:
                s.remove()
            _spans.clear()
            _spans.append(ax.axvspan(xmin, xmax, alpha=0.25, color='gray', zorder=0))
            print(f"Baseline window: JD {xmin:.4f} — {xmax:.4f}")
            fig.canvas.draw_idle()

        self._span = SpanSelector(
            ax, on_select, 'horizontal',
            useblit=False,
            props=dict(alpha=0.15, facecolor='steelblue'),
        )
        plt.tight_layout()
        plt.show()
