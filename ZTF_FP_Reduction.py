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
        df = df[df['diffimgstatus'] == 1].reset_index(drop=True)
        df = df[df['procstatus'] == 0].reset_index(drop=True)
        df = df[df['infobitssci'] < 33554432].reset_index(drop=True)
        df = df[df['scisigpix'] <= 25].reset_index(drop=True)
        df = df[df['sciinpseeing'] <= 4].reset_index(drop=True)
    return df


class ZTF_FP_Extractor():
    def __init__(self, name: str, df: pd.DataFrame):
        self.df = df
        self.name = name
        self._baseline_window: Optional[list] = None
        self._baseline_flux: Optional[dict[str, float]] = None
        self._span = None  # holds SpanSelector reference to prevent GC

    @property
    def baseline_window(self) -> Optional[list]:
        return self._baseline_window

    def _set_flux_ujy_column(self,
                            flux_id: str = 'forcediffimflux',
                            unc_id: str = 'forcediffimfluxunc',
                            zp_id: str = 'zpdiff') -> None:
        pre_factor = 10 ** ((23.9 - self.df[zp_id]) / 2.5)
        self.df[flux_id + '_ujy'] = pre_factor * self.df[flux_id]
        self.df[unc_id + '_ujy'] = pre_factor * self.df[unc_id]

    def get_single_filter_df(self,
                             required_filter: str,
                             filter_id: str = 'filter') -> pd.DataFrame:
        assert required_filter in VALID_FILTERS, \
            f"Not a valid filter, must be one of {VALID_FILTERS}"
        return self.df[self.df[filter_id] == required_filter]

    def calculate_baseline(self, 
                           y: np.ndarray, 
                           yerr:np.ndarray) -> tuple[float, float]:
        valid = np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
        y = y[valid]
        yerr = yerr[valid]
        weights = 1.0 / (yerr ** 2.0)
        return (np.sum(y * weights) / np.sum(weights)), (1.0 / np.sqrt(np.sum(weights)))

    def mag2flux_ujy(self,
                 mag: np.ndarray) -> np.ndarray:
        return 10**((23.9 - mag) / 2.5)

    def fluximage2fluxujy(self,
                          zp: np.ndarray,
                          imageflux: np.ndarray,
                          imagefluxunc: np.ndarray):
        pre_factor = 10 ** ((23.9 - zp) / 2.5)
        return pre_factor * imageflux, pre_factor * imagefluxunc
    
    def flux_ujy2mag(self,
                 flux: np.ndarray) -> np.ndarray:
        return 23.9 - 2.5 * np.log10(flux)

    def correct_baseline(self,
                         time_window: list,
                         flux_id: str = 'forcediffimflux_ujy',
                         flux_unc_id: str = 'forcediffimfluxunc_ujy',
                         time_id: str = 'jd',
                         filter_id: str = 'filter',
                         ccdid_id: str = 'ccdid',
                         field_id: str = 'field',
                         qid_id: str = 'qid',
                         zp_id: str = 'zpdiff') -> None:
        assert time_window[0] < time_window[1], \
            "Window start date is not smaller than window end date"

        if flux_id + '_corrected' not in self.df.columns:
            self.df[flux_id + '_corrected'] = np.nan

        baseline_fluxes: dict[str, list[float]] = {}
        groups = self.df.groupby([filter_id, field_id, ccdid_id, qid_id])
        for (filt, field, ccdid, qid), group_idx in groups.groups.items():
            group = self.df.loc[group_idx]
            t = group[time_id].values
            time_mask = (t >= time_window[0]) & (t <= time_window[1])
            y_flux = self.df.loc[group_idx, flux_id].values
            yerr_flux = self.df.loc[group_idx, flux_unc_id].values
            valid_in_window = y_flux[time_mask]
            valid_in_window = valid_in_window[~np.isnan(valid_in_window)]
            if len(valid_in_window) < 1:
                print(f"Warning: no valid baseline data for {filt}, field {field}, ccdid {ccdid}, qud {qid} — skipping")
                continue
            baseline_flux, baseline_flux_unc = self.calculate_baseline(y=y_flux[time_mask], yerr=yerr_flux[time_mask])
            self.df.loc[group_idx, 'baseline_flux'] = baseline_flux
            self.df.loc[group_idx, 'baseline_flux_unc'] = baseline_flux_unc
            y_corrected = y_flux - baseline_flux
            self.df.loc[group_idx, flux_id + '_corrected'] = y_corrected
            self.df.loc[group_idx, flux_unc_id + '_corrected'] = np.sqrt((baseline_flux_unc)**2.0 + (yerr_flux)**2.0)
            baseline_fluxes.setdefault(filt, []).append(baseline_flux)

        self._baseline_flux = {
            filt: float(np.mean(fluxes))
            for filt, fluxes in baseline_fluxes.items()
        }

    def get_lc_per_filter(self,
                          filter: str,
                          x: str,
                          y: str,
                          yerr: str = None):
        filtered_df = self.get_single_filter_df(required_filter=filter)
        if yerr is not None:
            return (filtered_df[x].values,
                    filtered_df[y].values,
                    filtered_df[yerr].values)
        return filtered_df[x].values, filtered_df[y].values

    def _available_filters(self, filter_id: str = 'filter') -> list[str]:
        return [f for f in VALID_FILTERS if f in self.df[filter_id].unique()]

    def plot_lc(self,
                ax: plt.Axes,
                x_id: str = 'jd',
                y_id: str = 'forcediffimflux_ujy',
                yerr_id: str = 'forcediffimfluxunc_ujy',
                zp_id: str = 'zpdiff',
                plot_mag: bool = False,
                use_corrected: bool = False) -> None:
        if use_corrected:
            y_id += '_corrected'
            yerr_id += '_corrected'
        for filt in self._available_filters():
            style = FILTER_STYLE[filt]
            x, y, yerr = self.get_lc_per_filter(
                filter=filt, 
                x=x_id, y=y_id, 
                yerr=yerr_id)
            if not plot_mag:
                valid = ~np.isnan(y)
                if valid.sum() == 0:
                    continue
                x_plot = x[valid]
                y_plot = y[valid]
                yerr_plot = yerr[valid]
            else:
                valid = np.logical_and(~np.isnan(y), y > 0.0)
                if valid.sum() == 0:
                    continue
                x_plot = x[valid]
                y_plot = self.flux_ujy2mag(y[valid])
                yerr_plot = UNC_FACTOR * yerr[valid] / y[valid]
            ax.errorbar(x_plot, y_plot, yerr=yerr_plot,
                        fmt='o', mec='k', color=style['color'],
                        label=style['label'], ms=4, lw=0.8,
                        capsize=2, elinewidth=0.8)
            if self._baseline_flux is not None and filt in self._baseline_flux:
                if not plot_mag:
                    baseline_plot = self._baseline_flux[filt]
                    ax.axhline(baseline_plot, color=style['color'],
                            ls='--', lw=0.8, zorder=0, alpha=0.6)
                else:
                    baseline_plot = self.flux_ujy2mag(self._baseline_flux[filt])
                    ax.axhline(baseline_plot, color=style['color'],
                            ls='--', lw=0.8, zorder=0, alpha=0.6)
        if plot_mag:
            ax.invert_yaxis()
        ax.set_xlabel(f'{x_id} [days]')
        if not plot_mag:
            ax.set_ylabel('Flux')
        else:
            ax.set_ylabel('AB Mag')
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.legend()

    def get_baseline_window(self,
                            x_id: str = 'jd',
                            y_id: str = 'forcediffimflux_ujy',
                            yerr_id: str = 'forcediffimfluxunc_ujy',
                            plot_mag: bool = False) -> None:
        """Plot raw LC and let the user drag to select a pre-transient baseline window.
        Requires %matplotlib widget (ipympl). Selected range stored in self.baseline_window.
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        self.plot_lc(ax, x_id=x_id, y_id=y_id, yerr_id=yerr_id, plot_mag=plot_mag)

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
