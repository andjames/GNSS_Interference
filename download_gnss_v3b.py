"""
GNSS Interference Event Downloader & Analyzer v3
Correctly handles tar containing 15-min .crx.gz chunks in subdirectories.
"""

import netrc, requests, tarfile, gzip, shutil, io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import georinex as gr

# ── Config ────────────────────────────────────────────────────────────────────

STATIONS = [
    ("METG", "FIN"),
    ("MATE", "ITA"),
    ("REYK", "ISL"),   # Reykjavik Iceland — good for triangulation
    ("ONSA", "SWE"),   # Onsala Sweden
]

EVENTS = [
    {"year": 2021, "doy": 160, "label": "Jun 9 2021 — 58 stations affected"},
    {"year": 2021, "doy": 146, "label": "May 26 2021 — double burst"},
    {"year": 2025, "doy": 14,  "label": "Jan 14 2025 — 10dB recorded"},
]

# Only extract these UTC hours (paper shows interference during business hours)
HOURS_TO_EXTRACT = range(9, 15)  # Narrow window — interference typically 10-13 UTC

STATION_COORDS = {
    "METG": (60.22,  24.40),
    "MATE": (40.65,  16.70),
    "REYK": (64.14, -21.96),
    "ONSA": (57.40,  11.93),
}

BASE_URL = "https://cddis.nasa.gov/archive/gnss/data/highrate"
DATA_DIR = Path("./gnss_data")
DATA_DIR.mkdir(exist_ok=True)


# ── Auth ──────────────────────────────────────────────────────────────────────

def make_session():
    n = netrc.netrc()
    auth = n.authenticators("urs.earthdata.nasa.gov")
    session = requests.Session()
    session.auth = (auth[0], auth[2])
    session.rebuild_auth = lambda prepared, response: None
    return session


# ── Download ──────────────────────────────────────────────────────────────────

def download_tar(session, station_id, country, year, doy, dest_dir):
    filename = f"{station_id}00{country}_R_{year}{doy:03d}0000_01D_01S_MO.crx.tar"
    url = f"{BASE_URL}/{year}/{doy:03d}/{filename}"
    dest = Path(dest_dir) / filename

    if dest.exists() and dest.stat().st_size > 10000:
        print(f"  Already downloaded: {filename}")
        return dest

    print(f"  Downloading {filename}...")
    r = session.get(url, allow_redirects=True, timeout=120, stream=True)
    if r.status_code != 200:
        print(f"  ✗ HTTP {r.status_code}")
        return None
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"  ✓ {dest.stat().st_size/1024/1024:.1f} MB")
    return dest


# ── Extract & Parse ───────────────────────────────────────────────────────────

def extract_hours(tar_path, hours, dest_dir):
    """
    Extract only the .crx.gz files for the requested hours from the tar.
    Returns list of extracted .crx.gz paths.
    """
    dest_dir = Path(dest_dir)
    extracted = []

    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if not member.name.endswith('.crx.gz'):
                continue
            # Filename contains HHMM e.g. ...0600_15M... = hour 06
            fname = Path(member.name).name
            # Extract hour from filename: SSSS00CCC_R_YYYYDDDHHMM_...
            # Position of HHMM is characters 16-20 of filename
            try:
                hour = int(fname.split('_')[2][7:9])
            except (IndexError, ValueError):
                continue

            if hour not in hours:
                continue

            out_path = dest_dir / fname
            if not out_path.exists():
                data = tf.extractfile(member).read()
                with open(out_path, 'wb') as f:
                    f.write(data)
            extracted.append(out_path)

    extracted.sort()
    print(f"  Extracted {len(extracted)} 15-min files (hours {min(hours)}-{max(hours)} UTC)")
    return extracted


def parse_crx_gz(crx_gz_path):
    """Decompress .crx.gz and parse with georinex, return S1C DataFrame."""
    crx_path = crx_gz_path.with_suffix('')  # remove .gz
    if not crx_path.exists():
        with gzip.open(crx_gz_path, 'rb') as f_in:
            with open(crx_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    try:
        obs = gr.load(str(crx_path), meas=['S1C'], verbose=False)
        if 'S1C' not in obs.data_vars:
            return None
        df = obs['S1C'].to_pandas()
        gps_cols = [c for c in df.columns if str(c).startswith('G')]
        df = df[gps_cols].dropna(how='all')
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"    Parse error {crx_gz_path.name}: {e}")
        return None


def load_station_cnr(tar_path, station_id, event_dir):
    """Extract business hours and concatenate into one CNR DataFrame."""
    chunks = extract_hours(tar_path, HOURS_TO_EXTRACT, event_dir)
    if not chunks:
        return None

    dfs = []
    for chunk in chunks:
        df = parse_crx_gz(chunk)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        return None

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated()]
    print(f"  ✓ {len(combined)} epochs, {len(combined.columns)} GPS sats")
    return combined


# ── Detection ─────────────────────────────────────────────────────────────────

def detection_statistic(cnr_df, l=3):
    per_sat = {}
    for col in cnr_df.columns:
        s = cnr_df[col].dropna()
        if len(s) > 2 * l + 1:
            per_sat[col] = 0.5 * (s.shift(-l) - 2*s + s.shift(l))
    if not per_sat:
        return None
    return pd.DataFrame(per_sat).mean(axis=1)


# ── Export CSV ────────────────────────────────────────────────────────────────

def export_csv(event, station_data):
    """Export detection events with lat/lon for D3 mapping."""
    THRESHOLD = 2.5
    rows = []

    # Also export per-station mean CNR timeseries
    ts_frames = {}

    for station, cnr_df in station_data.items():
        lam = detection_statistic(cnr_df)
        mean_cnr = cnr_df.mean(axis=1)
        ts_frames[station] = mean_cnr

        if lam is None:
            continue
        hits = lam[lam > THRESHOLD]
        lat, lon = STATION_COORDS.get(station, (0, 0))
        for ts, val in hits.items():
            rows.append({
                'timestamp': ts.isoformat(),
                'station':   station,
                'lat':       lat,
                'lon':       lon,
                'magnitude': round(val, 3),
                'event':     event['label'],
            })
        if not hits.empty:
            print(f"  ⚡ {station}: {len(hits)} detections "
                  f"@ {hits.index[0].strftime('%H:%M:%S')} UTC  "
                  f"(max {hits.max():.1f} dB)")

    safe = event['label'].replace(' ','_').replace('—','').replace(',','')

    # Events CSV — for map dots
    if rows:
        events_csv = DATA_DIR / f"events_{safe}.csv"
        pd.DataFrame(rows).to_csv(events_csv, index=False)
        print(f"  ✓ Events CSV: {events_csv}")

    # Timeseries CSV — for line chart
    if ts_frames:
        ts_csv = DATA_DIR / f"timeseries_{safe}.csv"
        pd.DataFrame(ts_frames).to_csv(ts_csv)
        print(f"  ✓ Timeseries CSV: {ts_csv}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_event(event, station_data):
    n = len(station_data)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(15, 3.5*n), squeeze=False)
    fig.suptitle(
        f"GPS L1 CNR — {event['label']}\n"
        f"Left: signal strength per satellite  |  "
        f"Right: detection statistic (spike = interference burst)",
        fontsize=11, fontweight='bold'
    )

    for i, (station, cnr_df) in enumerate(station_data.items()):
        ax_l, ax_r = axes[i]

        for col in cnr_df.columns:
            ax_l.plot(cnr_df.index, cnr_df[col], alpha=0.2, lw=0.4, color='steelblue')
        mean_cnr = cnr_df.mean(axis=1)
        ax_l.plot(mean_cnr.index, mean_cnr, color='navy', lw=1.5, label='Mean CNR')
        ax_l.set_ylabel(f'{station}\nCNR (dB-Hz)', fontsize=9)
        ax_l.set_ylim(25, 60)
        ax_l.legend(fontsize=8)
        ax_l.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        lam = detection_statistic(cnr_df)
        if lam is not None:
            ax_r.plot(lam.index, lam, color='crimson', lw=0.8)
            ax_r.axhline(2.5, color='red', ls='--', lw=0.8, label='Threshold')
            hits = lam[lam > 2.5]
            if not hits.empty:
                ax_r.scatter(hits.index, hits.values,
                             color='red', s=40, zorder=5, label='Detection')
            ax_r.set_ylabel('Λ (dB)', fontsize=9)
            ax_r.legend(fontsize=8)
            ax_r.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        if i == n - 1:
            ax_l.set_xlabel('UTC Time')
            ax_r.set_xlabel('UTC Time')

    plt.tight_layout()
    safe = event['label'].replace(' ','_').replace('—','').replace(',','')
    out = DATA_DIR / f"plot_{safe}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plot: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GNSS Interference Analyzer v3")
    print("=" * 60)

    session = make_session()

    for event in EVENTS:
        print(f"\n{'─'*55}")
        print(f"Event: {event['label']}")

        event_dir = DATA_DIR / f"{event['year']}_{event['doy']:03d}"
        event_dir.mkdir(exist_ok=True)

        station_data = {}

        for station_id, country in STATIONS:
            print(f"\n  Station: {station_id}")
            tar = download_tar(session, station_id, country,
                               event['year'], event['doy'], event_dir)
            if tar is None:
                continue

            cnr_df = load_station_cnr(tar, station_id, event_dir)
            if cnr_df is not None:
                station_data[station_id] = cnr_df

        if station_data:
            print(f"\n  Detections:")
            export_csv(event, station_data)
            plot_event(event, station_data)
        else:
            print(f"\n  ✗ No data for this event")

    print(f"\n{'='*60}")
    print(f"Done. Check ./gnss_data/ for plots and CSVs.")


if __name__ == "__main__":
    main()
