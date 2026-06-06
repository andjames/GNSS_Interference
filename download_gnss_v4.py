"""
GNSS Interference Event Downloader & Analyzer v4
- Handles both _R_ and _S_ filename variants
- Updated station list based on actual directory listing
"""

import netrc, requests, tarfile, gzip, shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import georinex as gr

# ── Config ────────────────────────────────────────────────────────────────────

# (station_id, country, source) — source is 'R' or 'S'
STATIONS = [
    ("METG", "FIN", "R"),   # Metsahovi, Finland
    ("MATE", "ITA", "R"),   # Matera, Italy
    ("ONSA", "SWE", "S"),   # Onsala, Sweden
    ("BRST", "FRA", "S"),   # Brest, France
    ("WSRT", "NLD", "R"),   # Westerbork, Netherlands
    ("BRUX", "BEL", "S"),   # Brussels, Belgium
]

EVENTS = [
    {"year": 2021, "doy": 160, "label": "Jun 9 2021 — 58 stations affected"},
    {"year": 2021, "doy": 146, "label": "May 26 2021 — double burst"},
    {"year": 2025, "doy": 14,  "label": "Jan 14 2025 — 10dB recorded"},
]

HOURS_TO_EXTRACT = range(9, 15)  # 09-14 UTC

STATION_COORDS = {
    "METG": (60.22,  24.40),
    "MATE": (40.65,  16.70),
    "ONSA": (57.40,  11.93),
    "BRST": (48.38,  -4.50),
    "WSRT": (52.91,   6.60),
    "BRUX": (50.80,   4.36),
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

def download_tar(session, station_id, country, source, year, doy, dest_dir):
    filename = f"{station_id}00{country}_{source}_{year}{doy:03d}0000_01D_01S_MO.crx.tar"
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
    dest_dir = Path(dest_dir)
    extracted = []
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if not member.name.endswith('.crx.gz'):
                continue
            fname = Path(member.name).name
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
    crx_path = crx_gz_path.with_suffix('')
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
        return None


def load_station_cnr(tar_path, station_id, event_dir):
    chunks = extract_hours(tar_path, HOURS_TO_EXTRACT, event_dir)
    if not chunks:
        return None
    dfs = [parse_crx_gz(c) for c in chunks]
    dfs = [d for d in dfs if d is not None and not d.empty]
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


# ── Export ────────────────────────────────────────────────────────────────────

def export_csv(event, station_data):
    THRESHOLD = 2.5
    rows = []
    ts_frames = {}

    for station, cnr_df in station_data.items():
        mean_cnr = cnr_df.mean(axis=1)
        ts_frames[station] = mean_cnr

        lam = detection_statistic(cnr_df)
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
        else:
            print(f"  — {station}: no detections above threshold")

    safe = event['label'].replace(' ','_').replace('—','').replace(',','')

    if rows:
        out = DATA_DIR / f"events_{safe}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  ✓ Events CSV: {out}")

    if ts_frames:
        out = DATA_DIR / f"timeseries_{safe}.csv"
        pd.DataFrame(ts_frames).to_csv(out, index_label='time')
        print(f"  ✓ Timeseries CSV: {out}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_event(event, station_data):
    n = len(station_data)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(15, 3*n), squeeze=False)
    fig.suptitle(
        f"GPS L1 CNR — {event['label']}\n"
        "Left: mean signal strength  |  Right: detection statistic",
        fontsize=11, fontweight='bold'
    )

    for i, (station, cnr_df) in enumerate(station_data.items()):
        ax_l, ax_r = axes[i]
        mean_cnr = cnr_df.mean(axis=1)
        ax_l.plot(mean_cnr.index, mean_cnr, color='steelblue', lw=0.8)
        ax_l.set_ylabel(f'{station}\ndB-Hz', fontsize=9)
        ax_l.set_ylim(35, 55)
        ax_l.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        lam = detection_statistic(cnr_df)
        if lam is not None:
            ax_r.plot(lam.index, lam, color='crimson', lw=0.8)
            ax_r.axhline(2.5, color='red', ls='--', lw=0.8, label='Threshold')
            hits = lam[lam > 2.5]
            if not hits.empty:
                ax_r.scatter(hits.index, hits.values,
                             color='red', s=40, zorder=5)
            ax_r.set_ylabel('Λ (dB)', fontsize=9)
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
    print("GNSS Interference Analyzer v4")
    print("=" * 60)

    session = make_session()

    for event in EVENTS:
        print(f"\n{'─'*55}")
        print(f"Event: {event['label']}")

        event_dir = DATA_DIR / f"{event['year']}_{event['doy']:03d}"
        event_dir.mkdir(exist_ok=True)

        station_data = {}

        for station_id, country, source in STATIONS:
            print(f"\n  Station: {station_id}")
            tar = download_tar(session, station_id, country, source,
                               event['year'], event['doy'], event_dir)
            if tar is None:
                continue
            cnr_df = load_station_cnr(tar, station_id, event_dir)
            if cnr_df is not None:
                station_data[station_id] = cnr_df

        if station_data:
            print(f"\n  Results:")
            export_csv(event, station_data)
            plot_event(event, station_data)
        else:
            print(f"\n  ✗ No data for this event")

    print(f"\n{'='*60}")
    print(f"Done. Check ./gnss_data/ for outputs.")


if __name__ == "__main__":
    main()
