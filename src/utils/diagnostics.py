import psutil
import os
import pandas as pd

PROC = psutil.Process(os.getpid())

def mem(msg: str):
    rss_gb = PROC.memory_info().rss / (1024**3)
    print(f"[MEM] {msg}: {rss_gb:.2f} GB", flush = True)

def debug_date(df, name, raw_col='Published', utc_col='Published_utc', link_col='Link'):
    try:
        if df is None:
            print(f"[DATCHECK] {name}: df is None", flush=True)
            return
        if len(df) == 0:
            print(f"[DATCHECK] {name}: EMPTY df", flush=True)
            return

        work = df.copy()

        def _dbg_coerce(x):
            if pd.isna(x):
                return pd.NaT
            if isinstance(x, (int, float)):
                if x > 1e12:
                    return pd.to_datetime(x, unit='ms', errors='coerce', utc=True)
                if x > 1e9:
                    return pd.to_datetime(x, unit='s', errors='coerce', utc=True)
            sx = str(x)
            sx = re.sub(r'\s(EST|EDT|PDT|CDT|MDT|GMT)\b', '', sx, flags=re.I)
            return pd.to_datetime(sx, errors='coerce', utc=True)

        if utc_col in work.columns:
            work[utc_col] = work[utc_col].apply(_dbg_coerce)
        elif raw_col in work.columns:
            work[utc_col] = work[raw_col].apply(_dbg_coerce)
        else:
            work[utc_col] = pd.NaT

        parsed = work[utc_col].notna().sum()
        total = len(work)
        min_dt = work[utc_col].min()
        max_dt = work[utc_col].max()

        print(
            f"[DATCHECK] {name}: rows={total}, parsed_dates={parsed}, min={min_dt}, max={max_dt}",
            flush=True
        )

        show_cols = [c for c in ['Title', link_col, raw_col] if c in work.columns]
        newest = work.sort_values(utc_col, ascending=False, na_position='last')[show_cols].head(5)
        print(f"[DATECHK] {name} newest rows:", flush=True)
        print(newest.to_string(index=False), flush=True)

    except Exception as e:
        print(f"[DATCHECK] {name}: FAILED with {e}", flush=True)