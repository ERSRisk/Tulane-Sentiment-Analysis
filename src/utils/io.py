from pathlib import Path
import os
import pickle

def atomic_write_csv(path: str, df, compress: bool = False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if compress:
        df.to_csv(tmp, index=False, compression="gzip")
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, p)
    print(f"✅ Wrote {p} ({p.stat().st_size/1e6:.2f} MB)")
    
def atomic_write_pickle(path: str, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, p)
    print(f"✅ Wrote {p} ({p.stat().st_size/1e6:.2f} MB)")