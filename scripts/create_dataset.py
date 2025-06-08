import os, math, json, random
import numpy as np, pandas as pd

# ---------- config ----------
SRC_DIR      = './'   # where the 7 CSVs live
MAX_LEN      = 60            # pad / truncate every sequence to this length
N_POS, N_NEG = 20_000, 20_000
OUT_X, OUT_Y = 'data.npy', 'labels.npy'
SEED         = 42
# ----------------------------

def load_templates(base_dir):
    return {f: pd.read_csv(f'{base_dir}/{f}')
            for f in os.listdir(base_dir) if f.endswith('.csv')}

def augment(t0, y0, rng,
            time_scale=(0.7,1.3), amp=(0.8,1.2),
            yaw_offset=(-5,5), noise_sigma=(1,3),
            length=(40,60), flip_prob=0.0):
    y = np.array(y0, float).copy()
    if rng.random() < flip_prob:
        y = -y
    y *= rng.uniform(*amp)
    t = np.array(t0, float) * rng.uniform(*time_scale)

    L   = rng.integers(*length, endpoint=True)
    new_t = np.linspace(t.min(), t.max(), L)
    new_y = np.interp(new_t, t, y)
    new_y += rng.uniform(*yaw_offset)
    new_y += rng.normal(0, rng.uniform(*noise_sigma), size=L)
    return new_t, new_y

def make_dataset():
    rng  = np.random.default_rng(SEED)
    tpl  = load_templates(SRC_DIR)

    pos_df = tpl['normal_rotation_yaw_data.csv']
    neg_df = {k:v for k,v in tpl.items() if k != 'normal_rotation_yaw_data.csv'}

    X, y = [], []

    # ---------- positives ----------
    for _ in range(N_POS):
        t, yaw = augment(pos_df.Time, pos_df.Yaw, rng)
        X.append(pack(t, yaw))
        y.append(1)

    # ---------- negatives ----------
    each = math.ceil(N_NEG / len(neg_df))
    made = 0
    for name, df in neg_df.items():
        for _ in range(each):
            if made >= N_NEG:
                break
            amp  = (0.2,0.6) if any(k in name for k in ['static','no_rotation']) else (0.8,1.2)
            flip = 0.5       if any(k in name for k in ['left_only','right_only']) else 0.0
            t, yaw = augment(df.Time, df.Yaw, rng, amp=amp, flip_prob=flip)
            X.append(pack(t, yaw))
            y.append(0)
            made += 1

    X = np.stack(X).astype('float32')
    y = np.array(y, dtype='int8')
    np.save(OUT_X, X)
    np.save(OUT_Y, y)
    print(f"✅  Saved {X.shape[0]:,} sequences → {OUT_X}, {OUT_Y}")

def pack(t, yaw):
    """Pad / normalise time + yaw into (MAX_LEN, 2)."""
    L = len(yaw)
    arr = np.zeros((MAX_LEN, 2), dtype='float32')
    # time channel → 0-1 inside each sequence
    t_norm = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else t
    arr[:L, 0] = t_norm
    arr[:L, 1] = yaw
    return arr

if __name__ == '__main__':
    make_dataset()
