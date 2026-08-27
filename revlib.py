"""Shared TensorFlow training code for the JJCIT revision experiments.

Everything that must stay constant between experimental arms lives here, so that
e.g. the official-split and patient-disjoint-split runs differ only in which
rows are labelled train/test.
"""
import numpy as np

PATCH, BATCH = 224, 16
SEEDS = [2021, 7, 123, 42, 2024, 5, 777, 31337, 99, 1234]


def rot(g, a):
    import cv2
    if a == 0:
        return g
    return cv2.rotate(g, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                          270: cv2.ROTATE_90_COUNTERCLOCKWISE}[a])


def make_ds(tf):
    class DS(tf.keras.utils.Sequence):
        """RAW [0,255] patches; EfficientNet carries its own normalisation."""

        def __init__(self, X, y, augment, shuffle, patch=PATCH):
            self.X, self.y = X, y.astype("float32")
            self.augment, self.shuffle, self.patch = augment, shuffle, patch
            self.idx = np.arange(len(X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / BATCH))

        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.idx)

        def __getitem__(self, b):
            ids = self.idx[b * BATCH:(b + 1) * BATCH]
            out = np.empty((len(ids), self.patch, self.patch, 3), "float32")
            for j, i in enumerate(ids):
                g = self.X[i]
                if self.augment:
                    g = rot(g, np.random.choice([0, 90, 180, 270]))
                    if np.random.rand() < 0.5:
                        g = np.fliplr(g)
                out[j] = np.stack([g.astype("float32")] * 3, -1)
            return out, self.y[ids]

    return DS


def as_batch(X):
    return np.stack([np.stack([g.astype("float32")] * 3, -1) for g in X])


def build_effb0(tf, patch=PATCH):
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
    from tensorflow.keras.models import Model
    base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(patch, patch, 3))
    base.trainable = False
    inp = Input((patch, patch, 3))
    x = base(inp, training=False)
    x = Dropout(0.4)(GlobalAveragePooling2D()(x))
    return Model(inp, Dense(1, activation="sigmoid")(x)), base


def train_effb0(tf, Xtr, ytr, Xva, yva, seed, epochs=(5, 20), patch=PATCH):
    """Two-phase transfer learning, exactly as in the submitted pipeline."""
    from tensorflow.keras.callbacks import EarlyStopping
    np.random.seed(seed)
    tf.random.set_seed(seed)
    DS = make_ds(tf)
    cw = {0: len(ytr) / (2 * (ytr == 0).sum()), 1: len(ytr) / (2 * (ytr == 1).sum())}
    m, base = build_effb0(tf, patch)
    m.compile(tf.keras.optimizers.Adam(1e-3), "binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    m.fit(DS(Xtr, ytr, True, True, patch), validation_data=DS(Xva, yva, False, False, patch),
          epochs=epochs[0], class_weight=cw, verbose=0)
    base.trainable = True
    for l in base.layers[:-40]:
        l.trainable = False
    m.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="auc")])
    m.fit(DS(Xtr, ytr, True, True, patch), validation_data=DS(Xva, yva, False, False, patch),
          epochs=epochs[1], class_weight=cw, verbose=0,
          callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                   restore_best_weights=True)])
    return m


def load_effb0(tf, weights_path, patch=PATCH):
    """Rebuild the fine-tuned model and restore saved weights.

    Keras stores HDF5 weights against the layer structure as it stood when
    `save_weights` was called, which was *after* phase 2 unfroze the top 40
    backbone layers. Rebuilding with a frozen base therefore fails to load, so
    the trainable state at save time has to be replicated first.
    """
    m, base = build_effb0(tf, patch)
    base.trainable = True
    for l in base.layers[:-40]:
        l.trainable = False
    m.load_weights(weights_path)
    return m, base


def patient_disjoint_split(cb):
    """Repair the CBIS-DDSM official split.

    The official mass and calc partitions were drawn independently. Each is
    internally patient-disjoint, but 31 patients contribute a mass case to one
    side and a calc case to the other, so pooling mass+calc -- as this study and
    much prior work does -- silently reintroduces patient-level leakage.

    The repair keeps the official test set intact, so results stay comparable
    with published CBIS-DDSM numbers, and removes the offending patients from
    training instead.

    Returns a copy of `cb` with a `split_pd` column in {train, test, dropped}.
    """
    out = cb.copy()
    test_patients = set(out.loc[out["split"] == "test", "group"])
    out["split_pd"] = out["split"]
    clash = (out["split"] == "train") & (out["group"].isin(test_patients))
    out.loc[clash, "split_pd"] = "dropped"
    return out, int(clash.sum()), sorted(
        set(out.loc[out["split"] == "train", "group"]) & test_patients)
