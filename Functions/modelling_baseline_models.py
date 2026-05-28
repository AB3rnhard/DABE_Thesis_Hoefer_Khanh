import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression as _PLS
from sklearn.ensemble import RandomForestRegressor as _RFR
from sklearn.feature_selection import VarianceThreshold as _VT
from sklearn.linear_model import LassoCV as _LassoCV
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.model_selection import TimeSeriesSplit as _TSS
from sklearn.pipeline import Pipeline as _SKPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


MAX_SAFE_PLS_COMPONENTS = 15
MODELS_REQUIRING_POLY_BLOCK = {'pls_ols', 'rf_pls', 'lstm_pls'}


try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential

    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as exc:
    tf = None
    EarlyStopping = None
    Dense = None
    Dropout = None
    Input = None
    LSTM = None
    Sequential = None
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = exc


def set_tf_seed(seed: int) -> None:
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


class PLSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, max_iter=500, scale=False):
        self.n_components = n_components
        self.max_iter = max_iter
        self.scale = scale

    def fit(self, X, y=None):
        self.pls_ = _PLS(
            n_components=self.n_components,
            max_iter=self.max_iter,
            scale=self.scale,
        )
        self.pls_.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'pls_')
        return self.pls_.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'pls_')
        return np.asarray([f'pls{i}' for i in range(self.pls_.n_components)], dtype=object)


def make_linear():
    return _SKPipeline([
        ('scaler', StandardScaler()),
        ('ols', _LinearRegression()),
    ])


def make_rf(rf_n_estimators: int, random_state: int):
    return _RFR(
        n_estimators=rf_n_estimators,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )


def _build_lstm(n_features: int, seq_len: int):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='linear'),
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


class LSTMRegressor:
    def __init__(self, seq_len: int, epochs: int, batch: int):
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch = batch
        self.xs = StandardScaler()
        self.ys = StandardScaler()
        self.model = None
        self._last_train_X = None

    def _seq(self, Xs, ys=None):
        X_out, y_out = [], []
        for idx in range(self.seq_len, len(Xs)):
            X_out.append(Xs[idx - self.seq_len:idx])
            if ys is not None:
                y_out.append(ys[idx])
        X_out = np.asarray(X_out)
        return (X_out, np.asarray(y_out)) if ys is not None else X_out

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        Xs = self.xs.fit_transform(X)
        ys = self.ys.fit_transform(y).ravel()
        Xseq, yseq = self._seq(Xs, ys)
        self.model = _build_lstm(X.shape[1], self.seq_len)
        early_stopping = EarlyStopping(patience=3, restore_best_weights=True, monitor='loss')
        self.model.fit(
            Xseq,
            yseq,
            epochs=self.epochs,
            batch_size=self.batch,
            verbose=0,
            callbacks=[early_stopping],
        )
        self._last_train_X = Xs[-self.seq_len:]
        return self

    def predict(self, X):
        X = np.asarray(X)
        Xs = self.xs.transform(X)
        Xs_ext = np.vstack([self._last_train_X, Xs]) if self._last_train_X is not None else Xs
        starts = range(self.seq_len, len(Xs_ext))
        if len(starts) == 0:
            return np.asarray([], dtype=float)
        windows = np.stack([Xs_ext[idx - self.seq_len:idx] for idx in starts]).astype('float32')
        preds = np.asarray(self.model(windows, training=False)).reshape(-1)
        return self.ys.inverse_transform(preds.reshape(-1, 1)).ravel()


def make_lstm(seq_len: int, epochs: int, batch: int):
    if not TF_AVAILABLE:
        return None
    return LSTMRegressor(seq_len=seq_len, epochs=epochs, batch=batch)


def make_pls_preprocessor(n_components: int, poly_cols: list, passthrough_cols: list):
    poly_cols = list(poly_cols or [])
    passthrough_cols = list(passthrough_cols or [])
    if not poly_cols:
        raise ValueError('PLS preprocessing requires at least one polymarket column.')

    safe_n = min(int(n_components), MAX_SAFE_PLS_COMPONENTS)
    transformers = [
        (
            'poly_pls',
            _SKPipeline([
                ('vt', _VT(threshold=1e-8)),
                ('scaler', StandardScaler()),
                ('pls', PLSTransformer(n_components=safe_n, max_iter=500, scale=False)),
            ]),
            poly_cols,
        ),
    ]
    if passthrough_cols:
        transformers.append(('passthru', 'passthrough', passthrough_cols))
    return ColumnTransformer(transformers)


def make_pls_ols(n_components: int, poly_cols: list, passthrough_cols: list):
    return _SKPipeline([
        ('preproc', make_pls_preprocessor(n_components, poly_cols, passthrough_cols)),
        ('ols', _LinearRegression()),
    ])


def make_lasso_cv(random_state: int):
    return _SKPipeline([
        ('scaler', StandardScaler()),
        ('lasso', _LassoCV(
            cv=_TSS(n_splits=3),
            n_alphas=20,
            max_iter=10_000,
            random_state=random_state,
        )),
    ])


def make_rf_pls(
    n_components: int,
    poly_cols: list,
    passthrough_cols: list,
    rf_n_estimators: int,
    random_state: int,
):
    return _SKPipeline([
        ('preproc', make_pls_preprocessor(n_components, poly_cols, passthrough_cols)),
        ('rf', _RFR(
            n_estimators=rf_n_estimators,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        )),
    ])


class LSTMPLSWrapper:
    def __init__(self, n_components: int, poly_cols: list, passthrough_cols: list, seq_len: int, epochs: int, batch: int):
        self.n_components = n_components
        self.poly_cols = list(poly_cols or [])
        self.passthrough_cols = list(passthrough_cols or [])
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch = batch
        self.transformer = None
        self.core = None

    def fit(self, X, y):
        self.transformer = make_pls_preprocessor(self.n_components, self.poly_cols, self.passthrough_cols)
        transformed = self.transformer.fit_transform(X, y)
        self.core = LSTMRegressor(seq_len=self.seq_len, epochs=self.epochs, batch=self.batch)
        self.core.fit(transformed, y)
        return self

    def predict(self, X):
        transformed = self.transformer.transform(X)
        return self.core.predict(transformed)


def build_model_registry(dataset_tag: str, dataset_spec: dict, config: dict) -> dict:
    base_model_builders = {}
    if config['run_linear']:
        base_model_builders['linear'] = lambda ds: make_linear()
    if config['run_rf']:
        base_model_builders['rf'] = lambda ds: make_rf(config['rf_n_estimators'], config['random_state'])
    if config['run_lstm'] and TF_AVAILABLE:
        base_model_builders['lstm'] = lambda ds: make_lstm(config['lstm_seq_len'], config['lstm_epochs'], config['lstm_batch'])
    if config['run_lasso_cv']:
        base_model_builders['lasso_cv'] = lambda ds: make_lasso_cv(config['random_state'])
    if config['run_pls_ols']:
        base_model_builders['pls_ols'] = lambda ds: make_pls_ols(config['pls_n_components'], ds['poly_cols'], ds['passthrough_cols'])
    if config['run_rf_pls']:
        base_model_builders['rf_pls'] = lambda ds: make_rf_pls(
            config['pls_n_components'],
            ds['poly_cols'],
            ds['passthrough_cols'],
            config['rf_n_estimators'],
            config['random_state'],
        )
    if config['run_lstm_pls'] and TF_AVAILABLE:
        base_model_builders['lstm_pls'] = lambda ds: LSTMPLSWrapper(
            config['pls_n_components'],
            ds['poly_cols'],
            ds['passthrough_cols'],
            config['lstm_seq_len'],
            config['lstm_epochs'],
            config['lstm_batch'],
        )

    if config['run_lstm'] and not TF_AVAILABLE:
        print('Warning: RUN_LSTM=True but TensorFlow is unavailable - skipping lstm registration.')
    if config['run_lstm_pls'] and not TF_AVAILABLE:
        print('Warning: RUN_LSTM_PLS=True but TensorFlow is unavailable - skipping lstm_pls registration.')

    registry = {}
    poly_cols = dataset_spec.get('poly_cols')
    has_poly_block = poly_cols is not None and len(poly_cols) > 0

    for model_name, builder in base_model_builders.items():
        if model_name in MODELS_REQUIRING_POLY_BLOCK:
            if config['fe_input_mode'] != 'raw_panel':
                print(
                    f"  Skip {model_name} for dataset '{dataset_tag}' "
                    f"(PLS variants are disabled in FE_INPUT_MODE={config['fe_input_mode']!r})."
                )
                continue
            if not has_poly_block:
                print(
                    f"  Skip {model_name} for dataset '{dataset_tag}' "
                    '(requires a polymarket block).'
                )
                continue
        registry[model_name] = lambda builder=builder, ds=dataset_spec: builder(ds)

    return registry
