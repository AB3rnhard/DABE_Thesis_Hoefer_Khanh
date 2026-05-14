import pandas as pd


def build_summary_table(all_results, RETURN_HORIZON_MIN, RIDGE_ALPHA, PCA_N_COMPONENTS,
                        panel_date, RESULTS_DIR):
    """Build a model-performance summary DataFrame, print it, and save a CSV snapshot.

    Returns
    -------
    pd.DataFrame
        The summary table sorted by dataset / model / window.
    """
    rows = []
    for r in all_results:
        m   = r.get("metrics", r)
        cfg = r.get("config", {}) if isinstance(r, dict) else {}
        rows.append({
            "dataset"       : r.get("dataset_tag", "poly"),
            "model"         : r.get("model"),
            "window"        : r.get("window"),
            "date"          : r.get("data_date"),
            "horizon_min"   : cfg.get("return_horizon_min", RETURN_HORIZON_MIN),
            "ridge_alpha"   : cfg.get("ridge_alpha", RIDGE_ALPHA),
            "pca_applied"   : cfg.get("pca_applied", False),
            "pca_n"         : cfg.get("pca_n") if cfg.get("pca_applied") else None,
            "n_features"    : r.get("n_features"),
            "rmse"          : m.get("rmse"),
            "mae"           : m.get("mae"),
            "r2"            : m.get("r2"),
            "dir_acc"       : m.get("dir_acc"),
            "n"             : m.get("n"),
            "train_time_s"  : r.get("train_time_s"),
        })
    summary = pd.DataFrame(rows).sort_values(["dataset", "model", "window"])
    print(summary.to_string(index=False))

    _snap_name = (f"summary_h{RETURN_HORIZON_MIN}m_a{str(RIDGE_ALPHA).replace('.', 'p')}"
                  f"_p{PCA_N_COMPONENTS or 0}_{panel_date}.csv")
    summary.to_csv(RESULTS_DIR / _snap_name, index=False)

    return summary
