import numpy as np
import matplotlib.pyplot as plt

def run_grouped_perm_importance(model, X, y, feature_index, n_repeats = 30, seed = 42):

    base_pred = model.predict(X)
    base_loss = np.mean((y - base_pred) ** 2)

    group_bases = ["mnth", "weekday", "season", "weathersit"]

    groups = {}

    for base in group_bases:

        if base in feature_index:
            groups[base] = [feature_index[base]]

        else:
            pref = base + "_"
            cols = []
            for name in feature_index:
                if name.startswith(pref):
                    cols.append(feature_index[name])
            
            if len(cols) > 0:
                cols.sort()
                groups[base] = cols

    for name in feature_index:
        
        idx = feature_index[name]

        already_covered = False
        for gname in groups:
            if idx in groups[gname]:
                already_covered = True
                break

        if not already_covered:
            groups[name] = [idx]

    rng = np.random.default_rng(seed)
    n = X.shape[0]

    results = []

    for gname in groups:
        idxs = groups[gname]
        drops = []

        for r in range(n_repeats):
            perm = rng.permutation(n)
            Xp = X.copy()

            Xp[:, idxs] = X[perm][:, idxs]

            if len(idxs) == 1:
                j = idxs[0]
                if np.all(Xp[:, j] == X[:, j]):
                    print("WARNING: permutation did not change", gname)

            pred_p = model.predict(Xp).reshape(-1)
            loss_p = np.mean((y - pred_p) ** 2)
            drops.append(loss_p - base_loss)

        drops = np.array(drops)
        imp = float(drops.mean())
        sd = float(drops.std(ddof = 1)) if n_repeats > 1 else 0.0

        results.append({"feature": gname, "importance": imp, "std": sd})

    results.sort(key = lambda d: d["importance"], reverse=True)

    return results

def print_table(results, top_k = 10):

    print("Feature".ljust(18), "Imp(Δloss)".rjust(12), "Std".rjust(10))
    print("-" * 52)

    k = min(top_k, len(results))
    for i in range(k):

        r = results[i]
        print(
            str(r["feature"]).ljust(18),
            ("%.6f" % r["importance"]).rjust(12),
            ("%.6f" % r["std"]).rjust(10),
        )

def plot_importance(results, top_k = 10):
    
    k = min(top_k, len(results))
    top = results[:k][::-1]

    labels = []
    vals = []
    for r in top:
        labels.append(r["feature"])
        vals.append(r["importance"])

    plt.figure(figsize=(9, max(4, 0.4 * k)))
    plt.xlabel("Permutation importance (Δ loss)")
    plt.barh(labels, vals)
    plt.tight_layout()
    plt.show()