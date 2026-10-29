"""Reportes al final de cada entrenamiento, con el mismo formato que FedMammoBench.

Port de `FedMammoBench/src/reporting.py` generalizado a N clases: mismos
nombres de archivo, mismas columnas y mismas gráficas, para que las corridas
de FLamby se analicen con las mismas herramientas que las de FedMammoBench.

FLamby no tiene conjunto de validación, así que el test de cada centro hace
de "val": las columnas `val_*` de `metrics.csv` se calculan sobre test (queda
anotado como `val_source: test` en `config.yaml`) y solo se genera `test/`.

Estructura de salida (igual que `runs/centralizado` y `runs/federado`):

    centralizado (federated=False):
        run_dir/{config.yaml, metrics.csv, plots/, test/}
    federado (federated=True):
        run_dir/server/{config.yaml, metrics.csv, best.json, plots/}
        run_dir/nodes/<centro>/{metrics.csv, rounds.csv, plots/, test/}
        run_dir/pooled_eval/test/

Con 2 clases las métricas son las binarias de FedMammoBench (clase positiva = 1).
Con más de 2 son el promedio macro one-vs-rest sobre las clases presentes, y
`confusion_matrix_metrics.json` agrega además el detalle por clase.

Uso:
    >>> reporter = RunReporter(
    ...     "runs/fed_isic2019/seed0/pooled",
    ...     BaselineLoss(),
    ...     "fed_isic2019",
    ...     train_dls,
    ...     test_dls,
    ... )
    >>> for epoch in range(num_epochs):
    ...     train_one_epoch(model)
    ...     reporter.log_epoch(model)
    >>> reporter.finish(model)
"""

import json
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

# En el test de un centro chico es normal que el modelo prediga clases que ahí
# no aparecen; sklearn lo avisaría en cada ronda y centro.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

# Nombres de centros (en el orden de `center=i` de FLamby) y de clases (en el
# orden de la etiqueta) de los datasets de clasificación. Los demás datasets
# (segmentación, supervivencia) no generan reportes.
DATASET_LABELS = {
    "fed_isic2019": {
        "centers": [
            "BCN",
            "HAM_vidir_molemax",
            "HAM_vidir_modern",
            "HAM_rosendahl",
            "MSK",
            "HAM_vienna_dias",
        ],
        "classes": ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    },
    "fed_isic2019_binary": {
        "centers": [
            "BCN",
            "HAM_vidir_molemax",
            "HAM_vidir_modern",
            "HAM_rosendahl",
            "MSK",
            "HAM_vienna_dias",
        ],
        "classes": ["Benigno", "Maligno"],
    },
    "fed_heart_disease": {
        "centers": ["cleveland", "hungarian", "switzerland", "va"],
        "classes": ["Sin enfermedad", "Con enfermedad"],
    },
}

# Mismos nombres a mostrar que `METRIC_DISPLAY_NAMES` de FedMammoBench.
METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "sensitivity": "Sensibilidad",
    "specificity": "Especificidad",
    "f1": "F1",
    "f1_macro": "F1-macro",
    "precision": "Precisión",
}

_METRIC_BAR_COLORS = {
    "accuracy": "#2a78d6",
    "auc": "#eb6834",
    "sensitivity": "#1baf7a",
    "specificity": "#eda100",
    "precision": "#e87ba4",
    "f1_macro": "#008300",
}


# --------------------------------------------------------------------------
# Predicciones y métricas
# --------------------------------------------------------------------------


def predict(model, dataloader, loss_fn, use_gpu=True):
    """Corre el modelo sobre un dataloader.

    Returns
    -------
    tuple
        `(y_true, probs, loss)`: etiquetas enteras `(N,)`, probabilidades
        `(N, K)` (con K = 2 también para modelos binarios de una salida) y la
        pérdida media.
    """
    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    y_true, probs, loss_sum = [], [], 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss_sum += loss_fn(out, y).item() * len(y)
            if out.ndim == 2 and out.shape[1] > 1:
                p = torch.softmax(out, dim=1)
            else:
                # Una sola salida: probabilidad de la clase 1. Si el modelo
                # devuelve logits (fuera de [0, 1]) se les aplica sigmoid.
                p1 = out.view(-1)
                if p1.min() < 0 or p1.max() > 1:
                    p1 = torch.sigmoid(p1)
                p = torch.stack([1 - p1, p1], dim=1)
            probs.append(p.cpu().numpy())
            y_true.append(y.view(-1).long().cpu().numpy())
    y_true = np.concatenate(y_true)
    return y_true, np.concatenate(probs), loss_sum / len(y_true)


def _safe_div(numerator, denominator):
    """División con 0.0 si el denominador es 0 (mismo criterio que FedMammoBench)."""
    return numerator / denominator if denominator != 0 else 0.0


def _binary_cm_metrics(tp, tn, fp, fn):
    """Las métricas `cm_*` de FedMammoBench a partir de los conteos de una clase."""
    total = tp + tn + fp + fn
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, total)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    f1_negative = _safe_div(2 * npv * specificity, npv + specificity)
    expected_agreement = _safe_div(
        (tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), total**2
    )
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "cm_tp": tp,
        "cm_tn": tn,
        "cm_fp": fp,
        "cm_fn": fn,
        "cm_accuracy": accuracy,
        "cm_balanced_accuracy": (sensitivity + specificity) / 2,
        "cm_sensitivity": sensitivity,
        "cm_specificity": specificity,
        "cm_precision": precision,
        "cm_npv": npv,
        "cm_fpr": _safe_div(fp, fp + tn),
        "cm_fnr": _safe_div(fn, fn + tp),
        "cm_fdr": _safe_div(fp, fp + tp),
        "cm_for": _safe_div(fn, fn + tn),
        "cm_f1": f1,
        "cm_f1_macro": (f1 + f1_negative) / 2,
        "cm_mcc": _safe_div(tp * tn - fp * fn, mcc_denominator),
        "cm_kappa": _safe_div(accuracy - expected_agreement, 1 - expected_agreement),
        "cm_youden_j": sensitivity + specificity - 1,
        "cm_markedness": precision + npv - 1,
        "cm_prevalence": _safe_div(tp + fn, total),
        "cm_positive_likelihood_ratio": _safe_div(sensitivity, 1 - specificity),
        "cm_negative_likelihood_ratio": _safe_div(1 - sensitivity, specificity),
        "cm_diagnostic_odds_ratio": _safe_div(tp * tn, fp * fn),
        "cm_g_mean": (sensitivity * specificity) ** 0.5,
        "cm_fowlkes_mallows": (precision * sensitivity) ** 0.5,
        "cm_threat_score": _safe_div(tp, tp + fn + fp),
    }


def _per_class_cm_metrics(y_true, y_pred, num_classes):
    """`_binary_cm_metrics` de cada clase contra el resto (one-vs-rest)."""
    cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    per_class = []
    for k in range(num_classes):
        tp = float(cm[k, k])
        fn = float(cm[k, :].sum() - tp)
        fp = float(cm[:, k].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        per_class.append(_binary_cm_metrics(tp, tn, fp, fn))
    return per_class


def compute_metrics(y_true, probs, loss):
    """Las 7 métricas de FedMammoBench (más `loss`) para predicciones completas.

    Binario: exactamente las de FedMammoBench (clase positiva = 1, umbral 0.5).
    Multiclase: promedio macro one-vs-rest sobre las clases presentes en
    `y_true`; `f1` es el F1 ponderado por soporte y se agrega
    `balanced_accuracy`, la métrica oficial de Fed-ISIC2019.
    """
    num_classes = probs.shape[1]
    y_pred = probs.argmax(axis=1)
    per_class = _per_class_cm_metrics(y_true, y_pred, num_classes)
    result = {"loss": loss, "accuracy": float((y_true == y_pred).mean())}

    if num_classes == 2:
        positive = per_class[1]
        result.update(
            auc=skm.roc_auc_score(y_true, probs[:, 1]) if len(set(y_true)) == 2 else 0.0,
            sensitivity=positive["cm_sensitivity"],
            specificity=positive["cm_specificity"],
            f1=positive["cm_f1"],
            f1_macro=positive["cm_f1_macro"],
            precision=positive["cm_precision"],
        )
        return result

    present = sorted(set(y_true))
    # AUC solo de las clases con positivos y negativos (en un centro chico
    # puede faltar alguna clase y ahí la AUC no está definida).
    aucs = [
        skm.roc_auc_score(y_true == k, probs[:, k])
        for k in present
        if 0 < (y_true == k).sum() < len(y_true)
    ]
    result.update(
        auc=float(np.mean(aucs)) if aucs else 0.0,
        sensitivity=float(np.mean([per_class[k]["cm_sensitivity"] for k in present])),
        specificity=float(np.mean([per_class[k]["cm_specificity"] for k in present])),
        f1=skm.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_macro=float(np.mean([per_class[k]["cm_f1"] for k in present])),
        precision=float(np.mean([per_class[k]["cm_precision"] for k in present])),
        balanced_accuracy=skm.balanced_accuracy_score(y_true, y_pred),
    )
    return result


def compute_confusion_matrix_metrics(y_true, y_pred, class_names):
    """Las métricas `cm_*` de FedMammoBench.

    Binario: las de la clase positiva, idénticas a FedMammoBench. Multiclase:
    promedio macro de cada `cm_*` sobre las clases presentes (con accuracy,
    kappa y MCC multiclase reales en vez del promedio) y, en `per_class`, las
    de cada clase contra el resto.
    """
    per_class = _per_class_cm_metrics(y_true, y_pred, len(class_names))
    if len(class_names) == 2:
        return per_class[1]

    present = sorted(set(y_true))
    macro = {
        key: float(np.mean([per_class[k][key] for k in present]))
        for key in per_class[0]
        if key not in ("cm_tp", "cm_tn", "cm_fp", "cm_fn")
    }
    macro["cm_accuracy"] = float((y_true == y_pred).mean())
    macro["cm_balanced_accuracy"] = skm.balanced_accuracy_score(y_true, y_pred)
    macro["cm_kappa"] = skm.cohen_kappa_score(y_true, y_pred)
    macro["cm_mcc"] = skm.matthews_corrcoef(y_true, y_pred)
    macro["per_class"] = dict(zip(class_names, per_class))
    return macro


# --------------------------------------------------------------------------
# Archivos y gráficas (mismo formato que FedMammoBench)
# --------------------------------------------------------------------------


def _new_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics_json(metrics, path):
    _new_file(path).write_text(json.dumps(metrics, indent=2, sort_keys=True))


def save_predictions_csv(y_true, probs, path):
    """Binario: `y_true,y_pred,y_prob` (como FedMammoBench). Multiclase: una
    columna `y_prob_<k>` por clase."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": probs.argmax(axis=1)})
    if probs.shape[1] == 2:
        df["y_prob"] = probs[:, 1]
    else:
        for k in range(probs.shape[1]):
            df[f"y_prob_{k}"] = probs[:, k]
    df.to_csv(_new_file(path), index=False)


def _draw_confusion_matrix(ax, y_true, y_pred, class_names):
    cm = skm.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    ticks = list(range(len(class_names)))
    ax.set_xticks(ticks, labels=class_names)
    ax.set_yticks(ticks, labels=class_names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    return im


def _panel_size(class_names):
    """5x5 por panel como FedMammoBench, más grande si hay muchas clases."""
    return max(5.0, 0.7 * len(class_names) + 1)


def plot_confusion_matrix(y_true, y_pred, path, class_names):
    side = _panel_size(class_names)
    fig, ax = plt.subplots(figsize=(side + 1, side))
    im = _draw_confusion_matrix(ax, y_true, y_pred, class_names)
    ax.set_title("Matriz de confusión")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(_new_file(path), dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true, probs, path, class_names):
    """Binario: la curva de la clase positiva. Multiclase: una curva
    one-vs-rest por clase presente."""
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = [1] if len(class_names) == 2 else range(len(class_names))
    for k in classes:
        positives = y_true == k
        if not 0 < positives.sum() < len(y_true):
            continue  # clase ausente: la curva no está definida
        fpr, tpr, _ = skm.roc_curve(positives, probs[:, k])
        name = "ROC" if len(class_names) == 2 else class_names[k]
        ax.plot(fpr, tpr, label=f"{name} (AUC = {skm.auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Aleatorio")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right", fontsize=8 if len(class_names) > 2 else None)
    fig.tight_layout()
    fig.savefig(_new_file(path), dpi=200)
    plt.close(fig)


def plot_metric_curve(
    train_values, val_values, path, metric_name, best_epoch=None, x_label="Época"
):
    """Curva train vs. val (= test). `best_epoch` en base 0, como en FedMammoBench.

    La loss tiene el eje Y desde 0 hasta al menos 1; las demás métricas, fijo en [0, 1].
    """
    steps = range(1, len(train_values) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_values, label=f"Train {metric_name}", color="#1f77b4")
    ax.plot(steps, val_values, label=f"Val (test) {metric_name}", color="#d62728")
    if best_epoch is not None:
        label = f"Mejor {x_label.lower()} ({best_epoch + 1})"
        ax.axvline(best_epoch + 1, linestyle="--", color="gray", lw=1, label=label)
    top = max(1.0, max(train_values), max(val_values)) if metric_name == "Loss" else 1
    ax.set_ylim(0, top)
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} — entrenamiento vs. validación (test)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_new_file(path), dpi=200)
    plt.close(fig)


def plot_curves(history, plots_dir, best_epoch=None, x_label="Época"):
    """Las 8 `*_curve.png` de FedMammoBench a partir de las filas de `metrics.csv`."""
    df = pd.DataFrame(history)
    names = {"loss": "Loss", **METRIC_DISPLAY_NAMES}
    for key, name in names.items():
        path = Path(plots_dir) / f"{key}_curve.png"
        plot_metric_curve(
            df[f"train_{key}"], df[f"val_{key}"], path, name, best_epoch, x_label
        )


def plot_confusion_matrix_by_database(cm_by_db, path, class_names):
    """Un panel por base de datos (centro), en filas de hasta 4 paneles."""
    side = _panel_size(class_names)
    ncols = min(4, len(cm_by_db))
    nrows = -(-len(cm_by_db) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(side * ncols, side * nrows), squeeze=False
    )
    n_panels = len(cm_by_db)
    for ax in axes.flat[n_panels:]:
        ax.axis("off")
    for ax, (db_name, (y_true, y_pred)) in zip(axes.flat, cm_by_db.items()):
        _draw_confusion_matrix(ax, y_true, y_pred, class_names)
        ax.set_title(db_name)
    fig.suptitle("Matriz de confusión por base de datos")
    fig.tight_layout()
    fig.savefig(_new_file(path), dpi=200)
    plt.close(fig)


def plot_metrics_by_database(metrics_by_db, path):
    metric_keys = list(_METRIC_BAR_COLORS)
    db_names = list(metrics_by_db)
    bar_width = 0.8 / len(metric_keys)
    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(db_names)), 5))
    for i, key in enumerate(metric_keys):
        offsets = [
            g + (i - (len(metric_keys) - 1) / 2) * bar_width
            for g in range(len(db_names))
        ]
        values = [metrics_by_db[db].get(key, 0.0) for db in db_names]
        bars = ax.bar(
            offsets,
            values,
            width=bar_width * 0.9,
            label=METRIC_DISPLAY_NAMES[key],
            color=_METRIC_BAR_COLORS[key],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(len(db_names)), labels=db_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Métricas de test por base de datos")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(metric_keys),
        fontsize=8,
        frameon=False,
    )
    fig.savefig(_new_file(path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_split(y_true, probs, loss, split_dir, class_names):
    """Escribe una carpeta `test/` como la de FedMammoBench y devuelve sus métricas."""
    split_dir = Path(split_dir)
    y_pred = probs.argmax(axis=1)
    split_metrics = compute_metrics(y_true, probs, loss)
    cm_metrics = compute_confusion_matrix_metrics(y_true, y_pred, class_names)
    save_metrics_json(split_metrics, split_dir / "metrics.json")
    save_metrics_json(cm_metrics, split_dir / "confusion_matrix_metrics.json")
    save_predictions_csv(y_true, probs, split_dir / "predictions.csv")
    plot_confusion_matrix(
        y_true, y_pred, split_dir / "confusion_matrix.png", class_names
    )
    plot_roc_curve(y_true, probs, split_dir / "roc_curve.png", class_names)
    return {**split_metrics, **cm_metrics}


# --------------------------------------------------------------------------
# Reporter: lo único que llaman los scripts de entrenamiento
# --------------------------------------------------------------------------


def _wandb_mode():
    """`WANDB_MODE` si el usuario lo fijó; si no, online solo con credenciales
    (variable `WANDB_API_KEY` o `wandb login` en ~/.netrc) y offline en otro
    caso, para no bloquear nunca un entrenamiento por falta de red o login."""
    if os.environ.get("WANDB_MODE"):
        return os.environ["WANDB_MODE"]
    netrc = Path.home() / ".netrc"
    logged_in = netrc.is_file() and "api.wandb.ai" in netrc.read_text()
    return "online" if os.environ.get("WANDB_API_KEY") or logged_in else "offline"


class RunReporter:
    """Registra cada época/ronda y escribe todos los reportes al terminar.

    Parameters
    ----------
    run_dir : str or Path
        Carpeta de la corrida (ej. `runs/fed_isic2019/seed0/FedAvg100`).
    loss_fn : callable
        La misma pérdida del entrenamiento, para calcular `train_loss`/`val_loss`.
    dataset_name : str
        Una de las claves de `DATASET_LABELS`.
    train_dls : list[torch.utils.data.DataLoader]
        Los dataloaders con los que se entrena (uno solo en pooled).
    test_dls : list[torch.utils.data.DataLoader]
        El test de cada centro, en el orden de `center=i` (None = centro vacío).
    federated : bool
        `False`: estructura de `runs/centralizado`. `True`: estructura de
        `runs/federado` (`server/`, `nodes/`, `pooled_eval/`).
    config : dict, optional
        Hiperparámetros a guardar en `config.yaml`.
    best_metric : str
        Métrica de val (= test) que elige la mejor época/ronda. Es solo
        informativa: la evaluación final usa siempre el modelo final, porque
        elegir el modelo con test inflaría los resultados.
    use_gpu : bool
    wandb_project : str, optional
        Si se da, la corrida también se registra en Weights & Biases (curvas
        por época/ronda, métricas de test en el summary y todos los PNG). Sin
        credenciales (`WANDB_API_KEY` o `wandb login`) queda en modo offline,
        para subirla después con `wandb sync`. None: `wandb` ni se importa.
    wandb_name, wandb_group, wandb_tags : optional
        Nombre, grupo y etiquetas de la corrida en W&B.
    """

    def __init__(
        self,
        run_dir,
        loss_fn,
        dataset_name,
        train_dls,
        test_dls,
        federated=False,
        config=None,
        best_metric="auc",
        use_gpu=True,
        wandb_project=None,
        wandb_name=None,
        wandb_group=None,
        wandb_tags=None,
    ):
        self.run_dir = Path(run_dir)
        self.loss_fn = loss_fn
        self.train_dls = train_dls
        self.test_dls = test_dls
        self.centers = DATASET_LABELS[dataset_name]["centers"]
        self.class_names = DATASET_LABELS[dataset_name]["classes"]
        self.federated = federated
        self.best_metric = best_metric
        self.use_gpu = use_gpu
        self.main_dir = self.run_dir / "server" if federated else self.run_dir
        self.history = []
        self.node_history = {center: [] for center in self.centers}
        self._last_time = time.time()

        config = {**(config or {}), "dataset": dataset_name, "val_source": "test"}
        # Se omiten los hiperparámetros que no aplican (NaN) y las clases
        # (ej. torch.optim.SGD) se guardan por su nombre.
        config = {
            k: f"{v.__module__}.{v.__qualname__}" if isinstance(v, type) else v
            for k, v in config.items()
            if not (isinstance(v, float) and np.isnan(v))
        }
        config = {
            k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
            for k, v in config.items()
        }
        path = _new_file(self.main_dir / "config.yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

        self._wandb = None
        if wandb_project:
            import wandb  # import perezoso: quien no usa W&B no lo necesita

            self._wandb = wandb.init(
                project=wandb_project,
                name=wandb_name,
                group=wandb_group,
                tags=wandb_tags,
                config=config,
                dir=str(self.run_dir),
                mode=_wandb_mode(),
            )

    def _predict_all(self, model, dataloaders):
        """`predict` sobre cada dataloader (None = centro vacío) sin alterar
        el entrenamiento: restaura el modo del modelo y las semillas, para que
        la corrida sea idéntica a una sin reportes."""
        rng_states = (
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            np.random.get_state(),
            random.getstate(),
        )
        was_training = model.training
        model.eval()
        results = [
            predict(model, dl, self.loss_fn, self.use_gpu) if dl is not None else None
            for dl in dataloaders
        ]
        model.train(was_training)
        torch.set_rng_state(rng_states[0])
        if rng_states[1] is not None:
            torch.cuda.set_rng_state_all(rng_states[1])
        np.random.set_state(rng_states[2])
        random.setstate(rng_states[3])
        return results

    @staticmethod
    def _pool(results):
        """Junta las predicciones de varios centros como si fueran un solo split."""
        results = [r for r in results if r is not None]
        y_true = np.concatenate([r[0] for r in results])
        probs = np.concatenate([r[1] for r in results])
        loss = sum(r[2] * len(r[0]) for r in results) / len(y_true)
        return y_true, probs, loss

    def log_epoch(self, model):
        """Evalúa el modelo actual en train y test y agrega una fila a `metrics.csv`."""
        seconds = time.time() - self._last_time
        train_results = self._predict_all(model, self.train_dls)
        test_results = self._predict_all(model, self.test_dls)
        train_metrics = compute_metrics(*self._pool(train_results))
        val_metrics = compute_metrics(*self._pool(test_results))

        step = len(self.history) + 1
        row = {"epoch": step if self.federated else step - 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row["fit_seconds" if self.federated else "duration_seconds"] = seconds
        self.history.append(row)
        pd.DataFrame(self.history).to_csv(
            _new_file(self.main_dir / "metrics.csv"), index=False
        )

        if self.federated:
            self._log_nodes(step, train_results, test_results)
        if self._wandb is not None:
            payload = {k: v for k, v in row.items() if k != "epoch"}
            for center, rows in self.node_history.items():
                if rows and rows[-1]["round"] == step:
                    payload.update(
                        {
                            f"nodes/{center}/{k}": v
                            for k, v in rows[-1].items()
                            if k.startswith("val_")
                        }
                    )
            self._wandb.log(payload, step=step)
        best_key = f"val_{self.best_metric}"
        print(
            f"[reporter] {self.main_dir.name} paso {step}: "
            f"val_loss={row['val_loss']:.4f} {best_key}={row[best_key]:.4f}"
        )
        self._last_time = time.time()

    def _log_nodes(self, step, train_results, test_results):
        """Por centro: `metrics.csv` (train y test del modelo global) y
        `rounds.csv` (solo test, como el `rounds.csv` de FedMammoBench)."""
        for center, train_r, test_r in zip(self.centers, train_results, test_results):
            if train_r is None or test_r is None:
                continue
            node_dir = self.run_dir / "nodes" / center
            val = compute_metrics(*test_r)
            row = {"epoch": step, "round": step}
            row.update({f"train_{k}": v for k, v in compute_metrics(*train_r).items()})
            row.update({f"val_{k}": v for k, v in val.items()})
            self.node_history[center].append(row)
            pd.DataFrame(self.node_history[center]).to_csv(
                _new_file(node_dir / "metrics.csv"), index=False
            )
            rounds = [
                {
                    "round": r["round"],
                    "num_examples": len(test_r[0]),
                    **{k[4:]: v for k, v in r.items() if k.startswith("val_")},
                }
                for r in self.node_history[center]
            ]
            pd.DataFrame(rounds).to_csv(node_dir / "rounds.csv", index=False)

    def finish(self, model):
        """Escribe las curvas, `best.json` (federado) y la evaluación final de test."""
        x_label = "Ronda" if self.federated else "Época"
        best_idx = None
        if self.history:
            values = [row[f"val_{self.best_metric}"] for row in self.history]
            best_idx = int(np.argmax(values))
            plot_curves(self.history, self.main_dir / "plots", best_idx, x_label)
        if self.federated:
            save_metrics_json(
                {
                    "best_round": best_idx + 1 if best_idx is not None else None,
                    "metric_name": self.best_metric,
                    "metric_value": self.history[best_idx][f"val_{self.best_metric}"]
                    if best_idx is not None
                    else None,
                    "evaluated_model": "final",
                    "note": "val = test: la mejor ronda es solo informativa, "
                    "test/ evalúa el modelo de la última ronda.",
                },
                self.main_dir / "best.json",
            )
            for center, rows in self.node_history.items():
                if rows:
                    plot_curves(
                        rows, self.run_dir / "nodes" / center / "plots", None, x_label
                    )

        test_results = self._predict_all(model, self.test_dls)
        test_dir = self.run_dir / ("pooled_eval/test" if self.federated else "test")
        test_metrics = evaluate_split(
            *self._pool(test_results), test_dir, self.class_names
        )

        # Desglose por centro, con los mismos nombres de archivo que FedMammoBench.
        metrics_by_db, cm_by_db = {}, {}
        for center, result in zip(self.centers, test_results):
            if result is None:
                continue
            y_true, probs, loss = result
            if self.federated:
                metrics_by_db[center] = evaluate_split(
                    y_true,
                    probs,
                    loss,
                    self.run_dir / "nodes" / center / "test",
                    self.class_names,
                )
            else:
                y_pred = probs.argmax(axis=1)
                metrics_by_db[center] = {
                    **compute_metrics(y_true, probs, loss),
                    **compute_confusion_matrix_metrics(y_true, y_pred, self.class_names),
                }
            cm_by_db[center] = (y_true, probs.argmax(axis=1))
            save_predictions_csv(y_true, probs, test_dir / f"predictions_{center}.csv")
        save_metrics_json(metrics_by_db, test_dir / "metrics_by_database.json")
        plot_confusion_matrix_by_database(
            cm_by_db, test_dir / "confusion_matrix_by_database.png", self.class_names
        )
        plot_metrics_by_database(metrics_by_db, test_dir / "metrics_by_database.png")
        if self._wandb is not None:
            self._log_wandb_final(test_metrics, metrics_by_db, test_dir)
        print(f"[reporter] reportes escritos en {self.run_dir}")

    def _log_wandb_final(self, test_metrics, metrics_by_db, test_dir):
        """Sube a W&B lo mismo que queda en disco: métricas de test (summary),
        todos los PNG y las predicciones; luego cierra la corrida."""
        import wandb

        summary = {f"test/{k}": v for k, v in test_metrics.items()}
        for center, metrics in metrics_by_db.items():
            summary.update(
                {f"test_by_center/{center}/{k}": v for k, v in metrics.items()}
            )
        self._wandb.summary.update(
            {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        )
        files = {
            str(png.relative_to(self.run_dir).with_suffix("")): wandb.Image(str(png))
            for png in sorted(self.run_dir.rglob("*.png"))
            if "wandb" not in png.parts
        }
        files["test/predictions"] = wandb.Table(
            dataframe=pd.read_csv(test_dir / "predictions.csv")
        )
        self._wandb.log(files, step=len(self.history) or None)
        self._wandb.finish()
