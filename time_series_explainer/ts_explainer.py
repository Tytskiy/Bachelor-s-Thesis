from typing import Iterable, List, Optional, Union, Tuple, Any, Callable
import numpy as np
from sklearn.metrics import pairwise_distances
from lime import lime_base, explanation
from functools import partial
from math import comb
from random import randint, sample
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import Ridge


def indexes_split(n: int, num_slices: int):
    indexes = []
    start = 0
    end = n//num_slices
    for i in range(n % num_slices):
        indexes.append((start, end + 1))
        start = end + 1
        end = start + n//num_slices
    for i in range(num_slices - n % num_slices):
        indexes.append((start, end))
        start = end
        end = start + n//num_slices
        if end > n:
            end = n
    return indexes


def generate_sythetic_data(
    ts_instance: np.ndarray,
    predict_fn: Callable,
    num_slices: int,
    num_samples: Union[int, None],
    metric: str = "jaccard",
    replacement_method: str = "random",
    data: Optional[np.ndarray] = None,
    gen_with_replacement=False,
) -> Tuple:
    if ts_instance.shape[1] <= num_slices:
        raise ValueError("num_slices must be less or equal len of time series")

    if num_slices >= 63:
        warnings.warn("Generated with gen_with_replacement = True because num_slices >= 63")

    if num_samples is None or num_samples >= 2**num_slices:
        num_samples = 2**num_slices

    if num_slices < 63 and not gen_with_replacement:
        # random.sample support range from 0 to 2**63 - 1
        _rng = np.array([2**num_slices - 1, 0] + sample(range(1, 2**num_slices-1), num_samples-2))
        bin_samples = (((_rng[:, None] & (1 << np.arange(num_slices-1, -1, -1)))) > 0).astype(int)
    else:
        bin_samples = np.random.binomial(1, p=0.5, size=(num_samples, num_slices))
        bin_samples[0] = 1  # first vector must match with source object
        bin_samples[1] = 0

    idxs_split = indexes_split(ts_instance.shape[1], num_slices)

    samples = []
    for i in range(num_samples):
        expanded_bin_samples = np.empty_like(ts_instance)
        for j, (s, e) in enumerate(idxs_split):
            expanded_bin_samples[:, s:e] = bin_samples[i, j]

        disabled_ts = np.empty_like(ts_instance)
        for j, (s, e) in enumerate(idxs_split):
            if bin_samples[i, j] == 1:
                continue
            if replacement_method == "dataset_mean" and data is not None:
                disabled_ts[:, s:e] = data[:, :, s:e].mean(axis=(0, 2))[:, None]
            elif replacement_method == "zeros":
                disabled_ts[:, s:e] = 0
            elif replacement_method == "normal_random":
                disabled_ts[:, s:e] = np.random.normal(
                    loc=ts_instance.mean(axis=1)[:, None],
                    scale=ts_instance.scale(axis=1)[:, None],
                    size=ts_instance[:, s:e].shape)
            elif replacement_method == "random":
                disabled_ts[:, s:e] = np.random.uniform(
                    low=ts_instance.min(axis=1)[:, None],
                    high=ts_instance.max(axis=1)[:, None],
                    size=ts_instance[:, s:e].shape)
            elif replacement_method == "dataset" and data is not None:
                k = randint(0, len(data)-1)
                disabled_ts[:, s:e] = data[k][:, s:e]
            else:
                raise ValueError("Incorrect replacement_method (and maybe data is None)")

        new_ts = np.where(expanded_bin_samples, ts_instance, disabled_ts)
        samples.append(new_ts[None, :, :])

    samples = np.concatenate(samples, axis=0)
    targets = predict_fn(samples)
    fi_0 = targets[1].copy()  # its need for shap
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)
        distances = pairwise_distances(bin_samples[0][None, :], bin_samples, metric=metric).ravel()
    return bin_samples, targets, distances, fi_0


class LimeTimeSeriesExplainer:

    def __init__(
        self,
        predict_fn: Callable,
        num_features: Optional[int] = 1,
        class_names: Optional[List[str]] = None,
        metric: str = "jaccard",
        replacement_method: str = "random",
        model_regressor=None,
        kernel: Optional[Callable] = None,
        kernel_width: Optional[float] = None,
        feature_selection='auto'
    ):
        self.predict_fn = predict_fn
        self.metric = metric
        self.replacement_method = replacement_method
        self.class_names = class_names
        self.model_regressor = model_regressor
        self.feature_selection = feature_selection

        if kernel_width is None:
            kernel_width = np.sqrt(num_features) * 0.75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.base = lime_base.LimeBase(kernel_fn)

    def explain_instance(
        self,
        ts_instance: np.array,
        num_slices: int,
        num_samples: int,
        num_features=10,
        labels: Optional[List[Any]] = None,
        top_labels: Optional[int] = None
    ):
        X, preds, distances, _ = generate_sythetic_data(
            ts_instance, self.predict_fn, num_slices, num_samples, self.metric, self.replacement_method)

        if self.class_names is None:
            self.class_names = [str(x) for x in range(preds.shape[1])]

        ret_exp = explanation.Explanation(domain_mapper=explanation.DomainMapper(),
                                          class_names=self.class_names)
        ret_exp.predict_proba = preds[0]

        if labels is None and top_labels is None:
            top_labels = 1

        if top_labels is not None:
            labels = np.argsort(preds[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()

        for l in labels:
            (ret_exp.intercept[l],
             ret_exp.local_exp[l],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                neighborhood_data=X,
                neighborhood_labels=preds,
                distances=distances,
                label=l,
                num_features=num_features,
                model_regressor=self.model_regressor,
                feature_selection=self.feature_selection,
            )
        return ret_exp


class ShapTimeSeriesExplainer:
    def __init__(
        self,
        predict_fn,
        data: np.ndarray,
        class_names: Optional[List[str]] = None,
        replacement_method="mean"
    ):
        self.predict_fn = predict_fn
        self.data = data

        def metric(x, y):
            M = int(x.sum())
            N = int(y.sum())
            if N == 0 or N == M:
                return 10**6
            return (M - 1)/(comb(M, N) * (M - N)*N)

        self.metric = metric
        self.replacement_method = replacement_method
        self.class_names = class_names
        self.model_regressor = Ridge(alpha=0, fit_intercept=False)
        self.base = lime_base.LimeBase(lambda x: x)

    def explain_instance(
        self,
        ts_instance: np.ndarray,
        num_slices: int,
        num_samples: int,
        num_features: int,
        labels: Optional[List[Any]] = None,
        top_labels: Optional[int] = None
    ):
        X, preds, distances, fi_0 = generate_sythetic_data(
            ts_instance, self.predict_fn, num_slices, num_samples, self.metric, self.replacement_method, self.data)
        preds -= fi_0  # its reason why we don't fit intercept
        if self.class_names is None:
            self.class_names = [str(x) for x in range(preds.shape[1])]

        ret_exp = explanation.Explanation(domain_mapper=explanation.DomainMapper(),
                                          class_names=self.class_names)
        ret_exp.predict_proba = preds[0]

        if labels is None and top_labels is None:
            top_labels = 1

        if top_labels is not None:
            labels = np.argsort(preds[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()

        for l in labels:
            (ret_exp.intercept[l],
             ret_exp.local_exp[l],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                neighborhood_data=X,
                neighborhood_labels=preds,
                distances=distances,
                label=l,
                num_features=num_features,
                model_regressor=self.model_regressor,
                feature_selection="highest_weights",

            )
            ret_exp.intercept[l] = fi_0[l]
        return ret_exp


def plot_eeg(
    X: np.ndarray,
    num_slices: int,
    exp_like_alpha: List[Tuple[int, float]],
    features_name: Optional[List[str]] = None,
    title: str = "",
    figsize: Tuple[int, int] = (10, 7),
    positive_color='#2ca02c',
    negative_color='#d62728'
):
    if features_name is None:
        features_name = [f"feature#{i}" for i in range(X.shape[0])]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(X.shape[0], hspace=0)
    axes = gs.subplots(sharex=True, sharey=False)
    if not isinstance(axes, list):
        axes = [axes]

    fig.suptitle(title)
    for i, ax in enumerate(axes):
        ax.plot(np.arange(X.shape[1]), X[i, :])
        l, u = ax.get_ylim()
        ax.set_ylim(l - 0.3*abs(l), u + 0.3*abs(u))
        l, u = ax.get_ylim()

        # Set number of ticks for x-axis
        ax.set_yticks([X[i, :].mean()])
        ax.set_yticklabels([features_name[i]])

        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axes[0].spines['top'].set_visible(True)
    axes[-1].spines['bottom'].set_visible(True)

    idxs_split = indexes_split(X.shape[1], num_slices)
    for feature, weight in exp_like_alpha:
        start, end = idxs_split[feature]
        color = negative_color if weight < 0 else positive_color
        for ax in axes:
            ax.axvspan(start, end, color=color, alpha=abs(weight))

    return fig, axes


def cumulative_explanation(
        X: np.ndarray,
        label: int,
        explainer_functor: Callable,
        slices_range: Iterable[int] = np.arange(2, 21)
):
    cumulative_exp = np.zeros(X.shape[1])
    slices_range = np.asarray(slices_range)
    for num_slices in slices_range:
        exp = explainer_functor(int(num_slices))

        explanations = exp.as_list(label=label)
        values_per_slice = X.shape[1] / num_slices

        for feature, weight in explanations:
            start = int(feature * values_per_slice)
            end = int(start + values_per_slice)
            cumulative_exp[start:end] += weight

    list_explanations = []
    max_num_slices = slices_range.max()
    idxs_split = indexes_split(X.shape[1], max_num_slices)
    for i in range(max_num_slices):
        start, end = idxs_split[i]
        list_explanations.append((i, cumulative_exp[start:end].mean()))
    return list_explanations, cumulative_exp


def evaluate_explanation(
    ts_instance: np.ndarray,
    predict_fn: Callable,
    explanations: List[Tuple[int, float]],
    num_slices: int,
    replacement_method: str = "random",
    quantile: float = 0.9,
    data: Optional[np.ndarray] = None
):
    Nth_quantile = np.quantile(np.array([x[1] for x in explanations]), quantile)
    pos_explanations = set([x[0] for x in explanations if x[1] >= Nth_quantile])
    idxs_split = indexes_split(ts_instance.shape[1], num_slices)
    new_instance = ts_instance.copy()
    for j, (s, e) in enumerate(idxs_split):
        if j not in pos_explanations:
            continue
        if replacement_method == "dataset_mean" and data is not None:
            new_instance[:, s:e] = data[:, :, s:e].mean(axis=(0, 2))[:, None]
        elif replacement_method == "zeros":
            new_instance[:, s:e] = 0
        elif replacement_method == "normal_random":
            new_instance[:, s:e] = np.random.normal(
                loc=ts_instance.mean(axis=1)[:, None],
                scale=ts_instance.scale(axis=1)[:, None],
                size=ts_instance[:, s:e].shape)
        elif replacement_method == "random":
            new_instance[:, s:e] = np.random.uniform(
                low=ts_instance.min(axis=1)[:, None],
                high=ts_instance.max(axis=1)[:, None],
                size=ts_instance[:, s:e].shape)
        elif replacement_method == "swap":
            new_instance[:, s:e] = 1 - new_instance[:, s:e]
        elif replacement_method == "reverse":
            new_instance[:, s:e] = new_instance[:, e-1:s-1:-1]
        else:
            raise ValueError("Incorrect replacement_method (and maybe data is None)")
    diff = predict_fn(ts_instance) - predict_fn(new_instance)
    return diff
