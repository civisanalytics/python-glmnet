import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.linewidth"] = 3
mpl.rcParams["lines.linewidth"] = 2


def coeff_path_plot(
    est,
    feature_names=None,
    figsize=(10, 6),
    linestyle="-",
    fontsize=18,
    grid=True,
    legend=True,
    legendloc="center",
    xlabel=None,
    ylabel=None,
    title=None,
    bbox_to_anchor=None,
    yscale=None,
    save_path=None,
):
    """Plot coefficient's paths vs -Log(lambda).

    Parameters
    ----------
    est : estimator
        The previously fitted estimator.

    feature_names : list, shape (n_features,)
        Input features names neede for legend.

    figsize : tuple or list, as (width, height)
        Figure size.

    linestyle: string
        Linestyle of coefficients' paths.

    fontsize : int, float
        Fontsize of the title. The fontsizes of xlabel, ylabel,
        tick_params, and legend are resized with 0.85, 0.85, 0.75,
        and 0.75 fraction of title fontsize, respectively.

    grid : bool
        Whether to show (x,y) grid on the plot.

    legend: bool
        Whether to show legend on the plot.

    legendloc: string
        Legend location.

    xlabel : string or None
        Xlabel of the plot.

    ylabel : string or None
        Ylabel of the plot.

    title : string or None
        Title of the plot.

    yscale: string or None
        Scale for y-axis (coefficients). Valid options are
        "linear", "log", "symlog", "logit". More on:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.yscale.html

    bbox_to_anchor: tuple, list or None
        Relative coordinates for legend location outside of the plot.

    save_path: string or None
        The full or relative path to save the plot including the image format.
        For example "myplot.png" or "../../myplot.pdf"

    Returns None
    """
    # initializing feature_names
    if feature_names is None:
        if isinstance(est.X_, pd.DataFrame):
            feature_names = est.X_.columns.tolist()
        else:
            feature_names = ["F_{i}" for i in range(est.X_.shape[1])]
    else:
        if (
            isinstance(feature_names, list) or isinstance(feature_names, tuple)
        ) and len(feature_names) == est.X_.shape[1]:
            feature_names = feature_names
        else:
            raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing figsize
    if isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing fontsize
    if isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only integer and float types are allowed for fontsize.")

    # initializing linestyle
    if isinstance(linestyle, str):
        linestyle = linestyle
    else:
        raise TypeError("Only string type is allowed for linestyle.")

    # initializing grid
    if isinstance(grid, bool):
        grid = grid
    else:
        raise TypeError("Only bool type is allowed for grid.")

    # initializing legend
    if isinstance(legend, bool):
        legend = legend
    else:
        raise TypeError("Only bool type is allowed for legend.")

    # initializing xlabel
    if xlabel is None:
        xlabel = r"-$Log(\lambda)$"
    elif isinstance(xlabel, str):
        xlabel = xlabel
    else:
        raise TypeError("Only string type is allowed for xlabel.")

    # initializing ylabel
    if ylabel is None:
        ylabel = "Coefficients"
    elif isinstance(xlabel, str):
        ylabel = ylabel
    else:
        raise TypeError("Only string type is allowed for ylabel.")

    # initializing title
    if title is None:
        title = fr"Best $\lambda$ = {est.lambda_best_[0]:.2} with {len(np.nonzero(np.reshape(est.coef_, (1,-1)))[1])} Features"
    elif isinstance(title, str):
        title = title
    else:
        raise TypeError("Only string type is allowed for title.")

    # initializing bbox_to_anchor
    if bbox_to_anchor is None:
        bbox_to_anchor = (1.2, 0.5)
    elif isinstance(bbox_to_anchor, tuple) or isinstance(bbox_to_anchor, list):
        bbox_to_anchor = bbox_to_anchor
    else:
        raise TypeError("Only tuple or list type is allowed for bbox_to_anchor.")

    # initializing yscale
    if yscale is None:
        yscale = "linear"
    elif isinstance(yscale, str):
        yscale = yscale
    else:
        raise TypeError("Only string type is allowed for yscale.")

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    for i in list(np.nonzero(np.reshape(est.coef_, (1, -1)))[1]):
        plt.plot(
            -np.log(est.lambda_path_),
            (est.coef_path_.reshape(-1, est.coef_path_.shape[-1]))[i, :],
            label=feature_names[i],
        )

    ax.tick_params(axis="both", which="major", labelsize=fontsize * 0.75)
    ax.set_xlabel(xlabel, fontsize=fontsize * 0.85)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize * 0.85)
    ax.grid(grid)
    ax.set_yscale(yscale)

    if legend:
        ax.legend(
            loc=legendloc,
            bbox_to_anchor=bbox_to_anchor,
            ncol=1,
            prop={"size": fontsize * 0.75},
            framealpha=0.0,
            fancybox=True,
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()


def cv_score_plot(
    est,
    figsize=(10, 6),
    linestyle="--",
    linewidth=3,
    colors=None,
    marker="o",
    markersize=5,
    fontsize=18,
    grid=True,
    legend=True,
    legendloc="best",
    xlabel=None,
    ylabel=None,
    title=None,
    save_path=None,
):
    """Plot n-folds cross-validation scores vs -Log(lambda).

    Parameters
    ----------
    est : estimator
        The previously fitted estimator.

    figsize : tuple or list, as (width, height)
        Figure size.

    linestyle: string
        Linestyle of the vertical lamda lines.

    linewidth: integer or float
        Linewidth of the vertical lamda lines.

    colors: list or tuple
        Colors of the marker, errorbar line, max_lambda line,
        and best_lambda line, respectively. The default colors
        are ("red", "black", "cyan", "blue"). The length of the
        passed tuple/list should be always four.

    marker: str, optional, (default="o")
        Marker style. More details on:
        (https://matplotlib.org/2.1.1/api/markers_api.html#module-matplotlib.markers)

    markersize: intger or float
        Markersize.

    fontsize : int, float
        Fontsize of the title. The fontsizes of xlabel, ylabel,
        tick_params, and legend are resized with 0.85, 0.85, 0.75,
        and 0.85 fraction of title fontsize, respectively.

    grid : bool
        Whether to show (x,y) grid on the plot.

    legend: bool
        Whether to show legend on the plot.

    legendloc: string
        Legend location.

    xlabel : string or None
        Xlabel of the plot.

    ylabel : string or None
        Ylabel of the plot.

    title : string or None
        Title of the plot.

    save_path: string or None
        The full or relative path to save the image including the image format.
        For example "myplot.png" or "../../myplot.pdf"

    Returns None
    """

    # initializing figsize
    if isinstance(figsize, list) or isinstance(figsize, tuple):
        figsize = figsize
    else:
        raise TypeError("Only tuple and list types are allowed for figsize.")

    # initializing colors
    if colors is None:
        colors = ["red", "black", "cyan", "blue"]
    elif (isinstance(colors, tuple) or isinstance(colors, list)) and len(colors) == 4:
        colors = colors
    else:
        raise TypeError("Only tuple or list with length 4 is allowed for colors.")

    # initializing marker
    if isinstance(marker, str):
        marker = marker
    else:
        raise TypeError("Only str type is allowed for marker.")

    # initializing markersize
    if isinstance(markersize, float) or isinstance(markersize, int):
        markersize = markersize
    else:
        raise TypeError("Only int and float types are allowed for markersize.")

    # initializing fontsize
    if isinstance(fontsize, float) or isinstance(fontsize, int):
        fontsize = fontsize
    else:
        raise TypeError("Only integer and float types are allowed for fontsize.")

    # initializing linestyle
    if isinstance(linestyle, str):
        linestyle = linestyle
    else:
        raise TypeError("Only string type is allowed for linestyle.")

    # initializing linewidth
    if isinstance(linewidth, int) or isinstance(linewidth, float):
        linewidth = linewidth
    else:
        raise TypeError("Only string type is allowed for linewidth.")

    # initializing grid
    if isinstance(grid, bool):
        grid = grid
    else:
        raise TypeError("Only bool type is allowed for grid.")

    # initializing grid
    if isinstance(legend, bool):
        legend = legend
    else:
        raise TypeError("Only bool type is allowed for legend.")

    # initializing xlabel
    if xlabel is None:
        xlabel = r"-$Log(\lambda)$"
    elif isinstance(xlabel, str):
        xlabel = xlabel
    else:
        raise TypeError("Only string type is allowed for xlabel.")

    # initializing ylabel
    if ylabel is None:
        if est.scoring is None:
            if est.__module__ == "glmnet.linear":
                ylabel = fr"""{est.n_splits}-Folds CV Mean $R^2$"""
            elif est.__module__ == "glmnet.logistic":
                ylabel = f"""{est.n_splits}-Folds CV Mean ACCURACY"""
        else:
            ylabel = f"""{est.n_splits}-Folds CV Mean {' '.join((est.scoring).split("_")).upper()}"""
    elif isinstance(ylabel, str):
        ylabel = ylabel
    else:
        raise TypeError("Only string type is allowed for ylabel.")

    # initializing title
    if title is None:
        title = fr"Best $\lambda$ = {est.lambda_best_[0]:.2} with {len(np.nonzero(np.reshape(est.coef_, (1,-1)))[1])} Features"
    elif isinstance(title, str):
        title = title
    else:
        raise TypeError("Only string type is allowed for title.")

    # plotting
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        -np.log(est.lambda_path_),
        est.cv_mean_score_,
        yerr=est.cv_standard_error_,
        c=colors[0],
        ecolor=colors[1],
        marker=marker,
        markersize=markersize,
    )

    ax.vlines(
        -np.log(est.lambda_max_),
        ymin=min(est.cv_mean_score_) - 0.05,
        ymax=max(est.cv_mean_score_) + 0.05,
        linewidth=linewidth,
        linestyles=linestyle,
        colors=colors[2],
        label=r"max $\lambda$",
    )

    ax.vlines(
        -np.log(est.lambda_best_),
        ymin=min(est.cv_mean_score_) - 0.05,
        ymax=max(est.cv_mean_score_) + 0.05,
        linewidth=linewidth,
        linestyles=linestyle,
        colors=colors[3],
        label=r"best $\lambda$",
    )

    ax.set_ylim([min(est.cv_mean_score_) - 0.05, max(est.cv_mean_score_) + 0.05])
    ax.tick_params(axis="both", which="major", labelsize=fontsize * 0.75)
    ax.set_xlabel(xlabel, fontsize=fontsize * 0.85)
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize * 0.85)
    ax.grid(grid)

    if legend:
        ax.legend(loc=legendloc, prop={"size": fontsize * 0.85}, framealpha=0.0)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
