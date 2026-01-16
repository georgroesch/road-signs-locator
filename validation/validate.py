import contextily as ctx
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors, cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_scalebar.scalebar import ScaleBar


def plot_geoframes(title: str,
                   ground_truth: gp.GeoDataFrame,
                   traces: dict[str, tuple[gp.GeoDataFrame, dict]],
                   tiles: gp.GeoDataFrame = None,
                   sign_point=None):
    """
    Plots GeoDataFrames and overlays additional map elements.

    This function visualizes GeoDataFrames, including ground truth data, optional geohash tiles, and trace
    information with varying confidence levels on a basemap. The map is centered on a specific point and is
    zoomed to a defined radius.

    Parameters:
    title: str
        The title of the plot to be displayed above the visualization.
    ground_truth: GeoDataFrame
        GeoDataFrame containing the ground truth data to be plotted. It is expected to have a defined CRS.
    traces: dict[str, tuple[GeoDataFrame, str]]
        A dictionary where keys represent labels for traces, and values represent a tuple containing a
        GeoDataFrame with trace data and a string specifying a colormap for their visualization. The confidence
        levels of the data points are expected to be provided in a column named "confidence".
    tiles: GeoDataFrame, optional
        GeoDataFrame containing geohash tiles to overlay on the map. Must have a defined CRS.
    sign_point: NoneType or geometric object, optional
        The geometric point to center the map on. If not provided, it defaults to the geometry of the first
        entry in the ground truth GeoDataFrame.

    Raises:
    ValueError
        If the ground truth GeoDataFrame does not contain any geometry data.

    Returns:
    None
    """
    single_trace = len(traces) == 1

    ground_truth = ground_truth.to_crs(epsg=3857)

    _, ax = plt.subplots(figsize=(10, 10))

    if tiles is not None:
        tiles = tiles.to_crs(epsg=3857)
        tiles.plot(
            ax=ax,
            facecolor="none",
            edgecolor="blue",
            linewidth=2,
            alpha=0.6,
            label="Geohash Tiles",
            zorder=3,
        )

    for label, trace in traces.items():
        style = trace[1]
        color = style.get("color")
        cmap = style.get("cmap")

        gdf = trace[0].to_crs(epsg=3857)

        gdf.plot(
            ax=ax,
            markersize=20,
            label=label,
            zorder=4,
            column="confidence" if single_trace else None,
            cmap=cmap,
            color=color
        )

    ground_truth.plot(
        ax=ax,
        color="red",
        marker="X",
        markersize=60,
        zorder=5
    )

    # legend
    legend_elements = [Line2D(
        [0], [0],
        marker="X",
        color="red",
        linestyle="None",
        markersize=8,
        label="Road Sign (Ground Truth)"
    )]

    if tiles is not None:
        legend_elements.append(
            Patch(
                facecolor="none",
                edgecolor="blue",
                linewidth=2,
                label="Geohash Tiles"
            )
        )

    if single_trace:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markeredgecolor="black",
                markersize=6,
                label="GPS Traces (colored by confidence)"
            )
        )

    else:
        for label, trace in traces.items():
            style = trace[1]
            color = style.get("color")

            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=6,
                    label=label
                )
            )

    ax.legend(handles=legend_elements, loc="upper right")

    if single_trace:
        cmap = list(traces.values())[0][1].get("cmap")

        norm = colors.Normalize(vmin=list(traces.values())[0][0]["confidence"].min(), vmax=1.0)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("Confidence")

    # Center map on ground truth - we only look at a certain area of Rostock
    if sign_point is None:
        sign_point = ground_truth.geometry.iloc[0]

    x_center, y_center = sign_point.x, sign_point.y
    radius = 100  # meters

    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)

    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # add scalebar (meters)
    ax.add_artist(ScaleBar(1, location="lower center"))

    ax.set_axis_off()
    ax.set_title(title)
    plt.show()


def validate_results(gdf_points: gp.GeoDataFrame, gdf_signs: gp.GeoDataFrame, method_name: str, epsg=25833):
    """
    Validates the results of geospatial data processing by calculating the distance between corresponding geometries
    from two GeoDataFrames and associating the calculation with a specific method.

    Parameters:
    gdf_points (gp.GeoDataFrame): A GeoDataFrame containing point geometries and additional attributes, including "uuid"
    used for merging.
    gdf_signs (gp.GeoDataFrame): A GeoDataFrame containing sign geometries and additional attributes, also including "uuid"
    used for merging.
    method_name (str): The name of the method used for validation, added to the resulting DataFrame for reference.

    Returns:
    gp.GeoDataFrame: A new GeoDataFrame containing the following columns:
    - method: The name of the selected validation method.
    - distance_to_sign: The calculated distance between corresponding sign and point geometries (in EPSG:25833).
    - is_centroid: Values carried over from the input GeoDataFrames, if applicable, indicating additional metadata.

    """
    # let's validate the results - epsg 25833 for Rostock
    validation_df = gdf_signs.to_crs(epsg=epsg).merge(gdf_points.to_crs(epsg=epsg), how="inner", on=["uuid"],
                                                      suffixes=("_sign", "_point"))
    validation_df["distance_to_sign"] = validation_df.geometry_sign.distance(validation_df.geometry_point)

    validation_df["method"] = method_name

    return validation_df[["method", "distance_to_sign", "is_centroid", "uuid"]]


def plot_error_distributions(validation_df: pd.DataFrame):
    """
    Plots the error distributions for different methods using Kernel Density Estimation (KDE).

    This function generates a KDE plot for the 'distance_to_sign' column in the provided
    DataFrame, grouped by the 'method' column. It allows for comparison of error
    distributions between different methods.

    Arguments:
        validation_df (pd.DataFrame): The input DataFrame containing the error distances
                                      as 'distance_to_sign' and the corresponding methods
                                      as 'method'.

    """
    plt.figure(figsize=(10, 6))

    for method, grp in validation_df.groupby("method"):
        grp["distance_to_sign"].plot.kde(
            linewidth=2,
            label=method
        )

    plt.xlabel("Error distance [m]")
    plt.ylabel("Density")
    plt.title("Error Distribution Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_aggregated_kpis(validation_df: pd.DataFrame, gdf_signs: gp.GeoDataFrame):
    """
    Plots aggregated Key Performance Indicators (KPIs) for different validation methods.

    The function calculates statistical summaries of certain KPI metrics based on a given
    validation DataFrame and a GeoDataFrame of signs. It then visualizes these metrics in
    a grouped bar chart layout for a clear comparison between methods.

    Arguments:
        validation_df (pd.DataFrame): DataFrame containing validation data where each row
            corresponds to an observation. It must include the columns "method",
            "distance_to_sign", and "is_centroid".
        gdf_signs (gp.GeoDataFrame): GeoDataFrame containing data about positional signs
            used for validation. The length of this GeoDataFrame is used to compute
            centroid differences.

    Raises:
        ValueError: If the required columns are missing in `validation_df`.

    Visualization:
        - The function produces a multi-metric bar chart visualization, where each subplot
          corresponds to a specific KPI metric (e.g., mean error, median error, centroid
          differences, etc.).
        - Metrics are shown per validation method for comparison.
        - Annotated text labels display the actual values on the bars.
    """

    kpi_df = validation_df.groupby("method").agg(
        mean_error=("distance_to_sign", "mean"),
        median_error=("distance_to_sign", "median"),
        min_error=("distance_to_sign", "min"),
        max_error=("distance_to_sign", "max"),
        num_centroids=("is_centroid", "sum")).reset_index()

    kpi_df["centroid_diff"] = (kpi_df["num_centroids"] - len(gdf_signs))
    kpi_df["centroid_diff_rel"] = ((kpi_df["num_centroids"] - len(gdf_signs)).abs() / len(gdf_signs)) * 100

    metrics = [
        ("mean_error", "Mean Error [m]"),
        ("median_error", "Median Error [m]"),
        ("max_error", "Max Error [m]"),
        ("min_error", "Min Error [m]"),
        ("centroid_diff", "Centroid Diff"),
        ("centroid_diff_rel", "Centroid Diff %")
    ]

    fig, axes = plt.subplots(
        1, len(metrics),
        figsize=(18, 5),
        sharex=True
    )

    for ax, (col, label) in zip(axes, metrics):
        ax.bar(kpi_df["method"], kpi_df[col])
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y")

        # annotate values
        for i, v in enumerate(kpi_df[col]):
            ax.text(i, v, f"{v:.2f}" if col != "centroid_diff" else f"{int(v)}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("Aggregated Validation KPIs per Method", fontsize=14)
    plt.tight_layout()
    plt.show()
