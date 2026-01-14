"""
Module for locating and clustering road sign traces based on geospatial data.

This module provides functionality to analyze GPS traces for road sign detection and cluster them into logical groups
based on various centroid calculation methods, such as K-Means, DBSCAN, or simple geohash neighborhood aggregation.
It processes trace data, calculates centroids for clusters, and supports flexible strategies for geospatial analytics.

Classes:
    SignLocator: A class to locate and cluster road sign traces using geospatial data.
"""
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pygeohash as pgh
from geopandas import GeoDataFrame
from shapely import Polygon, Point
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


class SignLocator:
    """
    Class used for locating road signs based on analyzed GPS traces.

    This class provides functionality to process and analyze GPS traces to derive
    clustered sign locations using geohash tiles or centroid calculation methods,
    such as 'simple', 'k-means', and 'DBSCAN'. Data is processed into GeoDataFrames
    for spatial manipulations and clustering. Designed for applications involving
    road sign detection and geospatial data analysis.

    :ivar precision: Precision level of geohash encoding, used for spatial grouping
                     of GPS trace data.
    :type precision: int
    """

    def __init__(self, precision: int = 8):
        self.precision = precision

    @staticmethod
    def extract_traces(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract traces from the given DataFrame based on confidence level.

        This function filters the input DataFrame to retain only rows where the
        "confidence" column has a value greater than 0.5. The resulting
        filtered DataFrame is returned.

        :param df: Input DataFrame containing the data to be filtered.
        :type df: pd.DataFrame
        :return: A DataFrame containing only rows with "confidence" values > 0.5.
        :rtype: pd.DataFrame
        """
        return df[df["confidence"] > 0.5]

    def calculate_sign_locations(self,
                                 df: pd.DataFrame,
                                 center_calc: str = "simple",
                                 min_samples: int = 4,
                                 min_radius: float = 4.0,
                                 min_confidence: float = 7.0,
                                 max_spread: float = 60.0) -> tuple[GeoDataFrame, GeoDataFrame]:
        """
        Calculates potential sign locations from the given dataset based on GPS traces and specified clustering
        methods.

        This function processes a DataFrame containing GPS trace data to identify geospatial groupings of points that
        might represent the locations of signs. It supports different clustering methods for determining the center
        locations of the groups and returns the processed GeoDataFrame along with geohashed tile data.

        Clustering methods supported
        Simple:
        This method performs an iterative calculation of centroids using cluster candidates and merging
        strategies. Cluster candidates are collected by including traces from geohash neighbors that share the same
        sign type and bearing.
        It aims to consolidate points into a single centroid per cluster. The process
        terminates if the centroids converge or the maximum number of iterations is reached.

        DBSCAN:
        This method utilizes DBSCAN clustering to identify clusters
        in a GeoDataFrame based on spatial proximity. The centroids of
        the identified clusters are then calculated as weighted centroids
        based on a specified clustering attribute.

        K-Means:
        This method applies an additional k-means clustering step to improve cluster centroids
        obtained from an earlier DBSCAN process. It is applied only to clusters with a significant
        spread, determined by a bounding box spread threshold. The number of clusters in the k-means
        step is dynamically derived based on the spread of the cluster. After refining clusters using
        k-means, weighted centroids are recomputed and returned in the final GeoDataFrame.



        Arguments:
            df: pd.DataFrame
                Input DataFrame containing GPS traces. Must include columns 'gps_traces_lat' and 'gps_traces_lon'.
            center_calc: str, optional
                The method used to calculate centroid locations for clustered points. Options are:
                - 'simple': Use simple weighted centroids.
                - 'k-means': Use k-means clustering.
                - 'DBSCAN': Use DBSCAN clustering. Defaults to 'simple'.
            min_samples: int, optional
                Minimum number of samples required for clustering. Default is 4.
            min_radius: float, optional
                The minimum radius (distance) used for clustering. Default is 4.0.
            min_confidence: float, optional
                Minimum confidence value used to filter centroid calculation results. Default is 7.0.
                Only relevant for 'simple' method.
            max_spread: float, optional
                Maximum allowed spread of the bounding box of a cluster. Default is 60.0.
                Only relevant for 'k-means' method.

        Returns:
            tuple[GeoDataFrame, GeoDataFrame]
                A tuple containing:
                - A GeoDataFrame with clustered data and calculated centroids based on the selected method.
                - A GeoDataFrame containing geohashed tile information derived from the input data.

        Raises:
            None
        """

        gdf = gpd.GeoDataFrame(df,
                               geometry=gpd.points_from_xy(df["gps_traces_lon"], df["gps_traces_lat"]),
                               crs="EPSG:4326")

        gdf["geohash"] = gdf.apply(
            lambda row: pgh.encode(row["gps_traces_lat"], row["gps_traces_lon"], precision=self.precision), axis=1)

        gdf_tiles = self._geohash_gdf(gdf)

        if center_calc == "k-means":
            gdf = self._calc_kmeans_centroids(gdf, min_samples, min_radius, max_spread)
        elif center_calc == "DBSCAN":
            gdf = self._calc_dbscan_centroids(gdf, min_samples, min_radius)
        else:
            gdf = self._calc_simple_centroids(gdf, min_samples, min_radius, min_confidence)

        return gdf, gdf_tiles

    def _calc_kmeans_centroids(self, gdf: GeoDataFrame, min_samples: int, min_radius: float,
                               max_spread: float) -> GeoDataFrame:

        gdf = self._collect_cluster_candidates_dbscan(gdf, min_samples, min_radius)

        refined = []
        for cluster_id, grp in gdf.groupby("cluster_id"):
            bbox_spread = self._cluster_spread_bbox(grp)

            # k-means only improves results when the spread is high
            if bbox_spread > max_spread:
                coords = np.column_stack([grp.geometry.x, grp.geometry.y])

                # num cluster is at least 2 (one more than after dbscan)
                k = max(2, int(bbox_spread / (max_spread / 2)))
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = kmeans.fit_predict(coords)

                grp["labels"] = labels
                grp["cluster_id"] = grp["cluster_id"].astype(str) + "_" + labels.astype(str)
                grp = grp.drop(columns=["labels"])

            refined.append(grp)

        gdf_refined = pd.concat(refined, ignore_index=True)

        return self._merge_weighted_centroid(gdf_refined, by=["cluster_id"])

    def _calc_dbscan_centroids(self, gdf: GeoDataFrame, min_samples: int, min_radius: float) -> gpd.GeoDataFrame:
        gdf = self._collect_cluster_candidates_dbscan(gdf, min_samples, min_radius)
        return self._merge_weighted_centroid(gdf, by=["cluster_id"])

    @staticmethod
    def _collect_cluster_candidates_dbscan(gdf: GeoDataFrame, min_samples: int, min_radius: float) -> GeoDataFrame:
        gdf_m = gdf.to_crs(epsg=25833)

        results = []
        for key, grp in gdf_m.groupby(["sign_type", "bearing"]):
            coords = np.column_stack([grp.geometry.x, grp.geometry.y])
            db = DBSCAN(eps=min_radius, min_samples=min_samples, metric="euclidean").fit(coords)

            # we add a unique identifier to the cluster id to ensure uniqueness across the global df
            grp["dblabels"] = db.labels_
            grp["cluster_id"] = grp["dblabels"].astype(str) + "_" + hashlib.sha1(
                f"{key[0]}|{key[1]}".encode()).hexdigest()
            clustered = grp[grp["dblabels"] >= 0]

            results.append(clustered)

        results_df = pd.concat(results, ignore_index=True)
        return gpd.GeoDataFrame(results_df, geometry="geometry", crs="EPSG:25833").to_crs(4326)

    def _calc_simple_centroids(self,
                               gdf: GeoDataFrame,
                               min_samples: int,
                               min_radius: float,
                               min_confidence: float,
                               max_iter: int = 10) -> gpd.GeoDataFrame:

        for i in range(max_iter):

            print(f"Iteration {i}: {len(gdf)} points")

            prev = gdf.copy()
            # we will have duplicate candidates in different clusters in this step
            gdf = self._collect_cluster_candidates_with_neighbors(gdf, min_samples, min_radius, min_confidence)
            # now we reduce our candidates to one centroid per trace_id (ride) and cluster
            gdf = self._merge_weighted_centroid(gdf, by=["cluster_id", "trace_id"])
            # in the last step we try to reduce the trace_id + cluster_id centroid to a single centroid per cluster
            gdf = self._merge_weighted_centroid(gdf, by=["cluster_id"])

            # re-geohash for the next iteration to merge centroids of the same cluster spread across different geohashes
            gdf["gps_traces_lat"] = gdf.geometry.y
            gdf["gps_traces_lon"] = gdf.geometry.x

            gdf["geohash"] = gdf.apply(
                lambda row: pgh.encode(row["gps_traces_lat"], row["gps_traces_lon"], precision=self.precision), axis=1)

            if i > 0 and self._has_converged(prev, gdf):
                print(f"Converged after {i + 1} iterations")
                break

        return gdf

    @staticmethod
    def _has_converged(prev: GeoDataFrame, curr: GeoDataFrame) -> bool:
        return prev["is_centroid"].count() == curr["is_centroid"].count()

    @staticmethod
    def _use_neighbors(gdf: GeoDataFrame,
                       neighbor: GeoDataFrame,
                       min_samples: int,
                       min_radius: float,
                       min_confidence: float) -> bool:
        gdf_m = gdf.to_crs(25833)
        neighbor_m = neighbor.to_crs(25833)

        centroid_dist = gdf_m.union_all().distance(neighbor_m.union_all())

        # we only consider neighbors if we don't have enough samples or the confidence is too low
        # and if the distance between current geohash centroid and neighbor centroid is lower than diameter
        return (len(gdf) < min_samples or gdf["confidence"].mean() < min_confidence) and centroid_dist < min_radius * 2

    def _collect_cluster_candidates_with_neighbors(self,
                                                   gdf: GeoDataFrame,
                                                   min_samples: int,
                                                   min_radius: float,
                                                   min_confidence: float) -> GeoDataFrame:
        # group the data first by geohash, sign_type and bearing (direction) - bearing works in our case as it is
        # constant per road sign trace (which is most likely not the case in a real world scenario)
        cluster_groups = {grp: sub for grp, sub in gdf.groupby(["geohash", "sign_type", "bearing"], as_index=False)}
        # we iterate over each group and collect the candidates for a cluster
        results = []
        for key, grp in cluster_groups.items():
            # get the neighbors of the current geohash
            neighbors = self._geohash_with_neighbors(key[0])

            # neighbor geohashes that actually contain traces of the same group - needed to create the cluster id
            neighborhood_tiles = [n for n in neighbors if (n, key[1], key[2]) in cluster_groups and
                                  self._use_neighbors(grp, cluster_groups[(n, key[1], key[2])], min_samples,
                                                      min_radius, min_confidence)]

            # collect traces from the neighborhood - we access the traces via (geohash, sign_type, bearing) tuple key
            neighborhood_traces = [cluster_groups[(n, key[1], key[2])] for n in neighborhood_tiles]

            if not neighborhood_traces:
                continue

            # we create a synthetic cluster id based on all adjacent geohashes that we can use later for grouping
            cluster_id = (hashlib.sha1(str(f"{key[1]}|{key[2]}|" + "|".join(sorted(neighborhood_tiles))).encode())
                          .hexdigest())

            candidates = pd.concat(neighborhood_traces)
            candidates["cluster_id"] = cluster_id
            candidates["num_traces"] = len(candidates)

            results.append(candidates)

        result_df = pd.concat(results, ignore_index=True)

        return gpd.GeoDataFrame(result_df,
                                geometry=gpd.points_from_xy(result_df["gps_traces_lon"], result_df["gps_traces_lat"]),
                                crs="EPSG:4326")

    @staticmethod
    def _merge_weighted_centroid(gdf: GeoDataFrame, by: list[str]) -> GeoDataFrame:
        # project to metric CRS
        gdf_m = gdf.to_crs(epsg=25833)

        # we calculate weighted centroids for each cluster based on the confidence of the traces
        def _weighted_centroid(sub):
            weight = sub["confidence"].to_numpy() * 2

            x = (sub.geometry.x.to_numpy() * weight).sum() / weight.sum()
            y = (sub.geometry.y.to_numpy() * weight).sum() / weight.sum()

            # look for the uuid with the most occurrences
            uuid = sub.groupby(["uuid"]).count().idxmax().iloc[0]

            weighted_series = pd.Series({
                "geometry": Point(x, y),
                "confidence": sub["confidence"].mean(),
                # we keep the uuid for later validation
                "uuid": uuid,
                "sign_type": sub["sign_type"].iloc[0],
                "bearing": sub["bearing"].iloc[0],
                "is_centroid": True
            })

            if "trace_id" not in by:
                weighted_series["trace_id"] = sub["trace_id"].iloc[0]

            return weighted_series

        merged = gdf_m.groupby(by=by, as_index=False).apply(_weighted_centroid, include_groups=False)
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:25833").drop_duplicates()

        # project back to geographic CRS
        return merged.to_crs(epsg=4326)

    @staticmethod
    def _cluster_spread_bbox(gdf: GeoDataFrame):
        # calculate bounding box of cluster
        gdf_m = gdf.to_crs(epsg=25833)

        x = gdf_m.geometry.x
        y = gdf_m.geometry.y

        dx = x.max() - x.min()
        dy = y.max() - y.min()

        # and return the max distance of any axis to estimate the spread
        return max(dx, dy)

    @staticmethod
    def _geohash_with_neighbors(gh: str) -> list[str]:
        neighbors = [gh]

        # we collect all 8 neighbors clockwise
        for direction in ["top", "right", "bottom", "bottom", "left", "left", "top", "top"]:
            neighbor = pgh.neighbor.get_adjacent(gh, direction)
            neighbors.append(neighbor)
            gh = neighbor

        return neighbors

    @staticmethod
    def _geohash_to_polygon(gh: str) -> Polygon:
        lat, lon, lat_err, lon_err = pgh.decode_exactly(gh)

        min_lat = lat - lat_err
        max_lat = lat + lat_err
        min_lon = lon - lon_err
        max_lon = lon + lon_err

        return Polygon([
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, max_lat),
            (max_lon, min_lat),
            (min_lon, min_lat),
        ])

    def _geohash_gdf(self, gdf, geohash_col="geohash"):
        polys = (
            gdf[geohash_col]
            .drop_duplicates()
            .rename("geohash")
            .to_frame()
        )

        polys["geometry"] = polys["geohash"].apply(self._geohash_to_polygon)

        return gpd.GeoDataFrame(
            polys,
            geometry="geometry",
            crs="EPSG:4326"
        )
