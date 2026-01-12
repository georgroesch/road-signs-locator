import uuid

import numpy as np
import pandas as pd


def emit_traces(input_df: pd.DataFrame):
    # we only keep relevant columns
    df = input_df[["latitude", "longitude", "uuid", "stvo_nummer", "standort_ausrichtung"]]

    # remove rows without bearing as we can't reliably emit traces for those
    df = df[~df["standort_ausrichtung"].isna()]

    # we add a speed column with constant speed - we simulate 10 cars emitting traces
    speed_vals = []
    for _ in range(10):
         speed_vals.append(pd.Series(_set_speed_in_ms(df.shape[0])))

    # now we explode it so that we have multiple speed observations per road sign
    df["speed"] = list(pd.concat(speed_vals, axis=1).to_numpy())
    df = df.explode("speed", ignore_index=True)

    # we now simulate multiple gps traces based on the original location of the road sign
    df["bearing"] = df["standort_ausrichtung"] #_sanitize_bearing(df["standort_ausrichtung"])
    df["gps_traces"] = _calc_observation_positions(df["speed"], df["latitude"], df["longitude"], df["bearing"])


    df = df.explode("gps_traces", ignore_index=True)

    # separate gps coords from confidence
    df["gps_traces_lat"] = df.gps_traces.apply(lambda x: x[0])
    df["gps_traces_lon"] = df.gps_traces.apply(lambda x: x[1])
    df["confidence"] = df.gps_traces.apply(lambda x: x[2])
    df["trace_id"] = df.gps_traces.apply(lambda x: x[3])

    # we keep the uuid to validate our results later
    # stvo nummer is the sign type; standort_ausrichtung is the bearing of the road sign / trace
    df["sign_type"] = df["stvo_nummer"]
    return df[["uuid", "sign_type", "bearing", "confidence", "gps_traces_lat", "gps_traces_lon", "trace_id"]]


def _sanitize_bearing(bearing: pd.Series):
    mask = bearing.isna()
    fallback = np.random.uniform(0, 360, size=mask.sum())
    bearing = bearing.copy()
    bearing.loc[mask] = fallback
    return bearing


def _set_speed_in_ms(size: int):
    # we assume a max of 120 km/h
    return np.random.randint(5, 34, size=size)


def _longitudinal_offsets(n: int, max_dist=50) -> np.ndarray:
    # we set observations in front of the road sign -
    # this returns an evenly distributed list of offsets as we assume constant speed
    return np.linspace(-max_dist, 0, n)


def _gps_noise(n):
    longitudinal_noise = np.random.normal(0, 3.0, size=n)  # meters
    lateral_noise = np.random.normal(0, 3.0, size=n)  # meters
    return longitudinal_noise, lateral_noise


def _meters_to_latlon(dx, dy, lat):
    dlat = dy / 111_111
    dlon = dx / (111_111 * np.cos(np.radians(lat)))
    return dlat, dlon

def _confidence_from_distance(dist: float) -> float:
    """
    Simplified confidence model.

    We intentionally avoid modelling the empirical confidence peak
    (~15–25 m before the physical sign), as this would introduce a
    systematic forward bias in the estimated sign location.

    A more realistic, trace-aware model (e.g. peak detection using
    timestamps and post-peak filtering) will be introduced in a
    later iteration once temporal information is implemented in this emitter.

    For now, confidence is modelled as a monotonically decreasing
    function of distance.
    """

    # we assume peak confidence around 15–25 m for sign recognition from a car - standard deviation 5 m
    # considering motion blur, sign resolution, perspective distortion, sign fully visible, etc.
    #return np.exp(-((dist - 20) ** 2) / (2 * 5 ** 2))

    sigma = 20.0  # controls how fast confidence decays with distance
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))



def _calc_num_observations(speed: pd.Series, visiable_dist: int = 50, sampling_rate: int = 5,
                           max_obs: int = 25) -> np.ndarray:
    # we assume a sampling rate of 5Hz and a vision of max 50m - depending on environmental conditions
    # like weather, road shape, urban vs. country side areas, objects etc.
    # for simplicity those external conditions remain constant
    # So e.g. for a speed of 10m/s we would have 25 oberservations for a single road sign
    n = visiable_dist // speed * sampling_rate
    n = np.maximum(n, 1)
    return np.minimum(n, max_obs)


def _bearing_to_unit_vector(bearing_deg: pd.Series):
    # the bearing of the sign needs to be corrected by 90° as 0° would point to the right (east) in mathematical terms
    # where as 0° would be north in geographical terms
    theta = np.radians(bearing_deg + 90)
    return np.cos(theta), np.sin(theta)


def _calc_observation_positions(speed: pd.Series, lat: pd.Series, lon: pd.Series, bearing: pd.Series):
    # GPS bearing: 0° = North, clockwise
    dir_x, dir_y = _bearing_to_unit_vector(bearing)

    n_obs = _calc_num_observations(speed)

    gps_traces = []

    for i in range(len(n_obs)):
        n = n_obs[i]

        ln, lat_n = _gps_noise(n)
        dx_unit, dy_unit = dir_x[i], dir_y[i]
        offsets = _longitudinal_offsets(int(n))

        dist = np.abs(offsets)
        # perception confidence * localization confidence
        confidence = _confidence_from_distance(dist)

        dx = (offsets + ln) * dx_unit - lat_n * dy_unit
        dy = (offsets + ln) * dy_unit + lat_n * dx_unit

        dlat, dlon = _meters_to_latlon(dx, dy, lat[i])

        obs_lat = lat[i] + dlat
        obs_lon = lon[i] + dlon

        # we create a unique trace id for each ride
        trace_id = [uuid.uuid1()] * n

        # let's create an array of tuples for obs_lat and obs_lon
        traces = np.column_stack([obs_lat, obs_lon, confidence, trace_id])
        gps_traces.append(traces)

    return gps_traces
