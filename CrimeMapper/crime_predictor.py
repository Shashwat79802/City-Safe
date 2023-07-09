import math
import numpy as np
import pandas as pd
from geopy import Point
from geopy.distance import distance
from math import radians, cos, sin, asin, sqrt, atan2
from pathlib import Path
import os
from joblib import Parallel, delayed
from sklearn.cluster import OPTICS


def calculate_distance(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    radius = 6371  # Earth's radius in kilometers
    distance = radius * c

    return distance


def process_crime(crime, df):
    data = df[crime].str.split(' ', expand=True).astype(float)
    data.dropna(axis=0, inplace=True)
    riots_data_rad = np.radians(data)

    optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
    optics.fit(riots_data_rad)

    cluster_labels = optics.labels_
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    sorted_counts_indices = np.argsort(counts)[::-1]
    top_clusters_indices = sorted_counts_indices[:4]

    cluster_list = []
    center_coordinates = []
    distance_list = []

    for cluster_index in top_clusters_indices:
        cluster_label = unique_labels[cluster_index]
        cluster_indices = np.where(cluster_labels == cluster_label)[0]

        cluster_points = data.iloc[cluster_indices]
        cluster_inner_list = []

        if cluster_label != -1:
            with open('coordinates' + str(cluster_label) + '.txt', 'w') as f:
                f.writelines(str(cluster_points))

            with open('coordinates' + str(cluster_label) + '.txt', 'r') as f:
                lines = f.readlines()

                for line in lines[1:]:
                    values = line.split()
                    coordinate = tuple(float(value) for value in values[1:])
                    cluster_inner_list.append(coordinate)

                cluster_list.append(cluster_inner_list)

    for coordinates in cluster_list:
        centroid = Point(sum(latitude for latitude, _ in coordinates) / len(coordinates),
                         sum(longitude for _, longitude in coordinates) / len(coordinates))

        centermost_coordinate = min(coordinates, key=lambda coord: distance(Point(coord), centroid).km)
        center_coordinates.append(centermost_coordinate)

    for i in range(0, len(center_coordinates)):
        center_latitude = radians(center_coordinates[i][0])
        center_longitude = radians(center_coordinates[i][1])
        max_distance = 0

        for j in range(0, len(cluster_list[i])):
            other_latitude = radians(cluster_list[i][j][0])
            other_longitude = radians(cluster_list[i][j][0])

            dlat = center_latitude - other_latitude
            dlon = center_longitude - other_longitude

            formula = sin(dlat / 2) ** 2 + cos(other_latitude) * cos(center_latitude) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(formula), sqrt(1 - formula))
            r = 6371  # in kilometers
            result = c * r

            max_distance = max(max_distance, result)

        distance_list.append(max_distance)

    return center_coordinates[:3], distance_list[:3]


def the_crime_detector() -> dict:
    final_json = {
        "data": {
            "Murder and Assault": [],
            "Kidnapping and Abduction": [],
            "Sexual Assault": [],
            "Theft and Burglary": [],
            "Riots": [],
            "Accident Prone Areas": [],
        }
    }

    jp_file = os.path.join(Path(__file__).resolve().parent.parent, "CrimeMapper/Jaipur.csv")
    crimes = ['Murder and Assault', 'Kidnapping and Abduction', 'Sexual Assault', 'Theft and Burglary', 'Riots',
              'Accident Prone Areas']

    df = pd.read_csv(jp_file)

    results = Parallel(n_jobs=-1)(delayed(process_crime)(crime, df) for crime in crimes)

    for i, crime in enumerate(crimes):
        centers, distances = results[i]
        for j in range(3):
            final_json["data"][crime].append({
                "coordinates": [centers[j][0], centers[j][1]],
                "distance": distances[j]
            })

    return final_json
