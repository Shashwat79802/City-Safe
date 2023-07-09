from sklearn.cluster import OPTICS
import numpy as np
import pandas as pd
from geopy import Point
from geopy.distance import distance
from math import radians, cos, sin, asin, sqrt, atan2
from pathlib import Path
import os


def the_crime_detector() -> dict:
    final_json = {
        "data": {
            "Murder and Assault": [
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                }
            ],
            "Kidnapping and Abduction": [
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                }
            ],
            "Sexual Assault": [
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                }
            ],
            "Theft and Burglary": [
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                }
            ],
            "Riots": [
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                }
            ],
            "Accident Prone Areas": [
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                },
                {
                    "coordinates": "coordinate1",
                    "distance": "distance1"
                }
            ],
        }
    }

    jp_file = os.path.join(Path(__file__).resolve().parent.parent, "CrimeMapper/Jaipur.csv")
    crimes = ['Murder and Assault', 'Kidnapping and Abduction', 'Sexual Assault', 'Theft and Burglary', 'Riots', 'Accident Prone Areas']

    df = pd.read_csv(jp_file)
    cluster_list = []

    for crime in crimes:
        data = df[crime].str.split(' ', expand=True).astype(float)
        data.dropna(axis=0, inplace=True)
        riots_data_rad = np.radians(data)

        optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
        optics.fit(riots_data_rad)

        cluster_labels = optics.labels_
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        sorted_counts_indices = np.argsort(counts)[::-1]
        top_clusters_indices = sorted_counts_indices[:4]

        for cluster_index in top_clusters_indices:
            cluster_label = unique_labels[cluster_index]
            cluster_indices = np.where(cluster_labels == cluster_label)[0]

            cluster_points = data.iloc[cluster_indices]
            cluster_inner_list = []

            if cluster_label != -1:
                # print(f"Cluster {cluster_label} points:")
                # print(cluster_points)

                with(open('coordinates' + str(cluster_label) + '.txt', 'w') as f):
                    f.writelines(str(cluster_points))
                f.close()

                with(open('coordinates' + str(cluster_label) + '.txt', 'r') as f):
                    lines = f.readlines()

                    for line in lines[1:]:
                        values = line.split()
                        coordinate = tuple(float(value) for value in values[1:])
                        cluster_inner_list.append(coordinate)

                    cluster_list.append(cluster_inner_list)
        # print(cluster_list)

        center_coordinates = []

        for coordinates in cluster_list:
            centroid = Point(sum(latitude for latitude, _ in coordinates) / len(coordinates),
                             sum(longitude for _, longitude in coordinates) / len(coordinates))

            centermost_coordinate = min(coordinates, key=lambda coord: distance(Point(coord), centroid).km)
            center_coordinates.append(centermost_coordinate)

        # print(center_coordinates)
        final_json["data"][crime][0]["coordinates"] = [center_coordinates[0][0], center_coordinates[0][1]]
        final_json["data"][crime][1]["coordinates"] = [center_coordinates[1][0], center_coordinates[1][1]]
        final_json["data"][crime][2]["coordinates"] = [center_coordinates[2][0], center_coordinates[2][1]]

        distance_list = []

        for i in range(0, len(center_coordinates)):

            center_latitude = radians(center_coordinates[i][0])
            center_longitude = radians(center_coordinates[i][1])
            max_distance = 0

            for j in range(0, len(cluster_list[i])):
                other_latitude = radians(cluster_list[i][j][0])
                other_longitude = radians(cluster_list[i][j][0])

                dlat = center_latitude - other_latitude
                dlon = center_longitude - other_longitude

                formula = sin(dlat / 2)**2 + cos(other_latitude) * cos(center_latitude) * sin(dlon / 2)**2
                c = 2 * atan2(sqrt(formula), sqrt(1-formula))
                r = 6371 # in kilometer
                result = c * r

                max_distance = max(max_distance, result)

            distance_list.append(max_distance)

        final_json["data"][crime][0]["distance"] = distance_list[0]
        final_json["data"][crime][1]["distance"] = distance_list[1]
        final_json["data"][crime][2]["distance"] = distance_list[2]

    return final_json


# Extract the latitude and longitude values from the 'Kidnapping' column
# data = df['Riots'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels
# unique_labels = np.unique(cluster_labels)
#
# # Iterate over the unique cluster labels and print the center coordinate of each cluster
# for label in unique_labels:
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Calculate the center coordinate of the current cluster
#     center_coordinate = cluster_points.mean(axis=0)
#
#     # Print the center coordinate of the cluster
#     print(f"Cluster {label} center coordinate:")
#     print(center_coordinate)
#     print()
#
# # ---------------------------------------------------------------------------------------------------------------------#
#
# # Assuming 'df' is your DataFrame containing the 'Murder and Assault' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Murder and Assault' column
# data = df['Murder and Assault'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top 3-4 clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and print their points
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Print the cluster points
#     print(f"Cluster {cluster_label} points:")
#     print(cluster_points)
#     print()
#
# # Assuming 'df' is your DataFrame containing the 'Accident Prone Areas' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Accident Prone Areas' column
# data = df['Murder and Assault'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and calculate their center coordinates
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Calculate the center coordinates of the current cluster
#     center_lat = cluster_points[0].mean()
#     center_lon = cluster_points[1].mean()
#
#     # Print the center coordinates of the current cluster
#     print(f"Center coordinates of Cluster {cluster_label}: ({center_lat}, {center_lon})")
#
# # ---------------------------------------------------------------------------------------------------------------------#
#
# # Extract the latitude and longitude values from the 'Theft and Burglary' column
# data = df['Theft and Burglary'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top 3-4 clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and print their points
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Print the cluster points
#     print(f"Cluster {cluster_label} points:")
#     print(cluster_points)
#     print()
#
# # Assuming 'df' is your DataFrame containing the 'Accident Prone Areas' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Accident Prone Areas' column
# data = df['Theft and Burglary'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and calculate their center coordinates
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Calculate the center coordinates of the current cluster
#     center_lat = cluster_points[0].mean()
#     center_lon = cluster_points[1].mean()
#
#     # Print the center coordinates of the current cluster
#     print(f"Center coordinates of Cluster {cluster_label}: ({center_lat}, {center_lon})")
#
# # ---------------------------------------------------------------------------------------------------------------------#
#
# # Assuming 'df' is your DataFrame containing the 'Sexual Assault' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Sexual Assault' column
# data = df['Sexual Assault'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top 3-4 clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and print their points
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Print the cluster points
#     print(f"Cluster {cluster_label} points:")
#     print(cluster_points)
#     print()
#
# # Assuming 'df' is your DataFrame containing the 'Accident Prone Areas' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Accident Prone Areas' column
# data = df['Sexual Assault'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and calculate their center coordinates
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Calculate the center coordinates of the current cluster
#     center_lat = cluster_points[0].mean()
#     center_lon = cluster_points[1].mean()
#
#     # Print the center coordinates of the current cluster
#     print(f"Center coordinates of Cluster {cluster_label}: ({center_lat}, {center_lon})")
#
# # ---------------------------------------------------------------------------------------------------------------------#
#
# # Assuming 'df' is your DataFrame containing the 'Sexual Assault' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Sexual Assault' column
# data = df['Kidnapping and Abduction'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top 3-4 clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and print their points
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Print the cluster points
#     print(f"Cluster {cluster_label} points:")
#     print(cluster_points)
#     print()
#
# # Assuming 'df' is your DataFrame containing the 'Accident Prone Areas' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Accident Prone Areas' column
# data = df['Kidnapping and Abduction'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and calculate their center coordinates
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Calculate the center coordinates of the current cluster
#     center_lat = cluster_points[0].mean()
#     center_lon = cluster_points[1].mean()
#
#     # Print the center coordinates of the current cluster
#     print(f"Center coordinates of Cluster {cluster_label}: ({center_lat}, {center_lon})")
#
# # ---------------------------------------------------------------------------------------------------------------------#
#
# # Assuming 'df' is your DataFrame containing the 'Sexual Assault' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Sexual Assault' column
# data = df['Accident Prone Areas'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top 3-4 clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and print their points
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Print the cluster points
#     print(f"Cluster {cluster_label} points:")
#     print(cluster_points)
#     print()
#
# # Assuming 'df' is your DataFrame containing the 'Accident Prone Areas' column with latitude and longitude values
#
# # Extract the latitude and longitude values from the 'Accident Prone Areas' column
# data = df['Accident Prone Areas'].str.split(' ', expand=True).astype(float)
# data.dropna(axis=0, inplace=True)
#
# # Convert the latitude and longitude values to radians
# data_rad = np.radians(data)
#
# # Initialize the OPTICS clustering algorithm
# optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=10)
#
# # Fit the OPTICS algorithm to the data
# optics.fit(data_rad)
#
# # Retrieve the cluster labels
# cluster_labels = optics.labels_
#
# # Retrieve the unique cluster labels and their corresponding counts
# unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#
# # Sort the counts in descending order
# sorted_counts_indices = np.argsort(counts)[::-1]
#
# # Select the top clusters with the maximum count
# top_clusters_indices = sorted_counts_indices[:4]  # Modify the number as per your requirement
#
# # Iterate over the top clusters and calculate their center coordinates
# for cluster_index in top_clusters_indices:
#     # Get the label of the current cluster
#     cluster_label = unique_labels[cluster_index]
#
#     # Get the indices of the points belonging to the current cluster
#     cluster_indices = np.where(cluster_labels == cluster_label)[0]
#
#     # Get the latitude and longitude points of the current cluster
#     cluster_points = data.iloc[cluster_indices]
#
#     # Calculate the center coordinates of the current cluster
#     center_lat = cluster_points[0].mean()
#     center_lon = cluster_points[1].mean()
#
#     # Print the center coordinates of the current cluster
#     print(f"Center coordinates of Cluster {cluster_label}: ({center_lat}, {center_lon})")
