import json
import math
import numpy as np
import pandas as pd
from geopy import Point
from geopy.distance import distance
from math import radians, cos, sin, sqrt, atan2
from pathlib import Path
import os
from joblib import Parallel, delayed
from sklearn.cluster import OPTICS
from django.contrib.gis.geos import GEOSGeometry
import psycopg2
import csv
from .db_conn import db_connection


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
    dist = radius * c

    return dist


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
            filename = 'coordinates' + str(cluster_label) + '.txt'
            with open(filename, 'w') as f:
                f.writelines(str(cluster_points))

            with open(filename, 'r') as f:
                lines = f.readlines()

                for line in lines[1:]:
                    values = line.split()
                    coordinate = tuple(float(value) for value in values[1:])
                    cluster_inner_list.append(coordinate)

                cluster_list.append(cluster_inner_list)
            os.remove(os.path.join(Path(__file__).resolve().parent, filename))

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


def data_sheet_creator(data: list, length: int, mode: str, file: str, write_header: bool):
    with open(os.path.join(Path(__file__).resolve().parent.parent, file), mode) as csv_file:
        writer = csv.writer(csv_file)

        if write_header:
            writer.writerow(['Murder and Assault', 'Kidnapping and Abduction', 'Sexual Assault', 'Theft and Burglary', 'Riots', 'Accident Prone Areas'])

        for row in range(0, length):
            row_to_add = []
            murder_assault = GEOSGeometry(data[row][1]) if data[row][1] is not None else None
            kidnapping_abduction = GEOSGeometry(data[row][2]) if data[row][2] is not None else None
            sexual_assault = GEOSGeometry(data[row][3]) if data[row][3] is not None else None
            theft_burglary = GEOSGeometry(data[row][4]) if data[row][4] is not None else None
            riots = GEOSGeometry(data[row][5]) if data[row][5] is not None else None
            accident_prone_areas = GEOSGeometry(data[row][6]) if data[row][6] is not None else None

            row_to_add.append(str(murder_assault.x) + " " + str(murder_assault.y)) if murder_assault is not None else row_to_add.append(None)
            row_to_add.append(str(kidnapping_abduction.x) + " " + str(kidnapping_abduction.y)) if kidnapping_abduction is not None else row_to_add.append(None)
            row_to_add.append(str(sexual_assault.x) + " " + str(sexual_assault.y)) if sexual_assault is not None else row_to_add.append(None)
            row_to_add.append(str(theft_burglary.x) + " " + str(theft_burglary.y)) if theft_burglary is not None else row_to_add.append(None)
            row_to_add.append(str(riots.x) + " " + str(riots.y)) if riots is not None else row_to_add.append(None)
            row_to_add.append(str(accident_prone_areas.x) + " " + str(accident_prone_areas.y)) if accident_prone_areas is not None else row_to_add.append(None)

            writer.writerow(row_to_add)


def the_crime_detector(city: str):

    conn, cursor = db_connection()

    final_json: dict = {
        "data": {
            "Murder and Assault": [],
            "Kidnapping and Abduction": [],
            "Sexual Assault": [],
            "Theft and Burglary": [],
            "Riots": [],
            "Accident Prone Areas": [],
        }
    }
    file = os.path.join(Path(__file__).resolve().parent, ('data/' + city + '.csv'))

    #data fetch
    # years = CrimeDataYear.objects.all()
    cursor.execute('SELECT * FROM "CrimeMapper_crimedatayear"')
    years = []
    for rows in cursor.fetchall():
        years.append(rows[0])

    # data1 = CrimeCoordinate.objects.all().filter(year_id=years[len(years)-1])
    cursor.execute('SELECT * FROM "CrimeMapper_crimecoordinate" WHERE year_id = %s AND city_id = (SELECT id FROM "CrimeMapper_citylist" WHERE city_name = %s)', (str(years[len(years)-1]), city))
    data1 = cursor.fetchall()
    data_sheet_creator(data=data1, length=int(len(data1) * (43/100)), mode='w', file=file, write_header=True)

    # data2 = CrimeCoordinate.objects.all().filter(year_id=years[len(years)-2])
    cursor.execute('SELECT * FROM "CrimeMapper_crimecoordinate" WHERE year_id = %s AND city_id = (SELECT id FROM "CrimeMapper_citylist" WHERE city_name = %s)', (str(years[len(years)-2]), city))
    data1 = cursor.fetchall()
    data_sheet_creator(data=data1, length=int(len(data1) * (33/100)), mode='a', file=file, write_header=False)

    # data3 = CrimeCoordinate.objects.all().filter(year_id=years[len(years)-3])
    cursor.execute('SELECT * FROM "CrimeMapper_crimecoordinate" WHERE year_id = %s AND city_id = (SELECT id FROM "CrimeMapper_citylist" WHERE city_name = %s)', (str(years[len(years)-3]), city))
    data1 = cursor.fetchall()
    data_sheet_creator(data=data1, length=int(len(data1) * (23/100)), mode='a', file=file, write_header=False)


    jp_file = os.path.join(Path(__file__).resolve().parent, "data/" + city + ".csv")
    crimes = ['Murder and Assault', 'Kidnapping and Abduction', 'Sexual Assault', 'Theft and Burglary', 'Riots',
              'Accident Prone Areas']


    df = pd.read_csv(file)
    results = Parallel(n_jobs=-1)(delayed(process_crime)(crime, df) for crime in crimes)

    for i, crime in enumerate(crimes):
        centers, distances = results[i]
        for j in range(3):
            final_json["data"][crime].append({
                "coordinates": [centers[j][0], centers[j][1]],
                "distance": distances[j]
            })
    print(final_json)

    cursor.execute('SELECT id FROM "CrimeMapper_citylist" WHERE city_name = %s;', (city, ))
    city_id = cursor.fetchall()

    cursor.execute('UPDATE "CrimeMapper_heatmapdata" SET data = %s WHERE city_id = %s;', (json.dumps(final_json), city_id[0]))
    conn.commit()

    cursor.close()
    conn.close()

    return json.dumps(final_json)


if __name__ == "__main__":
    print(the_crime_detector('Jaipur'))
