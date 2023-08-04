from django.http import JsonResponse
from .crime_predictor import the_crime_detector
import json
from .models import CrimeCoordinate, CityList, CrimeDataYear, HeatMapData
from .db_conn import db_connection


def heatmap_data(request, city: str):
    conn, cursor = db_connection()

    # if request.method == 'GET':
    #     url = request.get_full_path()                                   # taking the complete url of the current request to implement pagination query into it
    #     query = (request.GET.get('q', '')).split(', ')                          # getting the user input from the request query

    try:
        city = city.capitalize()

        cursor.execute('SELECT data FROM "CrimeMapper_heatmapdata" WHERE city_id=(SELECT id FROM "CrimeMapper_citylist" WHERE city_name=%s)', (city, ))
        data = cursor.fetchone()

        print(data)
        print(type(data))
        print()

        if data is None:
            return JsonResponse(data={"message": "Wrong input provided"}, status=404)
        else:
            return JsonResponse(data[0], status=200)

    except Exception as e:
        return JsonResponse(data={"message": f"Exception {e} occurred"}, status=500)
