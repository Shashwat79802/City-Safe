from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from .crime_predictor import the_crime_detector
import json


def heatmap_data(request):
    if request.method == 'GET':
        url = request.get_full_path()                                   # taking the complete url of the current request to implement pagination query into it
        query = (request.GET.get('q', '')).split(', ')                          # getting the user input from the request query
        json_data = the_crime_detector()
        return JsonResponse(json_data)

