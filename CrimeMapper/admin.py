from django.contrib import admin
from .models import CityList, CrimeCoordinate, CrimeDataYear


@admin.register(CityList)
class CityListAdmin(admin.ModelAdmin):
    list_display = ('id', 'city_name', 'city_state_name', 'city_boundary_coordinates')
    ordering = ['id']


@admin.register(CrimeCoordinate)
class CrimeCoordinateAdmin(admin.ModelAdmin):
    list_display = ('id', 'city', 'year', 'murder_assault', 'kidnapping_abduction', 'sexual_assault', 'riot', 'theft_burglary', 'accident_prone_areas')
    list_per_page = 500
    ordering = ['id']


@admin.register(CrimeDataYear)
class CrimeDataYearAdmin(admin.ModelAdmin):
    list_display = ('id', 'year')
    ordering = ['id']
