from django.db import models
# from django.contrib.gis.db.models import PointField, PolygonField
from django.contrib.gis.db.models.fields import PolygonField, PointField


class CityList(models.Model):
    city_name = models.CharField("City Name", max_length=30, blank=False, null=False, help_text="Name of City")
    city_state_name = models.CharField("City State", max_length=30, blank=False, null=False, help_text="Name of State")
    city_boundary_coordinates = PolygonField('Boundary Coordinates', geography=False, dim=2)
    # city_boundary_coordinates = models.TextField("Boundary Coordinates", max_length=150, blank=False, null=False, help_text="Contains the boundary coordinates of the city")

    class Meta:
        app_label = 'CrimeMapper'

    def __str__(self):
        return f"{self.city_name}, {self.city_state_name}"


class CrimeDataYear(models.Model):
    year = models.CharField(max_length=4)

    # city = models.ForeignKey(CityList, on_delete=models.PROTECT)

    def __str__(self):
        return self.year


class CrimeCoordinate(models.Model):
    year = models.ForeignKey(CrimeDataYear, on_delete=models.PROTECT)
    city = models.ForeignKey(CityList, on_delete=models.PROTECT)
    murder_assault = PointField('Murder and Assault', 2, null=True, blank=True)
    kidnapping_abduction = PointField("Kidnapping and Abduction", 2, null=True, blank=True)
    sexual_assault = PointField("Sexual Assault", 2, null=True, blank=True)
    riot = PointField("Riots", 2, null=True, blank=True)
    theft_burglary = PointField("Theft and Burglary", 2, null=True, blank=True)
    accident_prone_areas = PointField("Accident Prone Areas", 2, null=True, blank=True)

    def __str__(self):
        return f"{self.murder_assault}, {self.kidnapping_abduction}, {self.sexual_assault}, {self.riot}, {self.theft_burglary}, {self.accident_prone_areas}"
    

class HeatMapData(models.Model):
    city = models.ForeignKey(CityList, on_delete=models.PROTECT)
    data = models.JSONField("Heatmap Data")

    def __str__(self):
        return f"{self.city.city_name}, {self.data}"
