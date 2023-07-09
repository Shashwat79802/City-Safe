from django.urls import path
from . import views


urlpatterns = [
    path('place/', views.heatmap_data, name='heatmap_data')
]