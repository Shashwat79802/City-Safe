from django.urls import path
from . import views

urlpatterns = [
    path('place/<str:city>', views.heatmap_data, name='heatmap_data'),
]
