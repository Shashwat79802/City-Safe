# Generated by Django 4.2.3 on 2023-07-07 06:47

import django.contrib.gis.db.models.fields
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('CrimeMapper', '0002_alter_crimedatayear_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='crimecoordinate',
            name='accident_prone_areas',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326, verbose_name='Accident Prone Areas'),
        ),
        migrations.AlterField(
            model_name='crimecoordinate',
            name='kidnapping_abduction',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326, verbose_name='Kidnapping and Abduction'),
        ),
        migrations.AlterField(
            model_name='crimecoordinate',
            name='murder_assault',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326, verbose_name='Murder and Assault'),
        ),
        migrations.AlterField(
            model_name='crimecoordinate',
            name='riot',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326, verbose_name='Riots'),
        ),
        migrations.AlterField(
            model_name='crimecoordinate',
            name='sexual_assault',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326, verbose_name='Sexual Assault'),
        ),
        migrations.AlterField(
            model_name='crimecoordinate',
            name='theft_burglary',
            field=django.contrib.gis.db.models.fields.PointField(blank=True, null=True, srid=4326, verbose_name='Theft and Burglary'),
        ),
    ]