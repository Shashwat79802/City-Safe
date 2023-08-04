# Generated by Django 4.2.3 on 2023-07-10 10:00

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('CrimeMapper', '0003_alter_crimecoordinate_accident_prone_areas_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='HeatMapData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('data', models.JSONField(verbose_name='Heatmap Data')),
                ('city', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, to='CrimeMapper.citylist')),
            ],
        ),
    ]