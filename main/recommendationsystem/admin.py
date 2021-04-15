from django.contrib import admin
from recommendationsystem.models import LocationData, LocationRated, UserProfile


class LocationsAdmin(admin.ModelAdmin):
    list_display = ['title', 'description']


admin.site.register(UserProfile)
admin.site.register(LocationData, LocationsAdmin)
admin.site.register(LocationRated)