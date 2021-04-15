from django.db import models
from django.contrib.auth import get_user_model

import jsonfield
import json
import numpy as np

User = get_user_model()


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    array = jsonfield.JSONField()
    arrayratedlocationsindxs = jsonfield.JSONField()
    name = models.CharField(max_length=1000)
    lastrecs = jsonfield.JSONField()

    def __str__(self):
        return self.user.username

    def save(self, *args, **kwargs):
        create = kwargs.pop('create', None)
        recsvec = kwargs.pop('recsvec', None)
        print('create:', create)
        if create == True:
            super(UserProfile, self).save(*args, **kwargs)

        elif recsvec is not None and len(recsvec) != 0:
            # self.lastrecs = json.dumps(recsvec.tolist())
            self.lastrecs = json.dumps(recsvec)
            super(UserProfile, self).save(*args, **kwargs)
        else:
            nlocations = LocationData.objects.count()
            array = np.zeros(nlocations)
            ratedlocations = self.ratedlocations.all()
            self.arrayratedlocationsindxs = json.dumps([loc.locationindx for loc in ratedlocations])
            for loc in ratedlocations:
                array[loc.locationindx] = loc.value
            self.array = json.dumps(array.tolist())
            super(UserProfile, self).save(*args, **kwargs)


class LocationRated(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='ratedlocations')
    location = models.ForeignKey('LocationData', on_delete=models.CASCADE)
    locationindx = models.IntegerField(default=-1)
    value = models.IntegerField()

    def __str__(self):
        return self.location.title


class LocationData(models.Model):
    title = models.CharField(max_length=100)
    image = models.ImageField(upload_to='location/', null=True, blank=True)
    address = models.CharField(max_length=150)
    array = jsonfield.JSONField()
    ndim = models.IntegerField(default=300)
    description = models.TextField()