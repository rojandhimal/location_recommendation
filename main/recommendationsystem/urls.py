from django.urls import path
from .views import index, SignUpUser, LoginUser, LogoutUser, search, rate_location, location_recs, about,New_Rec

app_name = 'recommendationsystem'

urlpatterns = [
    path('', index, name='home'),
    path('register/', SignUpUser, name='register'),
    path('login/', LoginUser, name='login'),
    path('logout/', LogoutUser, name='logout'),
    path('search/', search, name='search'),
    path('rate-location/', rate_location, name='rate_location'),
    path('recommendations/', New_Rec, name='recommendations'),
    path('about/', about, name='about'),
]
