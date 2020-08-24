# -*- coding: utf-8 -*-

from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns=[
    path('',views.home, name='home'),
    path('facedetect/',views.facedetect,name='facedetect'),
    
    ]

#urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

#path('myapp/', views.play_audio, name='play_audio'),