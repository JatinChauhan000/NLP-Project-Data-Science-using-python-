from django.urls import path

from . import views # here . means all

urlpatterns = [
    path('',views.base, name='base'),
    path('contact',views.contact, name='contact'),
]