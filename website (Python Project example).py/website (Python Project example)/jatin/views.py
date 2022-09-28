from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def base(request):
    # if request.method == "POST":
    #     name = request.POST.get('name') 
    #     email = request.POST.get('email')
    #     phone = request.POST.get('phone') 
    #     base = Base(name=name, email=email, phone=phone)
    #     base.save()
    return render(request,'base.html')
def contact(request):
    return render(request,'contact.html')