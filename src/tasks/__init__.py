from .task1 import *
from .task2 import *
from .task3 import *
from .task4 import *
from .task5 import *
from .task6 import *


from django.http import HttpResponse


def index(request):
    return render(request, 'base.html', {})