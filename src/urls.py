from django.urls import path

from . import tasks

urlpatterns = [
    path('', tasks.index, name='index'),


    path('task1', tasks.Task1.as_view(), name='task1'),
    path('execute_task1', tasks.execute_task1, name='ExecuteTask1'),

    path('task2', tasks.Task2.as_view(), name='task2'),
    path('execute_task1', tasks.execute_task2, name='ExecuteTask2'),

    path('task3', tasks.Task3.as_view(), name='task3'),
    path('execute_task3', tasks.execute_task3, name='ExecuteTask3'),

    path('task4', tasks.Task4.as_view(), name='task4'),
    path('execute_task4', tasks.execute_task4, name='ExecuteTask4'),

    path('task5', tasks.Task5.as_view(), name='task5'),
    path('execute_task5', tasks.execute_task5, name='ExecuteTask5'),

    path('task6', tasks.Task6.as_view(), name='task6'),
    path('execute_task6', tasks.execute_task6, name='ExecuteTask6'),
    path('get_feedback', tasks.process_feedback, name='feedback_processor'),

]
