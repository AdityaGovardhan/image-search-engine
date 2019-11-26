from django.db import models

# Create your models here.

classifiers = (
        ('Support Vector Machine', 'Support Vector Machine'),
        ('Decision Tree Classifier', 'Decision Tree Classifier'),
        ('Personalized Page Rank', 'Personalized Page Rank'),
    )

task_6_relevance_feedbacks = (
        ('Support Vector Machine', 'Support Vector Machine'),
        ('Decision Tree Classifier', 'Decision Tree Classifier'),
        ('Personalized Page Rank', 'Personalized Page Rank'),
        ('Probabilistic', 'Probabilistic'),
    )


class Task1Model(models.Model):
    number_of_latent_semantics = models.CharField(max_length=2)


class Task2Model(models.Model):
    number_of_clusters = models.CharField(max_length=2)


class Task3Model(models.Model):
    most_similar_images = models.CharField(max_length=2, verbose_name = "Number of Most Similar Images")


class Task4Model(models.Model):
    classifier = models.CharField(max_length=6, choices=classifiers, default='Support Vector Machine')


class Task5Model(models.Model):
    number_of_layers = models.CharField(max_length=2)
    number_of_hashes_per_layer = models.CharField(max_length=2)
    set_of_vectors = models.CharField(max_length=2)
    query_image = models.CharField(max_length=100)
    most_similar_images = models.CharField(max_length=2, verbose_name = "Number of Most Similar Images")



class Task6Model(models.Model):
    relevance_feedback = models.CharField(max_length=6, choices=task_6_relevance_feedbacks, default='Probabilistic')
    # pass


