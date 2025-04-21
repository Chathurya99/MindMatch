from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

from datetime import date
from django.db import models

# Create your models here.


#user = models.OneToOneField(settings.AUTH_USER_MODEL)
# models.py


class ExcelData(models.Model):
    file = models.FileField(upload_to='main\templates\dashboard\Mental Health Dataset.xlsx')  # Files will be stored in the 'uploads/' folder
    uploaded_at = models.DateTimeField(auto_now_add=True)


class patient(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    
    is_patient = models.BooleanField(default=True)
    is_doctor = models.BooleanField(default=False)

    name = models.CharField(max_length = 50)
    dob = models.DateField()
    address = models.CharField(max_length = 500)
    mobile_no = models.CharField(max_length = 15)
    gender = models.CharField(max_length = 10)

    
    @property
    def age(self):
        today = date.today()
        db = self.dob
        age = today.year - db.year
        if today.month < db.month or today.month == db.month and today.day < db.day:
            age -= 1
        return age 



class doctor(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE, primary_key=True)
    
    is_patient = models.BooleanField(default=False)
    is_doctor = models.BooleanField(default=True)

    name = models.CharField(max_length = 50)
    dob = models.DateField()
    address = models.CharField(max_length = 500)
    mobile_no = models.CharField(max_length = 15)
    gender = models.CharField(max_length = 10)

    registration_no = models.CharField(max_length = 50)
    year_of_registration = models.DateField()
    qualification = models.CharField(max_length = 100)
    State_Medical_Council = models.CharField(max_length = 100)

    specialization = models.CharField(max_length = 50)

    rating = models.IntegerField(default=0)


class diseaseinfo(models.Model):

    patient = models.ForeignKey(patient , null=True, on_delete=models.SET_NULL)

    diseasename = models.CharField(max_length = 200)
    no_of_symp = models.IntegerField()
    symptomsname = ArrayField(models.CharField(max_length=200))
    confidence = models.DecimalField(max_digits=5, decimal_places=2)
    consultdoctor = models.CharField(max_length = 200)



class consultation(models.Model):

    patient = models.ForeignKey(patient ,null=True, on_delete=models.SET_NULL)
    doctor = models.ForeignKey(doctor ,null=True, on_delete=models.SET_NULL)
    diseaseinfo = models.OneToOneField(diseaseinfo, null=True, on_delete=models.SET_NULL)
    consultation_date = models.DateField()
    status = models.CharField(max_length = 20)


class rating_review(models.Model):

    patient = models.ForeignKey(patient ,null=True, on_delete=models.SET_NULL)
    doctor = models.ForeignKey(doctor ,null=True, on_delete=models.SET_NULL)
    
    rating = models.IntegerField(default=0)
    review = models.TextField( blank=True ) 


    @property
    def rating_is(self):
        new_rating = 0
        rating_obj = rating_review.objects.filter(doctor=self.doctor)
        for i in rating_obj:
            new_rating += i.rating
       
        new_rating = new_rating/len(rating_obj)
        new_rating = int(new_rating)
        
        return new_rating
    
class MentalHealthData(models.Model):
    gender = models.CharField(max_length=10)
    self_employed = models.BooleanField()
    family_history = models.BooleanField()
    treatment = models.BooleanField()
    days_indoors = models.CharField(max_length=20)
    stress_level = models.CharField(max_length=10)
    habit_changes = models.CharField(max_length=10)
    mental_health_history = models.CharField(max_length=10)
    mood_swings = models.CharField(max_length=10)
    coping_struggles = models.BooleanField()
    work_interest = models.CharField(max_length=10)
    social_weakness = models.CharField(max_length=10)