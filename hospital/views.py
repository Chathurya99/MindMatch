import pickle
from turtle import pd
from urllib import request
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date
from sklearn.preprocessing import LabelEncoder
from django.contrib import messages
from django.contrib.auth.models import User , auth

import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt 
from io import BytesIO
import base64
import seaborn as sns 
import plotly.express as px 
import matplotlib
matplotlib.use('Agg')  # Set to Agg backend (non-GUI)
import matplotlib.pyplot as plt


from .models import patient , doctor , diseaseinfo , consultation ,rating_review
from chats.models import Chat,Feedback

best_model = joblib.load('main/predictor.joblib')

import pandas as pd
import numpy as np
from django.shortcuts import render

from django.shortcuts import render
import plotly.graph_objs as go
import plotly.offline as opy
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as opy
import pandas as pd
from django.shortcuts import render

import plotly.graph_objects as go
import plotly.offline as opy
from django.shortcuts import render
import plotly.graph_objects as go
import plotly.offline as opy
from django.shortcuts import render

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as opy
from django.shortcuts import render

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as opy
from django.shortcuts import render

# def dashboard(request):
#     # **Data Counts**
#     male_count = 239850
#     female_count = 52150
#     no_self_employed = 257994
#     yes_self_employed = 34006
#     no_family_history = 176832
#     yes_family_history = 115168
#     no_treatment = 144758
#     yes_treatment = 147242

#     growing_stress_counts = {'Maybe': 75000, 'Yes': 89000, 'No': 97000}
#     days_indoors_counts = {
#         '1-14 days': 63548, '31-60 days': 60705, 'Go out Every day': 58366,
#         'More than 2 months': 55916, '15-30 days': 53465
#     }
#     mood_swings_counts = {'High': 91466, 'Medium': 101064, 'Low': 99470}
#     coping_struggles_counts = {'Yes': 138036, 'No': 153964}
#     work_interest_counts = {'Yes': 85336, 'Maybe': 101185, 'No': 105479}
#     social_weakness_counts = {'Yes': 91607, 'Maybe': 103393, 'No': 97000}

#     # **Time Series Data for Line Chart**
#     time_series = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
#     growing_stress_time = [10000, 15000, 25000, 35000, 45000, 60000, 75000, 85000, 95000, 110000]

#     # **Selected Attribute for Pie Chart**
#     selected_attribute = request.GET.get('attribute', 'Gender')

#     # **Pie Chart (Dynamic)**
#     attribute_counts = {
#         'Gender': {'Male': male_count, 'Female': female_count},
#         'self_employed': {'No': no_self_employed, 'Yes': yes_self_employed},
#         'family_history': {'No': no_family_history, 'Yes': yes_family_history},
#         'treatment': {'No': no_treatment, 'Yes': yes_treatment},
#         'Growing_Stress': growing_stress_counts,
#         'Days_Indoors': days_indoors_counts,
#         'Mood_Swings': mood_swings_counts,
#         'Coping_Struggles': coping_struggles_counts,
#         'Work_Interest': work_interest_counts,
#         'Social_Weakness': social_weakness_counts
#     }
    
#     pie_data = attribute_counts[selected_attribute]
#     pie_chart = go.Figure(data=[go.Pie(labels=list(pie_data.keys()), values=list(pie_data.values()), hole=0.3)])
#     pie_chart.update_layout(title=f"Distribution of {selected_attribute}", title_font_size=18)
#     pie_chart_div = opy.plot(pie_chart, output_type='div')

#     # **Line Chart (Stress Over Time)**
#     line_chart = go.Figure()
#     line_chart.add_trace(go.Scatter(x=time_series, y=growing_stress_time, mode='lines+markers', line=dict(color='blue')))
#     line_chart.update_layout(title="Growing Stress Over Time", xaxis_title="Month", yaxis_title="Stress Count", title_font_size=18)
#     line_chart_div = opy.plot(line_chart, output_type='div')

#     # **Bar Charts**
#     stress_days_bar = px.bar(x=list(days_indoors_counts.keys()), y=list(days_indoors_counts.values()), 
#                              title="Growing Stress by Days Indoors", labels={'x': 'Days Indoors', 'y': 'Count'},
#                              color=list(days_indoors_counts.keys()))
#     stress_days_bar_div = opy.plot(stress_days_bar, output_type='div')

#     coping_bar = px.bar(x=list(coping_struggles_counts.keys()), y=list(coping_struggles_counts.values()), 
#                         title="Growing Stress by Coping Struggles", labels={'x': 'Coping Struggles', 'y': 'Count'},
#                         color=list(coping_struggles_counts.keys()))
#     coping_bar_div = opy.plot(coping_bar, output_type='div')

#     work_interest_bar = px.bar(x=list(work_interest_counts.keys()), y=list(work_interest_counts.values()), 
#                                title="Growing Stress by Work Interest", labels={'x': 'Work Interest', 'y': 'Count'},
#                                color=list(work_interest_counts.keys()))
#     work_interest_bar_div = opy.plot(work_interest_bar, output_type='div')

#     # **Column Chart**
#     social_weakness_bar = px.bar(x=list(social_weakness_counts.keys()), y=list(social_weakness_counts.values()), 
#                                  title="Growing Stress by Social Weakness", labels={'x': 'Social Weakness', 'y': 'Count'},
#                                  color=list(social_weakness_counts.keys()))
#     social_weakness_bar_div = opy.plot(social_weakness_bar, output_type='div')
#     common_keys = list(set(work_interest_counts.keys()) & set(coping_struggles_counts.keys()))
#     scatter_chart = px.scatter(
#         x=common_keys,
#         y=[work_interest_counts[k] for k in common_keys],
#         title="Scatter Plot of Work Interest vs Coping Struggles"
#     )

#     context = {
#         'pie_chart_div': pie_chart_div,
#         'line_chart_div': line_chart_div,
#         'stress_days_bar_div': stress_days_bar_div,
#         'coping_bar_div': coping_bar_div,
#         'work_interest_bar_div': work_interest_bar_div,
#         'social_weakness_bar_div': social_weakness_bar_div,
#         'scatter_chart_div': scatter_chart,
#         'attributes': list(attribute_counts.keys()),
#     }

#     return render(request, "dashboard/dashboard.html", context)
from django.shortcuts import render
import plotly.graph_objs as go
import plotly.express as px
import plotly.offline as opy

def dashboard(request):
    # **Data Counts**
    male_count = 239850
    female_count = 52150
    no_self_employed = 257994
    yes_self_employed = 34006
    no_family_history = 176832
    yes_family_history = 115168
    no_treatment = 144758
    yes_treatment = 147242

    growing_stress_counts = {'Maybe': 75000, 'Yes': 89000, 'No': 97000}
    days_indoors_counts = {
        '1-14 days': 63548, '31-60 days': 60705, 'Go out Every day': 58366,
        'More than 2 months': 55916, '15-30 days': 53465
    }
    mood_swings_counts = {'High': 91466, 'Medium': 101064, 'Low': 99470}
    coping_struggles_counts = {'Yes': 138036, 'No': 153964}
    work_interest_counts = {'Yes': 85336, 'Maybe': 101185, 'No': 105479}
    social_weakness_counts = {'Yes': 91607, 'Maybe': 103393, 'No': 97000}

    # **Time Series Data for Line Chart**
    time_series = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    growing_stress_time = [10000, 15000, 25000, 35000, 45000, 60000, 75000, 85000, 95000, 110000]

    # **Selected Attribute for Pie Chart**
    selected_attribute = request.GET.get('attribute', 'Gender')

    # **Pie Chart (Dynamic)**
    attribute_counts = {
        'Gender': {'Male': male_count, 'Female': female_count},
        'self_employed': {'No': no_self_employed, 'Yes': yes_self_employed},
        'family_history': {'No': no_family_history, 'Yes': yes_family_history},
        'treatment': {'No': no_treatment, 'Yes': yes_treatment},
        'Growing_Stress': growing_stress_counts,
        'Days_Indoors': days_indoors_counts,
        'Mood_Swings': mood_swings_counts,
        'Coping_Struggles': coping_struggles_counts,
        'Work_Interest': work_interest_counts,
        'Social_Weakness': social_weakness_counts
    }
    
    pie_data = attribute_counts[selected_attribute]
    pie_chart = go.Figure(data=[go.Pie(labels=list(pie_data.keys()), values=list(pie_data.values()), hole=0.3)])
    pie_chart.update_layout(title=f"Distribution of {selected_attribute}", title_font_size=18)
    pie_chart_div = opy.plot(pie_chart, output_type='div')

    # **Line Chart (Stress Over Time)**
    line_chart = go.Figure()
    line_chart.add_trace(go.Scatter(x=time_series, y=growing_stress_time, mode='lines+markers', line=dict(color='blue')))
    line_chart.update_layout(title="Growing Stress Over Time", xaxis_title="Month", yaxis_title="Stress Count", title_font_size=18)
    line_chart_div = opy.plot(line_chart, output_type='div')

    # **Bar Charts**
    stress_days_bar = px.bar(x=list(days_indoors_counts.keys()), y=list(days_indoors_counts.values()), 
                             title="Growing Stress by Days Indoors", labels={'x': 'Days Indoors', 'y': 'Count'},
                             color=list(days_indoors_counts.keys()))
    stress_days_bar_div = opy.plot(stress_days_bar, output_type='div')

    coping_bar = px.bar(x=list(coping_struggles_counts.keys()), y=list(coping_struggles_counts.values()), 
                        title="Growing Stress by Coping Struggles", labels={'x': 'Coping Struggles', 'y': 'Count'},
                        color=list(coping_struggles_counts.keys()))
    coping_bar_div = opy.plot(coping_bar, output_type='div')

    work_interest_bar = px.bar(x=list(work_interest_counts.keys()), y=list(work_interest_counts.values()), 
                               title="Growing Stress by Work Interest", labels={'x': 'Work Interest', 'y': 'Count'},
                               color=list(work_interest_counts.keys()))
    work_interest_bar_div = opy.plot(work_interest_bar, output_type='div')

    # **Column Chart**
    social_weakness_bar = px.bar(x=list(social_weakness_counts.keys()), y=list(social_weakness_counts.values()), 
                                 title="Growing Stress by Social Weakness", labels={'x': 'Social Weakness', 'y': 'Count'},
                                 color=list(social_weakness_counts.keys()))
    social_weakness_bar_div = opy.plot(social_weakness_bar, output_type='div')

    # **Additional Bar Chart for Mood Swings**
    mood_swings_bar = px.bar(x=list(mood_swings_counts.keys()), y=list(mood_swings_counts.values()), 
                             title="Mood Swings Distribution", labels={'x': 'Mood', 'y': 'Count'},
                             color=list(mood_swings_counts.keys()))
    mood_swings_bar_div = opy.plot(mood_swings_bar, output_type='div')

    context = {
        'pie_chart_div': pie_chart_div,
        'line_chart_div': line_chart_div,
        'stress_days_bar_div': stress_days_bar_div,
        'coping_bar_div': coping_bar_div,
        'work_interest_bar_div': work_interest_bar_div,
        'social_weakness_bar_div': social_weakness_bar_div,
        'mood_swings_bar_div': mood_swings_bar_div,
        'attributes': list(attribute_counts.keys()),
        'selected_attribute': selected_attribute,
    }

    return render(request, "dashboard/dashboard.html", context)

# Path to the uploaded Excel file
EXCEL_FILE_PATH = r"D:\6th sem\SEM6-MY\doctor_management\main\templates\dashboard\Mental Health Dataset.xlsx"

def home(request):

  if request.method == 'GET':
        
      if request.user.is_authenticated:
        return render(request,'homepage/index.html')

      else :
        return render(request,'homepage/index.html') 


def admin_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        auser = request.user
        Feedbackobj = Feedback.objects.all()

        return render(request,'admin/admin_ui/admin_ui.html' , {"auser":auser,"Feedback":Feedbackobj})

      else :
        return redirect('home')

    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')


def patient_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)

        return render(request,'patient/patient_ui/profile.html' , {"puser":puser})

      else :
        return redirect('home')

    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')


def pviewprofile(request, patientusername):

    if request.method == 'GET':

          puser = User.objects.get(username=patientusername)

          return render(request,'patient/view_profile/view_profile.html', {"puser":puser})

def checkstress(request):
    pred = 0  # Initialize prediction value

    # Initialize the feature list (assuming the model expects 35 features as in your example)
    feature_list = [0] * 35
    
    # Check if the request method is POST (when form is submitted)
    if request.method == 'POST':
        try:
            # Retrieve form data and populate feature list
            feature_list[0] = int(request.POST.get('Gender'))  # Gender
            feature_list[1] = int(request.POST.get('family_history'))  # Family History
            feature_list[2] = int(request.POST.get('treatment'))  # Treatment
            feature_list[3] = int(request.POST.get('Coping_Struggles'))  # Coping Struggles
            
            # Occupation mapping (One-hot encoding)
            occupation_map = {
                'Business': 4,
                'Corporate': 5,
                'Housewife': 6,
                'Others': 7,
                'Student': 8
            }
            occupation = request.POST.get('Occupation')
            feature_list[occupation_map.get(occupation, -1)] = 1

            # Days Indoors mapping (One-hot encoding)
            days_indoors_map = {
                '1-14 days': 9,
                '15-30 days': 10,
                '31-60 days': 11,
                'Go out Every day': 12,
                'More than 2 months': 13
            }
            days_indoors = request.POST.get('Days_Indoors')
            feature_list[days_indoors_map.get(days_indoors, -1)] = 1

            # Changes in Habits mapping (One-hot encoding)
            changes_habits_map = {'Maybe': 14, 'No': 15, 'Yes': 16}
            changes_habits = request.POST.get('Changes_Habits')
            feature_list[changes_habits_map.get(changes_habits, -1)] = 1

            # Mental Health History mapping (One-hot encoding)
            mental_health_history_map = {'Maybe': 17, 'No': 18, 'Yes': 19}
            mental_health_history = request.POST.get('Mental_Health_History')
            feature_list[mental_health_history_map.get(mental_health_history, -1)] = 1

            # Mood Swings mapping (One-hot encoding)
            mood_swings_map = {'High': 20, 'Low': 21, 'Medium': 22}
            mood_swings = request.POST.get('Mood_Swings')
            feature_list[mood_swings_map.get(mood_swings, -1)] = 1

            # Work Interest mapping (One-hot encoding)
            work_interest_map = {'Maybe': 23, 'No': 24, 'Yes': 25}
            work_interest = request.POST.get('Work_Interest')
            feature_list[work_interest_map.get(work_interest, -1)] = 1

            # Social Weakness mapping (One-hot encoding)
            social_weakness_map = {'Maybe': 26, 'No': 27, 'Yes': 28}
            social_weakness = request.POST.get('Social_Weakness')
            feature_list[social_weakness_map.get(social_weakness, -1)] = 1

            # Mental Health Interview mapping (One-hot encoding)
            mental_health_interview_map = {'Maybe': 29, 'No': 30, 'Yes': 31}
            mental_health_interview = request.POST.get('mental_health_interview')
            feature_list[mental_health_interview_map.get(mental_health_interview, -1)] = 1

            # Care Options mapping (One-hot encoding)
            care_options_map = {'No': 32, 'Not sure': 33, 'Yes': 34}
            care_options = request.POST.get('care_options')
            feature_list[care_options_map.get(care_options, -1)] = 1
            
            print("Feature List: ", feature_list)  # Debugging
            
            # Convert feature_list to a numpy array (model input)
            feature_array = np.array([feature_list])

            # Make a prediction using the model
            pred = best_model.predict(feature_array)[0]  # Get first value of the prediction

            print("Prediction: ", pred)  # Debugging
            
            # Pass prediction result to the template context
            return render(request, 'patient/checkstress/checkstress.html', {'pred': pred})

        except Exception as e:
            print(f"Error: {e}")
            return render(request, 'patient/checkstress/checkstress.html', {'error': 'There was an error processing your request. Please try again.'})
    
    # Handle case when request is not POST, perhaps showing form again
    return render(request, 'patient/checkstress/checkstress.html')
    
    
def quickpredict(request):
    pred = 0  # Initialize prediction value

    # Initialize the feature list (assuming the model expects 35 features as in your example)
    feature_list = [0] * 35
    # Initialize variables for form data (to be passed back to the template)
    form_data = {
        'Gender': '',
        'family_history': '',
        'treatment': '',
        'Coping_Struggles': '',
        'Occupation': '',
        'Days_Indoors': '',
        'Changes_Habits': '',
        'Mental_Health_History': '',
        'Mood_Swings': '',
        'Work_Interest': '',
        'Social_Weakness': '',
        'mental_health_interview': '',
        'care_options': ''
    }
    # Check if the request method is POST (when form is submitted)
    if request.method == 'POST':
        try:
            # Retrieve form data and populate feature list
            form_data['Gender'] = request.POST.get('Gender')
            feature_list[0] = int(request.POST.get('Gender'))  # Gender
            form_data['family_history'] = request.POST.get('family_history')
            feature_list[1] = int(request.POST.get('family_history'))  # Family History
            form_data['treatment'] = request.POST.get('treatment')
            feature_list[2] = int(request.POST.get('treatment'))  # Treatment
            form_data['Coping_Struggles'] = request.POST.get('Coping_Struggles')
            feature_list[3] = int(request.POST.get('Coping_Struggles'))  # Coping Struggles
            
            # Occupation mapping (One-hot encoding)
            occupation_map = {
                'Business': 4,
                'Corporate': 5,
                'Housewife': 6,
                'Others': 7,
                'Student': 8
            }
            form_data['Occupation'] = request.POST.get('Occupation')
            occupation = request.POST.get('Occupation')
            feature_list[occupation_map.get(occupation, -1)] = 1

            # Days Indoors mapping (One-hot encoding)
            days_indoors_map = {
                '1-14 days': 9,
                '15-30 days': 10,
                '31-60 days': 11,
                'Go out Every day': 12,
                'More than 2 months': 13
            }
            form_data['Days_Indoors'] = request.POST.get('Days_Indoors')
            days_indoors = request.POST.get('Days_Indoors')
            feature_list[days_indoors_map.get(days_indoors, -1)] = 1

            # Changes in Habits mapping (One-hot encoding)
            changes_habits_map = {'Maybe': 14, 'No': 15, 'Yes': 16}
            form_data['Changes_Habits'] = request.POST.get('Changes_Habits')
            changes_habits = request.POST.get('Changes_Habits')
            feature_list[changes_habits_map.get(changes_habits, -1)] = 1

            # Mental Health History mapping (One-hot encoding)
            mental_health_history_map = {'Maybe': 17, 'No': 18, 'Yes': 19}
            form_data['Mental_Health_History'] = request.POST.get('Mental_Health_History')
            mental_health_history = request.POST.get('Mental_Health_History')
            feature_list[mental_health_history_map.get(mental_health_history, -1)] = 1

            # Mood Swings mapping (One-hot encoding)
            mood_swings_map = {'High': 20, 'Low': 21, 'Medium': 22}
            form_data['Mood_Swings'] = request.POST.get('Mood_Swings')
            mood_swings = request.POST.get('Mood_Swings')
            feature_list[mood_swings_map.get(mood_swings, -1)] = 1

            # Work Interest mapping (One-hot encoding)
            work_interest_map = {'Maybe': 23, 'No': 24, 'Yes': 25}
            form_data['Work_Interest'] = request.POST.get('Work_Interest')
            work_interest = request.POST.get('Work_Interest')
            feature_list[work_interest_map.get(work_interest, -1)] = 1

            # Social Weakness mapping (One-hot encoding)
            social_weakness_map = {'Maybe': 26, 'No': 27, 'Yes': 28}
            form_data['Social_Weakness'] = request.POST.get('Social_Weakness')
            social_weakness = request.POST.get('Social_Weakness')
            feature_list[social_weakness_map.get(social_weakness, -1)] = 1

            # Mental Health Interview mapping (One-hot encoding)
            mental_health_interview_map = {'Maybe': 29, 'No': 30, 'Yes': 31}
            form_data['mental_health_interview'] = request.POST.get('mental_health_interview')
            mental_health_interview = request.POST.get('mental_health_interview')
            feature_list[mental_health_interview_map.get(mental_health_interview, -1)] = 1

            # Care Options mapping (One-hot encoding)
            care_options_map = {'No': 32, 'Not sure': 33, 'Yes': 34}
            form_data['care_options'] = request.POST.get('care_options')
            care_options = request.POST.get('care_options')
            feature_list[care_options_map.get(care_options, -1)] = 1
            
            print("Feature List: ", feature_list)  # Debugging
            
            # Convert feature_list to a numpy array (model input)
            feature_array = np.array([feature_list])

            # Make a prediction using the model
            pred = best_model.predict(feature_array)[0]  # Get first value of the prediction

            print("Prediction: ", pred)  # Debugging
            
            # Pass prediction result to the template context
            return render(request, 'patient/quickpredict/quickpredict.html', {'pred': pred,'form_data': form_data})

        except Exception as e:
            print(f"Error: {e}")
            return render(request, 'patient/quickpredict/quickpredict.html', {'error': 'There was an error processing your request. Please try again.'})
    
    # Handle case when request is not POST, perhaps showing form again
    return render(request, 'patient/quickpredict/quickpredict.html', {'form_data': form_data})

def pconsultation_history(request):

    if request.method == 'GET':

      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      patient_obj = puser.patient
        
      consultationnew = consultation.objects.filter(patient = patient_obj)
      
    
      return render(request,'patient/consultation_history/consultation_history.html',{"consultation":consultationnew})


def dconsultation_history(request):

    if request.method == 'GET':

      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      doctor_obj = duser.doctor
        
      consultationnew = consultation.objects.filter(doctor = doctor_obj)
      
    
      return render(request,'doctor/consultation_history/consultation_history.html',{"consultation":consultationnew})


def doctor_ui(request):

    if request.method == 'GET':

      doctorid = request.session['doctorusername']
      duser = User.objects.get(username=doctorid)

    
      return render(request,'doctor/doctor_ui/profile.html',{"duser":duser})


def dviewprofile(request, doctorusername):

    if request.method == 'GET':

         
         duser = User.objects.get(username=doctorusername)
         r = rating_review.objects.filter(doctor=duser.doctor)
       
         return render(request,'doctor/view_profile/view_profile.html', {"duser":duser, "rate":r} )

     
def  consult_a_doctor(request):


    if request.method == 'GET':

        
        # doctortype = request.session['doctortype']
        # print(doctortype)
        dobj = doctor.objects.all()
        #dobj = doctor.objects.filter(specialization=doctortype)


        return render(request,'patient/consult_a_doctor/consult_a_doctor.html',{"dobj":dobj})


def  make_consultation(request, doctorusername):

    if request.method == 'POST':
       

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient_obj = puser.patient
        
        
        #doctorusername = request.session['doctorusername']
        duser = User.objects.get(username=doctorusername)
        doctor_obj = duser.doctor
        request.session['doctorusername'] = doctorusername


        # diseaseinfo_id = request.session['diseaseinfo_id']
        # diseaseinfo_obj = diseaseinfo.objects.get(id=diseaseinfo_id)

        consultation_date = date.today()
        status = "active"
        
        consultation_new = consultation( patient=patient_obj, doctor=doctor_obj, consultation_date=consultation_date,status=status)
        consultation_new.save()

        request.session['consultation_id'] = consultation_new.id

        print("consultation record is saved sucessfully.............................")

         
        return redirect('consultationview',consultation_new.id)



def  consultationview(request,consultation_id):
   
    if request.method == 'GET':

   
      request.session['consultation_id'] = consultation_id
      consultation_obj = consultation.objects.get(id=consultation_id)

      return render(request,'consultation/consultation.html', {"consultation":consultation_obj })

   #  if request.method == 'POST':
   #    return render(request,'consultation/consultation.html' )


def rate_review(request,consultation_id):
   if request.method == "POST":
         
         consultation_obj = consultation.objects.get(id=consultation_id)
         patient = consultation_obj.patient
         doctor1 = consultation_obj.doctor
         rating = request.POST.get('rating')
         review = request.POST.get('review')

         rating_obj = rating_review(patient=patient,doctor=doctor1,rating=rating,review=review)
         rating_obj.save()

         rate = int(rating_obj.rating_is)
         doctor.objects.filter(pk=doctor1).update(rating=rate)
         

         return redirect('consultationview',consultation_id)


def close_consultation(request,consultation_id):
   if request.method == "POST":
         
         consultation.objects.filter(pk=consultation_id).update(status="closed")
         
         return redirect('home')

def post(request):
    if request.method == "POST":
        msg = request.POST.get('msgbox', None)

        consultation_id = request.session['consultation_id'] 
        consultation_obj = consultation.objects.get(id=consultation_id)

        c = Chat(consultation_id=consultation_obj,sender=request.user, message=msg)

        #msg = c.user.username+": "+msg

        if msg != '':            
            c.save()
            print("msg saved"+ msg )
            return JsonResponse({ 'msg': msg })
    else:
        return HttpResponse('Request must be POST.')



def chat_messages(request):
   if request.method == "GET":

         consultation_id = request.session['consultation_id'] 

         c = Chat.objects.filter(consultation_id=consultation_id)
         return render(request, 'consultation/chat_body.html', {'chat': c})

