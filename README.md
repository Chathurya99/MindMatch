# MindMatch: Stress Prediction & Counselor Matching Platform

MindMatch is a machine learning-powered web application that predicts user stress levels based on lifestyle, mental health, and demographic data. If stress is detected, the platform allows users to consult with doctors through a secure and role-based system.

## Key Features

- Stress level prediction using XGBoost (best-performing model)
- Role-based login for **Patients**, **Doctors**, and **Admins**
- Interactive dashboard with charts and visualizations
- Secure messaging and doctor-patient consultation
- Admin panel to manage users and platform settings

## Tech Stack

- **Backend**: Django, Python, Machine Learning (XGBoost)
- **Frontend**: HTML, CSS, Bootstrap, JavaScript, jQuery
- **Database**: PostgreSQL
- **Visualization**: Matplotlib, Seaborn, Plotly (optional)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chathurya99/MindMatch
cd mindmatch

2. Create & activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate

python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
