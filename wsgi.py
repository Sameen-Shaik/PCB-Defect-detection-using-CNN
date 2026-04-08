"""Production WSGI entry point for the Flask application.

This file is used by production WSGI servers like Gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

Or with uWSGI:
    uwsgi --http 0.0.0.0:5000 --wsgi-file wsgi.py --callable app
"""
import os
from app import create_app

# Use production config by default, override with FLASK_ENV
config_name = os.environ.get('FLASK_ENV', 'production')
app = create_app(config_name)
