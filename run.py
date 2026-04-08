"""Development entry point for the Flask application."""
from app import create_app

# Create application instance
app = create_app()

if __name__ == "__main__":
    # Development server
    # Use debug=True for auto-reload during development
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
