```markdown
# KyrgyzRealty

KyrgyzRealty is a Flask-based web application for managing real estate listings in Kyrgyzstan. It supports user registration, login, apartment listing, search, price prediction, and favorites management, with multilingual support (Russian and Polish).

## Prerequisites

- Python 3.8+
- SQLite3
- Git (optional, for version control)

## Setup Instructions

1. **Clone the Repository** (if using version control):
   ```bash
   git clone <repository-url>
   cd kyrgyzrealty
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**:
   ```bash
   pip install flask flask-wtf flask-babel pandas scikit-learn itsdangerous requests
   ```

4. **Initialize the Database**:
   - Ensure `database.sql` is in the project root (`C:\Users\Admin\Desktop\ira\`).
   - The application automatically creates `real_estate.db` when run, using `database.sql`.

5. **Create Required Directories**:
   ```bash
   mkdir static\uploads
   ```
   - Ensure `static/uploads` has write permissions for image uploads.

6. **Prepare the Machine Learning Model**:
   - Run `generate_test_data.py` to create test data and train the model:
     ```bash
     python generate_test_data.py
     ```
   - This generates `model.pkl` for price predictions.

7. **Run the Application**:
   ```bash
   python app.py
   ```
   - Access the app at `http://localhost:5000`.

## Project Structure

- `app.py`: Main Flask application with routes and logic.
- `templates/`: HTML templates (e.g., `register.html`, `login.html`).
- `static/`: Static files (CSS, images).
  - `static/uploads/`: Directory for apartment images.
- `database.sql`: SQL schema for `real_estate.db`.
- `generate_test_data.py`: Script to populate the database and train the model.
- `model.pkl`: Trained machine learning model for price predictions.

## Database Schema

The `users` table in `database.sql` should include:
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created_at DATETIME NOT NULL
);
```
Other tables: `apartments`, `apartment_images`, `favorites`, `comments`.

If `real_estate.db` exists and includes an `email` column, remove it:
```bash
sqlite3 real_estate.db
ALTER TABLE users DROP COLUMN email;
```

## Testing

1. **Register**:
   - Go to `http://localhost:5000/register`.
   - Enter username (`testuser`), password (`password123`), confirm password (`password123`).
   - Verify "Регистрация успешна" message and redirect to `/login`.

2. **Login**:
   - Go to `http://localhost:5000/login`.
   - Use registered credentials (e.g., `testuser`, `password123`).
   - Verify "Вход успешен" message and redirect to `/`.

3. **Password Reset**:
   - Go to `http://localhost:5000/reset_password`.
   - Enter username (`testuser`), check flash message with reset link.

4. **Add Apartment**:
   - Log in, go to `http://localhost:5000/add_apartment`.
   - Submit valid data (e.g., city: `Бишкек`, rooms: `2`, area: `50`, price: `50000`).
   - Check logs for "All apartment predictions updated".

5. **Language Switching**:
   - Test `/change_language/ru` and `/change_language/pl`.

## Troubleshooting

- **Missing `styles.css`**: Create an empty `static/styles.css` if not present.
- **Database Errors**: Verify `database.sql` matches the schema above. Share `database.sql` if issues persist.
- **Model Issues**: Ensure `model.pkl` exists. Rerun `generate_test_data.py` if missing.
- **Permission Errors**: Grant write permissions to `static/uploads`:
  ```bash
  chmod -R 777 static/uploads  # Linux/Mac
  ```
  On Windows, ensure folder permissions allow writing.

## Production Notes

- Use a WSGI server (e.g., Gunicorn):
  ```bash
  pip install gunicorn
  gunicorn -w 4 -b 0.0.0.0:8000 app:app
  ```
- Secure `app.secret_key` in `app.py` (replace `'your_secret_key_change_me'`).
- Use Celery for background tasks (e.g., `update_all_predictions`).
- Add email functionality with a proper SMTP setup (e.g., Gmail app-specific password) if needed.

## Support

For issues, share:
- Error traceback.
- Contents of `database.sql`, `styles.css`, or other templates (e.g., `add_apartment.html`).
- Logs from `app.py` (set `logging.basicConfig(level=logging.DEBUG)`).
```