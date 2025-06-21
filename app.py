from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, IntegerField, FloatField, SelectField, SelectMultipleField, FileField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from flask_mail import Mail, Message
from flask_babel import Babel, _
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import uuid
import logging
from datetime import datetime
import itsdangerous
from functools import wraps
import requests
import threading

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_change_me'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB
app.config['BABEL_DEFAULT_LOCALE'] = 'ru'
app.config['BABEL_DEFAULT_TIMEZONE'] = 'Asia/Bishkek'

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_app_password'  # Replace with your app-specific password
mail = Mail(app)

# CSRF Protection
csrf = CSRFProtect(app)

# Flask-Babel
babel = Babel(app)

def get_locale():
    return session.get('lang', 'ru')

babel.init_app(app, locale_selector=get_locale)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Database connection
def get_db():
    conn = sqlite3.connect('real_estate.db')
    conn.row_factory = sqlite3.Row
    return conn

# Forms
class SearchForm(FlaskForm):
    city = SelectField(_('Город'), choices=[('', _('Выберите город')), ('Бишкек', 'Бишкек'), ('Ош', 'Ош'), ('Джалал-Абад', 'Джалал-Абад')])
    address = StringField(_('Адрес'))
    min_price = IntegerField(_('Минимальная цена'))
    max_price = IntegerField(_('Максимальная цена'))
    min_area = FloatField(_('Минимальная площадь'))
    max_area = FloatField(_('Максимальная площадь'))
    rooms = SelectField(_('Комнаты'), choices=[('', _('Любое')), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')])
    amenities = SelectMultipleField(_('Удобства'), choices=[
        ('Парковка', _('Парковка')), ('Балкон', _('Балкон')), ('Лифт', _('Лифт')),
        ('Мебель', _('Мебель')), ('Кондиционер', _('Кондиционер')), ('Охрана', _('Охрана'))
    ])
    submit = SubmitField(_('Найти'))

class ResetPasswordForm(FlaskForm):
    email = StringField(_('Email'), validators=[DataRequired(), Email()])
    submit = SubmitField(_('Отправить ссылку'))

class ResetPasswordConfirmForm(FlaskForm):
    new_password = PasswordField(_('Новый пароль'), validators=[DataRequired()])
    confirm_password = PasswordField(_('Подтвердите пароль'), validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField(_('Сохранить'))

class CommentForm(FlaskForm):
    content = StringField(_('Комментарий'), validators=[DataRequired()])
    submit = SubmitField(_('Отправить'))

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def format_price(value):
    if value is None or value == '':
        logger.warning("format_price received None or empty value")
        return "0"
    try:
        return "{:,.0f}".format(float(value)).replace(",", " ")
    except (ValueError, TypeError) as e:
        logger.error(f"format_price error: {e}, value: {value}")
        return str(value)

app.jinja_env.filters['format_price'] = format_price

def geocode_address(address, city):
    try:
        query = f"{address}, {city}, Kyrgyzstan"
        response = requests.get('https://nominatim.openstreetmap.org/search', params={
            'q': query,
            'format': 'json',
            'limit': 1
        }, headers={'User-Agent': 'KyrgyzRealty/1.0'})
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        return 42.8746, 74.6122  # Бишкек по умолчанию
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return 42.8746, 74.6122

def train_model():
    try:
        with get_db() as conn:
            df = pd.read_sql_query("SELECT city, rooms, floor, area, price FROM apartments WHERE status = 'active'", conn)
        if df.empty:
            logger.warning("No data to train model")
            return None

        df = df.dropna()
        if df.empty:
            logger.warning("No valid data after cleaning")
            return None

        df = df[df['price'] > 0]
        df = df[df['area'] > 10]
        df = df[df['rooms'].isin([1, 2, 3, 4])]
        df = df[df['floor'] > 0]

        le = LabelEncoder()
        df['city_encoded'] = le.fit_transform(df['city'])
        X = df[['city_encoded', 'rooms', 'floor', 'area']]
        y = df['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        with open('model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'label_encoder': le}, f)
        logger.debug("Model trained and saved")
        return {'model': model, 'label_encoder': le}
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        return None

def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        logger.debug("Model loaded successfully")
        return model_data
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return None

def update_all_predictions():
    try:
        model_data = load_model()
        if model_data is None:
            logger.warning("Cannot update predictions: model not loaded")
            return

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, city, rooms, floor, area FROM apartments WHERE status = 'active'")
            apartments = [dict(row) for row in cur.fetchall()]
            if not apartments:
                logger.warning("No active apartments to update predictions")
                return

            df = pd.DataFrame(apartments)
            df['city_encoded'] = model_data['label_encoder'].transform(df['city'])
            X = df[['city_encoded', 'rooms', 'floor', 'area']]
            predicted_prices = model_data['model'].predict(X)

            for apartment, predicted_price in zip(apartments, predicted_prices):
                cur.execute("UPDATE apartments SET price = ? WHERE id = ?", (round(predicted_price, 2), apartment['id']))
            conn.commit()
            logger.debug("All apartment predictions updated")
    except Exception as e:
        logger.error(f"Error updating predictions: {e}")

model_data = load_model()
if model_data is None:
    model_data = train_model()

def generate_token(email):
    serializer = itsdangerous.URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(email, salt='password-reset-salt')

def confirm_token(token, expiration=3600):
    serializer = itsdangerous.URLSafeTimedSerializer(app.secret_key)
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except:
        return False
    return email

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash(_('Пожалуйста, войдите в систему'), 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a WHERE status = 'active' ORDER BY price DESC LIMIT 6")
        featured_apartments = [dict(row) for row in cur.fetchall()]
        if featured_apartments:
            df = pd.DataFrame(featured_apartments)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['description'] = df['description'].fillna(df['city'].apply(lambda x: f"Уютная квартира в центре {x}."))
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            featured_apartments = df.to_dict('records')
    return render_template('index.html', featured_apartments=featured_apartments)

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    page = int(request.args.get('page', 1))
    per_page = 9
    sort_by = request.args.get('sort_by', 'price_asc')

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT city FROM apartments WHERE status = 'active'")
        cities = [row['city'] for row in cur.fetchall()]
        form.city.choices = [('', _('Выберите город'))] + [(city, city) for city in cities]

        if 'user_id' in session:
            cur.execute("SELECT apartment_id FROM favorites WHERE user_id = ?", (session['user_id'],))
            favorites = [row['apartment_id'] for row in cur.fetchall()]
        else:
            favorites = []

        if form.validate_on_submit():
            city = form.city.data
            address = form.address.data
            min_price = form.min_price.data or 0
            max_price = form.max_price.data or float('inf')
            min_area = form.min_area.data or 0
            max_area = form.max_area.data or float('inf')
            rooms = form.rooms.data
            amenities = form.amenities.data
            sort_by = request.form.get('sort_by', 'price_asc')

            query_conditions = ["status = 'active'"]
            params = []

            if city:
                query_conditions.append("city = ?")
                params.append(city)
            if address:
                query_conditions.append("address LIKE ?")
                params.append(f'%{address}%')
            if min_price:
                query_conditions.append("price >= ?")
                params.append(float(min_price))
            if max_price != float('inf'):
                query_conditions.append("price <= ?")
                params.append(float(max_price))
            if min_area:
                query_conditions.append("area >= ?")
                params.append(float(min_area))
            if max_area != float('inf'):
                query_conditions.append("area <= ?")
                params.append(float(max_area))
            if rooms:
                query_conditions.append("rooms = ?")
                params.append(int(rooms))
            if amenities:
                for amenity in amenities:
                    query_conditions.append("amenities LIKE ?")
                    params.append(f'%{amenity}%')

            query = "SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a"
            if query_conditions:
                query += " WHERE " + " AND ".join(query_conditions)

            if sort_by == 'price_asc':
                query += " ORDER BY price ASC"
            elif sort_by == 'price_desc':
                query += " ORDER BY price DESC"
            elif sort_by == 'area_asc':
                query += " ORDER BY area ASC"
            elif sort_by == 'area_desc':
                query += " ORDER BY area DESC"

            count_query = f"SELECT COUNT(*) FROM apartments WHERE {' AND '.join(query_conditions)}" if query_conditions else "SELECT COUNT(*) FROM apartments"
            cur.execute(count_query, params)
            total = cur.fetchone()[0]
            total_pages = (total + per_page - 1) // per_page

            query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
            cur.execute(query, params)
            apartments = [dict(row) for row in cur.fetchall()]
            if apartments:
                df = pd.DataFrame(apartments)
                df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
                df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
                apartments = df.to_dict('records')

            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(apartments)
            return render_template('search.html', form=form, apartments=apartments, favorites=favorites, page=page, total_pages=total_pages)

    return render_template('search.html', form=form, apartments=[], favorites=[], page=page, total_pages=1)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model_data
    if model_data is None:
        model_data = load_model()
        if model_data is None:
            return jsonify({'error': _('Модель не загружена.')}), 500

    if request.method == 'POST':
        try:
            city = request.form.get('city')
            rooms = request.form.get('rooms')
            floor = request.form.get('floor')
            area = request.form.get('area')

            if not city or city not in ['Бишкек', 'Ош', 'Джалал-Абад']:
                return jsonify({'error': _('Неверный город')})
            if not rooms or not rooms.isdigit() or int(rooms) not in [1, 2, 3, 4]:
                return jsonify({'error': _('Неверное количество комнат')})
            if not floor or not floor.isdigit() or int(floor) < 1 or int(floor) > 20:
                return jsonify({'error': _('Неверный этаж')})
            if not area or float(area) < 10:
                return jsonify({'error': _('Площадь должна быть больше 10 м²')})

            input_data = pd.DataFrame({
                'city': [city],
                'rooms': [int(rooms)],
                'floor': [int(floor)],
                'area': [float(area)]
            })

            le = model_data['label_encoder']
            input_data['city_encoded'] = le.transform(input_data['city'])
            X = input_data[['city_encoded', 'rooms', 'floor', 'area']]
            predicted_price = model_data['model'].predict(X)[0]

            return jsonify({
                'city': city,
                'rooms': int(rooms),
                'floor': int(floor),
                'area': float(area),
                'price': round(predicted_price, 2)
            })
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 400

    return render_template('predict.html')

@app.route('/add_apartment', methods=['GET', 'POST'])
@login_required
def add_apartment():
    if request.method == 'POST':
        try:
            city = request.form.get('city')
            rooms = request.form.get('rooms')
            floor = request.form.get('floor')
            area = request.form.get('area')
            price = request.form.get('price')
            total_floors = request.form.get('total_floors')
            description = request.form.get('description')
            amenities = request.form.getlist('amenities')
            address = request.form.get('address')
            files = request.files.getlist('images')

            if not city or city not in ['Бишкек', 'Ош', 'Джалал-Абад']:
                flash(_('Неверный город'), 'danger')
                return redirect(url_for('add_apartment'))
            if not rooms or not rooms.isdigit() or int(rooms) not in [1, 2, 3, 4]:
                flash(_('Неверное количество комнат'), 'danger')
                return redirect(url_for('add_apartment'))
            if not floor or not floor.isdigit() or int(floor) < 1 or int(floor) > 20:
                flash(_('Неверный этаж'), 'danger')
                return redirect(url_for('add_apartment'))
            if not area or float(area) < 10:
                flash(_('Площадь должна быть больше 10 м²'), 'danger')
                return redirect(url_for('add_apartment'))
            if not price or float(price) < 0:
                flash(_('Неверная цена'), 'danger')
                return redirect(url_for('add_apartment'))
            if not total_floors or int(total_floors) < int(floor):
                flash(_('Неверное количество этажей'), 'danger')
                return redirect(url_for('add_apartment'))
            if not address:
                flash(_('Адрес обязателен'), 'danger')
                return redirect(url_for('add_apartment'))
            if len(files) > 10:
                flash(_('Максимум 10 изображений'), 'danger')
                return redirect(url_for('add_apartment'))

            image_paths = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    logger.debug(f"Изображение сохранено: {file_path}")
                    image_paths.append(f"/static/uploads/{filename}")
                elif file:
                    flash(f'Недопустимый формат: {file.filename}', 'danger')

            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO apartments (city, rooms, floor, area, price, total_floors, description, amenities, address, status, created_at, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """, (city, int(rooms), int(floor), float(area), float(price), int(total_floors), description, ','.join(amenities), address, datetime.now(), session['user_id']))
                apartment_id = cur.lastrowid

                for path in image_paths:
                    cur.execute("INSERT INTO apartment_images (apartment_id, image_path) VALUES (?, ?)", (apartment_id, path))

                conn.commit()
                global model_data
                model_data = train_model()
                threading.Thread(target=update_all_predictions).start()
                flash(_('Квартира добавлена'), 'success')
                msg = Message(_('Новое объявление'), sender=app.config['MAIL_USERNAME'], recipients=[session['email']])
                msg.body = _(f'Ваше объявление "{city}, {address}" успешно добавлено.')
                mail.send(msg)
                return redirect(url_for('my_apartments'))
        except Exception as e:
            logger.error(f"Ошибка добавления квартиры: {e}")
            flash(_(f'Ошибка: {str(e)}'), 'danger')
            return redirect(url_for('add_apartment'))

    return render_template('add_apartment.html')

@app.route('/edit_apartment/<int:apartment_id>', methods=['GET', 'POST'])
@login_required
def edit_apartment(apartment_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM apartments WHERE id = ? AND user_id = ?", (apartment_id, session['user_id']))
        apartment = cur.fetchone()
        if not apartment:
            flash(_('Квартира не найдена'), 'danger')
            return redirect(url_for('my_apartments'))

        cur.execute("SELECT id, image_path FROM apartment_images WHERE apartment_id = ?", (apartment_id,))
        images = [dict(row) for row in cur.fetchall()]

        if request.method == 'POST':
            try:
                city = request.form.get('city')
                rooms = request.form.get('rooms')
                floor = request.form.get('floor')
                area = request.form.get('area')
                price = request.form.get('price')
                total_floors = request.form.get('total_floors')
                description = request.form.get('description')
                amenities = request.form.getlist('amenities')
                address = request.form.get('address')
                files = request.files.getlist('images')
                delete_images = request.form.getlist('delete_images')

                if not city or city not in ['Бишкек', 'Ош', 'Джалал-Абад']:
                    flash(_('Неверный город'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if not rooms or not rooms.isdigit() or int(rooms) not in [1, 2, 3, 4]:
                    flash(_('Неверное количество комнат'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if not floor or not floor.isdigit() or int(floor) < 1 or int(floor) > 20:
                    flash(_('Неверный этаж'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if not area or float(area) < 10:
                    flash(_('Площадь должна быть больше 10 м²'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if not price or float(price) < 0:
                    flash(_('Неверная цена'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if not total_floors or int(total_floors) < int(floor):
                    flash(_('Неверное количество этажей'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if not address:
                    flash(_('Адрес обязателен'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                if len(files) + len(images) - len(delete_images) > 10:
                    flash(_('Максимум 10 изображений'), 'danger')
                    return redirect(url_for('edit_apartment', apartment_id=apartment_id))

                image_paths = [img['image_path'] for img in images if str(img['id']) not in delete_images]
                for file in files:
                    if file and allowed_file(file.filename):
                        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        logger.debug(f"Изображение сохранено: {file_path}")
                        image_paths.append(f"/static/uploads/{filename}")
                    elif file:
                        flash(f'Недопустимый формат: {file.filename}', 'danger')

                cur.execute("""
                    UPDATE apartments SET city = ?, rooms = ?, floor = ?, area = ?, price = ?, total_floors = ?, description = ?, amenities = ?, address = ?
                    WHERE id = ? AND user_id = ?
                """, (city, int(rooms), int(floor), float(area), float(price), int(total_floors), description, ','.join(amenities), address, apartment_id, session['user_id']))

                for image_id in delete_images:
                    cur.execute("SELECT image_path FROM apartment_images WHERE id = ? AND apartment_id = ?", (image_id, apartment_id))
                    image = cur.fetchone()
                    if image:
                        try:
                            os.remove(os.path.join(app.root_path, image['image_path'].lstrip('/')))
                            logger.debug(f"Изображение удалено: {image['image_path']}")
                        except OSError as e:
                            logger.warning(f"Не удалось удалить файл {image['image_path']}: {e}")
                        cur.execute("DELETE FROM apartment_images WHERE id = ? AND apartment_id = ?", (image_id, apartment_id))

                for path in image_paths:
                    if not cur.execute("SELECT id FROM apartment_images WHERE apartment_id = ? AND image_path = ?", (apartment_id, path)).fetchone():
                        cur.execute("INSERT INTO apartment_images (apartment_id, image_path) VALUES (?, ?)", (apartment_id, path))

                conn.commit()
                global model_data
                model_data = train_model()
                threading.Thread(target=update_all_predictions).start()
                flash(_('Квартира обновлена'), 'success')
                return redirect(url_for('my_apartments'))
            except Exception as e:
                logger.error(f"Ошибка обновления квартиры: {e}")
                flash(_(f'Ошибка: {str(e)}'), 'danger')
                return redirect(url_for('edit_apartment', apartment_id=apartment_id))

        apartment = dict(apartment)
        apartment['amenities'] = apartment['amenities'].split(',') if apartment['amenities'] else []
        return render_template('edit_apartment.html', apartment=apartment, images=images)

@app.route('/delete_apartment/<int:apartment_id>', methods=['POST'])
@login_required
def delete_apartment(apartment_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM apartments WHERE id = ?", (apartment_id,))
        apartment = cur.fetchone()
        if not apartment or (apartment['user_id'] != session['user_id'] and session['username'] != 'admin'):
            flash(_('Вы не имеете прав для удаления этого объявления'), 'danger')
            return redirect(url_for('my_apartments'))

        cur.execute("UPDATE apartments SET status = 'inactive' WHERE id = ?", (apartment_id,))
        conn.commit()
        global model_data
        model_data = train_model()
        threading.Thread(target=update_all_predictions).start()
        flash(_('Квартира удалена'), 'success')
        msg = Message(_('Объявление удалено'), sender=app.config['MAIL_USERNAME'], recipients=[session['email']])
        msg.body = _(f'Ваше объявление с ID {apartment_id} было удалено.')
        mail.send(msg)
        return redirect(url_for('my_apartments'))

@app.route('/my_apartments')
@login_required
def my_apartments():
    page = int(request.args.get('page', 1))
    per_page = 9

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM apartments WHERE user_id = ? AND status = 'active'", (session['user_id'],))
        total = cur.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page

        cur.execute("SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a WHERE user_id = ? AND status = 'active' LIMIT ? OFFSET ?", 
                    (session['user_id'], per_page, (page - 1) * per_page))
        apartments = [dict(row) for row in cur.fetchall()]
        if apartments:
            df = pd.DataFrame(apartments)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            apartments = df.to_dict('records')

    return render_template('my_apartments.html', apartments=apartments, page=page, total_pages=total_pages)

@app.route('/favorites')
@login_required
def favorites():
    page = int(request.args.get('page', 1))
    per_page = 9

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM favorites f JOIN apartments a ON f.apartment_id = a.id WHERE f.user_id = ? AND a.status = 'active'", (session['user_id'],))
        total = cur.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page

        cur.execute("""
            SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image 
            FROM apartments a JOIN favorites f ON a.id = f.apartment_id 
            WHERE f.user_id = ? AND a.status = 'active' LIMIT ? OFFSET ?
        """, (session['user_id'], per_page, (page - 1) * per_page))
        favorites = [dict(row) for row in cur.fetchall()]
        if favorites:
            df = pd.DataFrame(favorites)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            favorites = df.to_dict('records')

    return render_template('favorites.html', favorites=favorites, page=page, total_pages=total_pages)

@app.route('/toggle_favorite/<int:apartment_id>', methods=['POST'])
@login_required
def toggle_favorite(apartment_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM apartments WHERE id = ? AND status = 'active'", (apartment_id,))
        if not cur.fetchone():
            flash(_('Объявление не найдено'), 'danger')
            return redirect(url_for('search'))

        cur.execute("SELECT * FROM favorites WHERE user_id = ? AND apartment_id = ?", (session['user_id'], apartment_id))
        if cur.fetchone():
            cur.execute("DELETE FROM favorites WHERE user_id = ? AND apartment_id = ?", (session['user_id'], apartment_id))
            flash(_('Удалено из избранного'), 'success')
        else:
            cur.execute("INSERT INTO favorites (user_id, apartment_id) VALUES (?, ?)", (session['user_id'], apartment_id))
            flash(_('Добавлено в избранное'), 'success')
        conn.commit()
    return redirect(request.referrer or url_for('search'))

@app.route('/details/<int:apartment_id>', methods=['GET', 'POST'])
def details(apartment_id):
    form = CommentForm()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM apartments WHERE id = ? AND status = 'active'", (apartment_id,))
        apartment = cur.fetchone()
        if not apartment:
            flash(_('Квартира не найдена'), 'danger')
            return redirect(url_for('index'))

        cur.execute("SELECT image_path FROM apartment_images WHERE apartment_id = ?", (apartment_id,))
        images = [row['image_path'] for row in cur.fetchall()] or ['https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80']

        cur.execute("SELECT c.*, u.username FROM comments c JOIN users u ON c.user_id = u.id WHERE c.apartment_id = ? ORDER BY c.created_at DESC", (apartment_id,))
        comments = [dict(row) for row in cur.fetchall()]

        if 'user_id' in session:
            cur.execute("SELECT * FROM favorites WHERE user_id = ? AND apartment_id = ?", (session['user_id'], apartment_id))
            is_favorite = bool(cur.fetchone())
        else:
            is_favorite = False

        if form.validate_on_submit() and 'user_id' in session:
            content = form.content.data
            cur.execute("INSERT INTO comments (apartment_id, user_id, content, created_at) VALUES (?, ?, ?, ?)",
                        (apartment_id, session['user_id'], content, datetime.now()))
            conn.commit()
            flash(_('Комментарий добавлен'), 'success')
            return redirect(url_for('details', apartment_id=apartment_id))

        apartment = dict(apartment)
        apartment['images'] = images
        apartment['amenities'] = apartment['amenities'].split(',') if apartment['amenities'] else []
        apartment['coordinates'] = geocode_address(apartment['address'], apartment['city'])

    return render_template('details.html', apartment=apartment, comments=comments, is_favorite=is_favorite, form=form)

@app.route('/contact', methods=['POST'])
def contact():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        apartment_id = request.form.get('apartment_id')

        if not name or not email or not message:
            return jsonify({'error': _('Все поля обязательны.')}), 400

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT city, address FROM apartments WHERE id = ?", (apartment_id,))
            apt = cur.fetchone()

        subject = f"Запрос по квартире #{apartment_id}"
        body = f"Имя: {name}\nEmail: {email}\nКвартира: {apt['city']}, {apt['address']}\nСообщение: {message}"

        msg = Message(subject, sender=app.config['MAIL_USERNAME'], recipients=['info@kyrgyzrealty.kg'])
        msg.body = body
        mail.send(msg)

        return jsonify({'success': _('Сообщение отправлено.')})
    except Exception as e:
        logger.error(f"Contact error: {e}")
        return jsonify({'error': _('Ошибка отправки сообщения.')}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cur.fetchone()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['email'] = user.get('email', 'no-email@kyrgyzrealty.kg')
                flash(_('Вход успешен'), 'success')
                return redirect(url_for('index'))
            else:
                flash(_('Неверное имя пользователя или пароль'), 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash(_('Пароли не совпадают'), 'danger')
            return redirect(url_for('register'))

        with get_db() as conn:
            cur = conn.cursor()
            try:
                cur.execute("INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
                            (username, email, generate_password_hash(password), datetime.now()))
                conn.commit()

                msg = Message(_('Добро пожаловать в KyrgyzRealty!'), sender=app.config['MAIL_USERNAME'], recipients=[email])
                msg.body = _(f'Здравствуйте, {username}!\n\nСпасибо за регистрацию в KyrgyzRealty. Теперь вы можете добавлять объявления и искать недвижимость.')
                mail.send(msg)

                flash(_('Регистрация успешна. Пожалуйста, войдите.'), 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash(_('Имя пользователя или email уже заняты'), 'danger')

    return render_template('register.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, created_at FROM users WHERE id = ?", (session['user_id'],))
        user = cur.fetchone()

        cur.execute("SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a WHERE user_id = ? AND status = 'active'", (session['user_id'],))
        active_apartments = [dict(row) for row in cur.fetchall()]
        if active_apartments:
            df = pd.DataFrame(active_apartments)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            active_apartments = df.to_dict('records')

        cur.execute("SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a WHERE user_id = ? AND status = 'inactive'", (session['user_id'],))
        inactive_apartments = [dict(row) for row in cur.fetchall()]
        if inactive_apartments:
            df = pd.DataFrame(inactive_apartments)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            inactive_apartments = df.to_dict('records')

        if request.method == 'POST':
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            if not check_password_hash(user['password'], current_password):
                flash(_('Неверный текущий пароль'), 'danger')
                return redirect(url_for('profile'))

            if new_password != confirm_password:
                flash(_('Новые пароли не совпадают'), 'danger')
                return redirect(url_for('profile'))

            cur.execute("UPDATE users SET password = ? WHERE id = ?", (generate_password_hash(new_password), session['user_id']))
            conn.commit()
            flash(_('Пароль успешно изменен'), 'success')
            return redirect(url_for('profile'))

    return render_template('profile.html', user=dict(user), active_apartments=active_apartments, inactive_apartments=inactive_apartments)

@app.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if session['username'] != 'admin':
        flash(_('Доступ запрещен'), 'danger')
        return redirect(url_for('index'))

    page = int(request.args.get('page', 1))
    per_page = 10
    city = request.form.get('city', '') if request.method == 'POST' else request.args.get('city', '')
    status = request.form.get('status', '') if request.method == 'POST' else request.args.get('status', '')
    amenities = request.form.getlist('amenities') if request.method == 'POST' else request.args.getlist('amenities')

    with get_db() as conn:
        cur = conn.cursor()
        query = "SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a WHERE 1=1"
        params = []

        if city:
            query += " AND city = ?"
            params.append(city)
        if status:
            query += " AND status = ?"
            params.append(status)
        if amenities:
            for amenity in amenities:
                query += " AND amenities LIKE ?"
                params.append(f'%{amenity}%')

        cur.execute(f"SELECT COUNT(*) FROM apartments WHERE 1=1 {' AND '.join(['?']*len(params))}" if params else "SELECT COUNT(*) FROM apartments", params)
        total_apartments = cur.fetchone()[0]
        total_apartment_pages = (total_apartments + per_page - 1) // per_page

        query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
        cur.execute(query, params)
        apartments = [dict(row) for row in cur.fetchall()]
        if apartments:
            df = pd.DataFrame(apartments)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            apartments = df.to_dict('records')

        cur.execute("SELECT id, username, email, created_at FROM users")
        users = [dict(row) for row in cur.fetchall()]
        cur.execute("SELECT DISTINCT city FROM apartments")
        cities = [row['city'] for row in cur.fetchall()]

    return render_template('admin.html', apartments=apartments, users=users, page=page, total_apartment_pages=total_apartment_pages, 
                          cities=cities, selected_city=city, selected_status=status, selected_amenities=amenities)

@app.route('/admin/toggle_status/<int:apartment_id>', methods=['POST'])
@login_required
def toggle_status(apartment_id):
    if session['username'] != 'admin':
        return jsonify({'error': _('Доступ запрещен.')}), 403

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT status FROM apartments WHERE id = ?", (apartment_id,))
        current_status = cur.fetchone()
        if not current_status:
            return jsonify({'error': _('Квартира не найдена.')}), 404

        new_status = 'inactive' if current_status['status'] == 'active' else 'active'
        cur.execute("UPDATE apartments SET status = ? WHERE id = ?", (new_status, apartment_id))
        conn.commit()
        global model_data
        model_data = train_model()
        threading.Thread(target=update_all_predictions).start()
        return jsonify({'success': _(f'Статус изменен на {new_status}.')})

@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if session['username'] != 'admin':
        flash(_('Доступ запрещен'), 'danger')
        return redirect(url_for('index'))

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE apartments SET status = 'inactive' WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM favorites WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM comments WHERE user_id = ?", (user_id,))
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        global model_data
        model_data = train_model()
        threading.Thread(target=update_all_predictions).start()
        flash(_('Пользователь и его объявления удалены'), 'success')
    return redirect(url_for('admin'))

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    form = ResetPasswordForm()
    if form.validate_on_submit():
        email = form.email.data
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cur.fetchone()
            if user:
                token = generate_token(email)
                reset_url = url_for('reset_password_confirm', token=token, _external=True)
                msg = Message(_('Сброс пароля'), sender=app.config['MAIL_USERNAME'], recipients=[email])
                msg.body = _(f'Для сброса пароля перейдите по ссылке: {reset_url}\nСсылка действительна 1 час.')
                mail.send(msg)
                flash(_('Ссылка для сброса пароля отправлена на ваш email'), 'success')
            else:
                flash(_('Email не найден'), 'danger')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password_confirm(token):
    email = confirm_token(token)
    if not email:
        flash(_('Недействительная или просроченная ссылка'), 'danger')
        return redirect(url_for('login'))

    form = ResetPasswordConfirmForm()
    if form.validate_on_submit():
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE users SET password = ? WHERE email = ?", 
                        (generate_password_hash(form.new_password.data), email))
            conn.commit()
        flash(_('Пароль успешно изменен'), 'success')
        return redirect(url_for('login'))
    return render_template('reset_password_confirm.html', form=form, token=token)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('email', None)
    flash(_('Выход выполнен'), 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        with get_db() as conn:
            with app.open_resource('database.sql', mode='r') as f:
                conn.cursor().executescript(f.read())
            conn.commit()
    app.run(debug=True)