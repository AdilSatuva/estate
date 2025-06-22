from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, IntegerField, FloatField, SelectField, SelectMultipleField, FileField, PasswordField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, EqualTo, NumberRange
from flask_babel import Babel, _
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
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

# CSRF Protection
csrf = CSRFProtect(app)

# Flask-Babel
babel = Babel(app)

def get_locale():
    return session.get('lang', app.config['BABEL_DEFAULT_LOCALE'])

babel.init_app(app, locale_selector=get_locale)
app.jinja_env.globals['get_locale'] = get_locale

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Database connection
def get_db():
    conn = sqlite3.connect('real_estate.db')
    conn.row_factory = sqlite3.Row
    return conn

# Forms
class LoginForm(FlaskForm):
    username = StringField(_('Имя пользователя'), validators=[DataRequired()])
    password = PasswordField(_('Пароль'), validators=[DataRequired()])
    submit = SubmitField(_('Войти'))

class RegisterForm(FlaskForm):
    username = StringField(_('Имя пользователя'), validators=[DataRequired()])
    password = PasswordField(_('Пароль'), validators=[DataRequired()])
    confirm_password = PasswordField(_('Подтвердите пароль'), validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField(_('Зарегистрироваться'))

class AddApartmentForm(FlaskForm):
    city = SelectField(_('Город'), choices=[('Бишкек', 'Бишкек'), ('Ош', 'Ош'), ('Джалал-Абад', 'Джалал-Абад')], validators=[DataRequired()])
    rooms = SelectField(_('Комнаты'), choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')], validators=[DataRequired()])
    floor = IntegerField(_('Этаж'), validators=[DataRequired(), NumberRange(min=1, max=20)])
    area = FloatField(_('Площадь (м²)'), validators=[DataRequired(), NumberRange(min=10)])
    price = FloatField(_('Цена'), validators=[DataRequired(), NumberRange(min=0)])
    total_floors = IntegerField(_('Всего этажей'), validators=[DataRequired(), NumberRange(min=1)])
    description = TextAreaField(_('Описание'))
    amenities = SelectMultipleField(_('Удобства'), choices=[
        ('Парковка', _('Парковка')), ('Балкон', _('Балкон')), ('Лифт', _('Лифт')),
        ('Мебель', _('Мебель')), ('Кондиционер', _('Кондиционер')), ('Охрана', _('Охрана'))
    ])
    address = StringField(_('Адрес'), validators=[DataRequired()])
    images = FileField(_('Изображения'), validators=[])
    submit = SubmitField(_('Добавить квартиру'))

class SearchForm(FlaskForm):
    city = SelectField(_('Город'), choices=[('', _('Выберите город'))])
    address = StringField(_('Адрес'))
    min_price = IntegerField(_('Минимальная цена'))
    max_price = IntegerField(_('Максимальная цена'))
    min_area = FloatField(_('Минимальная площадь'))
    max_area = FloatField(_('Максимальная площадь'))
    rooms = SelectField(_('Комнаты'), choices=[('', _('Любое')), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')])
    amenities = SelectMultipleField(_('Удобства'), choices=[])
    sort_by = SelectField(_('Сортировать по'), choices=[
        ('price_asc', _('Цена (по возрастанию)')),
        ('price_desc', _('Цена (по убыванию)')),
        ('area_asc', _('Площадь (по возрастанию)')),
        ('area_desc', _('Площадь (по убыванию)'))
    ])
    submit = SubmitField(_('Найти'))

class ResetPasswordForm(FlaskForm):
    username = StringField(_('Имя пользователя'), validators=[DataRequired()])
    submit = SubmitField(_('Отправить ссылку'))

class ResetPasswordConfirmForm(FlaskForm):
    new_password = PasswordField(_('Новый пароль'), validators=[DataRequired()])
    confirm_password = PasswordField(_('Подтвердите пароль'), validators=[DataRequired(), EqualTo('new_password')])
    submit = SubmitField(_('Сохранить'))

class CommentForm(FlaskForm):
    content = TextAreaField(_('Комментарий'), validators=[DataRequired()])
    submit = SubmitField(_('Отправить'))

class PredictForm(FlaskForm):
    city = SelectField(_('Город'), choices=[('Бишкек', 'Бишкек'), ('Ош', 'Ош'), ('Джалал-Абад', 'Джалал-Абад')], validators=[DataRequired()])
    rooms = SelectField(_('Комнаты'), choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')], validators=[DataRequired()])
    floor = IntegerField(_('Этаж'), validators=[DataRequired(), NumberRange(min=1, max=20)])
    area = FloatField(_('Площадь (м²)'), validators=[DataRequired(), NumberRange(min=10, max=500)])
    total_floors = IntegerField(_('Всего этажей'), validators=[DataRequired(), NumberRange(min=1, max=50)])
    amenities = SelectMultipleField(_('Удобства'), choices=[
        ('Парковка', _('Парковка')), ('Балкон', _('Балкон')), ('Лифт', _('Лифт')),
        ('Мебель', _('Мебель')), ('Кондиционер', _('Кондиционер')), ('Охрана', _('Охрана'))
    ])
    submit = SubmitField(_('Предсказать цену'))

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
        logger.warning(f"Geocoding failed for {query}, using default coordinates")
        return 42.8746, 74.6122  # Bishkek default
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
        return 42.8746, 74.6122

def train_model():
    try:
        with get_db() as conn:
            df = pd.read_sql_query("SELECT city, rooms, floor, area, total_floors, amenities, price FROM apartments WHERE status = 'active'", conn)
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
        df = df[df['total_floors'] > 0]

        le = LabelEncoder()
        df['city_encoded'] = le.fit_transform(df['city'])

        amenities_list = ['Парковка', 'Балкон', 'Лифт', 'Мебель', 'Кондиционер', 'Охрана']
        for amenity in amenities_list:
            df[amenity] = df['amenities'].apply(lambda x: 1 if x and amenity in x.split(',') else 0)

        X = df[['city_encoded', 'rooms', 'floor', 'area', 'total_floors'] + amenities_list]
        y = df['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        with open('model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'label_encoder': le, 'features': X.columns.tolist()}, f)
        logger.debug("Model trained and saved")
        return {'model': model, 'label_encoder': le, 'features': X.columns.tolist()}
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
            cur.execute("SELECT id, city, rooms, floor, area, total_floors, amenities FROM apartments WHERE status = 'active'")
            apartments = [dict(row) for row in cur.fetchall()]
            if not apartments:
                logger.warning("No active apartments to update predictions")
                return

            df = pd.DataFrame(apartments)
            df['city_encoded'] = model_data['label_encoder'].transform(df['city'])
            amenities_list = ['Парковка', 'Балкон', 'Лифт', 'Мебель', 'Кондиционер', 'Охрана']
            for amenity in amenities_list:
                df[amenity] = df['amenities'].apply(lambda x: 1 if x and amenity in x.split(',') else 0)

            X = df[['city_encoded', 'rooms', 'floor', 'area', 'total_floors'] + amenities_list]
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

def generate_token(username):
    serializer = itsdangerous.URLSafeTimedSerializer(app.secret_key)
    return serializer.dumps(username, salt='password-reset-salt')

def confirm_token(token, expiration=3600):
    serializer = itsdangerous.URLSafeTimedSerializer(app.secret_key)
    try:
        username = serializer.loads(token, salt='password-reset-salt', max_age=expiration)
    except:
        return False
    return username

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
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'price_asc'))

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT city FROM apartments WHERE status = 'active'")
        cities = [row['city'] for row in cur.fetchall()]
        form.city.choices = [('', _('Выберите город'))] + [(city, city) for city in cities]

        cur.execute("SELECT amenities FROM apartments WHERE status = 'active' AND amenities IS NOT NULL")
        amenities_set = set()
        for row in cur.fetchall():
            if row['amenities']:
                amenities_set.update(row['amenities'].split(','))
        amenities = sorted(amenities_set)
        form.amenities.choices = [(amenity, amenity) for amenity in amenities]

        favorite_ids = []
        if 'user_id' in session:
            cur.execute("SELECT apartment_id FROM favorites WHERE user_id = ?", (session['user_id'],))
            favorite_ids = [row['apartment_id'] for row in cur.fetchall()]

        query_conditions = ["status = 'active'"]
        params = []

        if form.validate_on_submit():
            city = form.city.data
            address = form.address.data
            min_price = form.min_price.data or 0
            max_price = form.max_price.data or float('inf')
            min_area = form.min_area.data or 0
            max_area = form.max_area.data or float('inf')
            rooms = form.rooms.data
            amenities = form.amenities.data
            sort_by = form.sort_by.data if form.sort_by.data else sort_by

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

        query = f"""
            SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image
            FROM apartments a
            WHERE {' AND '.join(query_conditions)}
        """
        if sort_by == 'price_asc':
            query += " ORDER BY price ASC"
        elif sort_by == 'price_desc':
            query += " ORDER BY price DESC"
        elif sort_by == 'area_asc':
            query += " ORDER BY area ASC"
        elif sort_by == 'area_desc':
            query += " ORDER BY area DESC"

        count_query = f"SELECT COUNT(*) FROM apartments WHERE {' AND '.join(query_conditions)}"
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

        logger.debug(f"Search results: {len(apartments)} apartments, page {page}, sort_by {sort_by}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(apartments)
        return render_template('search.html', form=form, apartments=apartments, favorite_ids=favorite_ids, page=page, total_pages=total_pages, sort_by=sort_by)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model_data
    form = PredictForm()
    if model_data is None:
        model_data = load_model()
        if model_data is None:
            flash(_('Модель не загружена.'), 'danger')
            return render_template('predict.html', form=form)

    comparables = []
    if form.validate_on_submit():
        try:
            city = form.city.data
            rooms = int(form.rooms.data)
            floor = int(form.floor.data)
            area = float(form.area.data)
            total_floors = int(form.total_floors.data)
            amenities = form.amenities.data or []

            logger.debug(f"Predict input: city={city}, rooms={rooms}, floor={floor}, area={area}, total_floors={total_floors}, amenities={amenities}")

            input_data = pd.DataFrame({
                'city': [city],
                'rooms': [rooms],
                'floor': [floor],
                'area': [area],
                'total_floors': [total_floors]
            })

            amenities_list = ['Парковка', 'Балкон', 'Лифт', 'Мебель', 'Кондиционер', 'Охрана']
            for amenity in amenities_list:
                input_data[amenity] = 1 if amenity in amenities else 0

            le = model_data['label_encoder']
            input_data['city_encoded'] = le.transform(input_data['city'])
            X = input_data[['city_encoded', 'rooms', 'floor', 'area', 'total_floors'] + amenities_list]
            predicted_price = model_data['model'].predict(X)[0]

            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image
                    FROM apartments a
                    WHERE status = 'active' AND city = ? AND rooms = ? AND ABS(area - ?) < 10
                    LIMIT 3
                """, (city, rooms, area))
                comparables = [dict(row) for row in cur.fetchall()]
                if comparables:
                    df = pd.DataFrame(comparables)
                    df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
                    df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
                    comparables = df.to_dict('records')

            prediction = {
                'city': city,
                'rooms': rooms,
                'floor': floor,
                'area': area,
                'total_floors': total_floors,
                'amenities': amenities,
                'price': round(predicted_price, 2)
            }

            return render_template('predict.html', form=form, prediction=prediction, comparables=comparables)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            flash(_(f'Ошибка предсказания: {str(e)}'), 'danger')
            return render_template('predict.html', form=form)

    return render_template('predict.html', form=form, comparables=comparables)

@app.route('/add_apartment', methods=['GET', 'POST'])
@login_required
def add_apartment():
    form = AddApartmentForm()
    if form.validate_on_submit():
        try:
            city = form.city.data
            rooms = form.rooms.data
            floor = form.floor.data
            area = form.area.data
            price = form.price.data
            total_floors = form.total_floors.data
            description = form.description.data
            amenities = form.amenities.data
            address = form.address.data
            files = form.images.data

            if isinstance(files, FileStorage):
                files = [files] if files and files.filename else []
            elif not files:
                files = []

            if not files:
                flash(_('Выберите хотя бы одно изображение'), 'danger')
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
                else:
                    flash(f'Недопустимый формат: {file.filename}', 'danger')
                    return redirect(url_for('add_apartment'))

            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO apartments (city, rooms, floor, area, price, total_floors, description, amenities, address, status, created_at, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """, (city, int(rooms), int(floor), float(area), float(price), int(total_floors), description, ','.join(amenities), address, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), session['user_id']))
                apartment_id = cur.lastrowid

                for path in image_paths:
                    cur.execute("INSERT INTO apartment_images (apartment_id, image_path) VALUES (?, ?)", (apartment_id, path))

                conn.commit()
                global model_data
                model_data = train_model()
                threading.Thread(target=update_all_predictions).start()
                flash(_('Квартира добавлена'), 'success')
                return redirect(url_for('my_apartments'))
        except Exception as e:
            logger.error(f"Ошибка добавления квартиры: {e}")
            flash(_(f'Ошибка: {str(e)}'), 'danger')
            return redirect(url_for('add_apartment'))

    return render_template('add_apartment.html', form=form)

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

        form = AddApartmentForm()
        if request.method == 'POST':
            if form.validate_on_submit():
                try:
                    city = form.city.data
                    rooms = form.rooms.data
                    floor = form.floor.data
                    area = form.area.data
                    price = form.price.data
                    total_floors = form.total_floors.data
                    description = form.description.data
                    amenities = form.amenities.data
                    address = form.address.data
                    files = form.images.data
                    delete_images = request.form.getlist('delete_images')

                    if isinstance(files, FileStorage):
                        files = [files] if files and files.filename else []
                    elif not files:
                        files = []

                    image_paths = [img['image_path'] for img in images if str(img['id']) not in delete_images]
                    if files:
                        if len(files) + len(image_paths) > 10:
                            flash(_('Максимум 10 изображений'), 'danger')
                            return redirect(url_for('edit_apartment', apartment_id=apartment_id))
                        for file in files:
                            if file and allowed_file(file.filename):
                                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                file.save(file_path)
                                logger.debug(f"Изображение сохранено: {file_path}")
                                image_paths.append(f"/static/uploads/{filename}")
                            else:
                                flash(f'Недопустимый формат: {file.filename}', 'danger')
                                return redirect(url_for('edit_apartment', apartment_id=apartment_id))

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
        else:
            form.city.data = apartment['city']
            form.rooms.data = str(apartment['rooms'])
            form.floor.data = apartment['floor']
            form.area.data = apartment['area']
            form.price.data = apartment['price']
            form.total_floors.data = apartment['total_floors']
            form.description.data = apartment['description']
            form.amenities.data = apartment['amenities'].split(',') if apartment['amenities'] else []
            form.address.data = apartment['address']

        apartment = dict(apartment)
        apartment['amenities'] = apartment['amenities'].split(',') if apartment['amenities'] else []
        return render_template('edit_apartment.html', apartment=apartment, images=images, form=form)

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
        return redirect(url_for('my_apartments'))

@app.route('/delete_comment/<int:comment_id>/<int:apartment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id, apartment_id):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM comments WHERE id = ?", (comment_id,))
        comment = cur.fetchone()
        if not comment or (comment['user_id'] != session['user_id'] and session['username'] != 'admin'):
            flash(_('Вы не имеете прав для удаления этого комментария'), 'danger')
            return redirect(url_for('details', apartment_id=apartment_id))

        cur.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
        conn.commit()
        flash(_('Комментарий удален'), 'success')
        return redirect(url_for('details', apartment_id=apartment_id))

@app.route('/my_apartments', methods=['GET', 'POST'])
@login_required
def my_apartments():
    form = SearchForm()
    page = int(request.args.get('page', 1))
    per_page = 9
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'price_asc'))

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT city FROM apartments WHERE status = 'active' AND user_id = ?", (session['user_id'],))
        cities = [row['city'] for row in cur.fetchall()]
        form.city.choices = [('', _('Выберите город'))] + [(city, city) for city in cities]

        cur.execute("SELECT amenities FROM apartments WHERE status = 'active' AND amenities IS NOT NULL")
        amenities_set = set()
        for row in cur.fetchall():
            if row['amenities']:
                amenities_set.update(row['amenities'].split(','))
        amenities = sorted(amenities_set)
        form.amenities.choices = [(amenity, amenity) for amenity in amenities]

        query_conditions = ["user_id = ? AND status = 'active'"]
        params = [session['user_id']]

        if form.validate_on_submit():
            city = form.city.data
            address = form.address.data
            min_price = form.min_price.data or 0
            max_price = form.max_price.data or float('inf')
            min_area = form.min_area.data or 0
            max_area = form.max_area.data or float('inf')
            rooms = form.rooms.data
            amenities = form.amenities.data
            sort_by = form.sort_by.data if form.sort_by.data else sort_by

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

        count_query = f"SELECT COUNT(*) FROM apartments WHERE {' AND '.join(query_conditions)}"
        cur.execute(count_query, params)
        total = cur.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page

        query = f"SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image FROM apartments a WHERE {' AND '.join(query_conditions)}"
        if sort_by == 'price_asc':
            query += " ORDER BY price ASC"
        elif sort_by == 'price_desc':
            query += " ORDER BY price DESC"
        elif sort_by == 'area_asc':
            query += " ORDER BY area ASC"
        elif sort_by == 'area_desc':
            query += " ORDER BY area DESC"

        query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
        cur.execute(query, params)
        apartments = [dict(row) for row in cur.fetchall()]
        if apartments:
            df = pd.DataFrame(apartments)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            apartments = df.to_dict('records')

    return render_template('my_apartments.html', form=form, apartments=apartments, page=page, total_pages=total_pages, sort_by=sort_by)

@app.route('/favorites', methods=['GET', 'POST'])
@login_required
def favorites():
    form = SearchForm()
    page = int(request.args.get('page', 1))
    per_page = 9
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'price_asc'))

    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT city FROM apartments a JOIN favorites f ON a.id = f.apartment_id WHERE f.user_id = ? AND a.status = 'active'", (session['user_id'],))
        cities = [row['city'] for row in cur.fetchall()]
        form.city.choices = [('', _('Выберите город'))] + [(city, city) for city in cities]

        cur.execute("SELECT amenities FROM apartments WHERE status = 'active' AND amenities IS NOT NULL")
        amenities_set = set()
        for row in cur.fetchall():
            if row['amenities']:
                amenities_set.update(row['amenities'].split(','))
        amenities = sorted(amenities_set)
        form.amenities.choices = [(amenity, amenity) for amenity in amenities]

        query_conditions = ["f.user_id = ? AND a.status = 'active'"]
        params = [session['user_id']]

        if form.validate_on_submit():
            city = form.city.data
            address = form.address.data
            min_price = form.min_price.data or 0
            max_price = form.max_price.data or float('inf')
            min_area = form.min_area.data or 0
            max_area = form.max_area.data or float('inf')
            rooms = form.rooms.data
            amenities = form.amenities.data
            sort_by = form.sort_by.data if form.sort_by.data else sort_by

            if city:
                query_conditions.append("a.city = ?")
                params.append(city)
            if address:
                query_conditions.append("a.address LIKE ?")
                params.append(f'%{address}%')
            if min_price:
                query_conditions.append("a.price >= ?")
                params.append(float(min_price))
            if max_price != float('inf'):
                query_conditions.append("a.price <= ?")
                params.append(float(max_price))
            if min_area:
                query_conditions.append("a.area >= ?")
                params.append(float(min_area))
            if max_area != float('inf'):
                query_conditions.append("a.area <= ?")
                params.append(float(max_area))
            if rooms:
                query_conditions.append("a.rooms = ?")
                params.append(int(rooms))
            if amenities:
                for amenity in amenities:
                    query_conditions.append("a.amenities LIKE ?")
                    params.append(f'%{amenity}%')

        count_query = f"SELECT COUNT(*) FROM favorites f JOIN apartments a ON f.apartment_id = a.id WHERE {' AND '.join(query_conditions)}"
        cur.execute(count_query, params)
        total = cur.fetchone()[0]
        total_pages = (total + per_page - 1) // per_page

        query = f"""
            SELECT a.*, (SELECT image_path FROM apartment_images WHERE apartment_id = a.id LIMIT 1) as image 
            FROM apartments a JOIN favorites f ON a.id = f.apartment_id 
            WHERE {' AND '.join(query_conditions)}
        """
        if sort_by == 'price_asc':
            query += " ORDER BY a.price ASC"
        elif sort_by == 'price_desc':
            query += " ORDER BY a.price DESC"
        elif sort_by == 'area_asc':
            query += " ORDER BY a.area ASC"
        elif sort_by == 'area_desc':
            query += " ORDER BY a.area DESC"

        query += f" LIMIT {per_page} OFFSET {(page - 1) * per_page}"
        cur.execute(query, params)
        favorites = [dict(row) for row in cur.fetchall()]
        if favorites:
            df = pd.DataFrame(favorites)
            df['image'] = df['image'].fillna('https://images.unsplash.com/photo-1560448204-e02f11c3d0e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80')
            df['amenities'] = df['amenities'].apply(lambda x: x.split(',') if x else [])
            favorites = df.to_dict('records')

    return render_template('favorites.html', form=form, favorites=favorites, page=page, total_pages=total_pages, sort_by=sort_by)

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
                        (apartment_id, session['user_id'], content, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
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

        logger.info(f"Контактное сообщение: Имя: {name}, Email: {email}, Квартира: {apt['city']}, {apt['address']}, Сообщение: {message}")
        return jsonify({'success': _('Сообщение отправлено.')})
    except Exception as e:
        logger.error(f"Contact error: {e}")
        return jsonify({'error': _('Ошибка отправки сообщения.')}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cur.fetchone()

            if user and check_password_hash(user['password'], password):
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash(_('Вход успешен'), 'success')
                return redirect(url_for('index'))
            else:
                flash(_('Неверное имя пользователя или пароль'), 'danger')

    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        with get_db() as conn:
            cur = conn.cursor()
            try:
                cur.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                            (username, generate_password_hash(password), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                flash(_('Регистрация успешна. Пожалуйста, войдите.'), 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash(_('Имя пользователя уже занято'), 'danger')

    return render_template('register.html', form=form)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, created_at FROM users WHERE id = ?", (session['user_id'],))
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

        cur.execute("SELECT id, username, created_at FROM users")
        users = [dict(row) for row in cur.fetchall()]
        cur.execute("SELECT DISTINCT city FROM apartments")
        cities = [row['city'] for row in cur.fetchall()]
        cur.execute("SELECT amenities FROM apartments WHERE amenities IS NOT NULL")
        amenities_set = set()
        for row in cur.fetchall():
            if row['amenities']:
                amenities_set.update(row['amenities'].split(','))
        amenities = sorted(amenities_set)

    return render_template('admin.html', apartments=apartments, users=users, page=page, total_apartment_pages=total_apartment_pages, 
                          cities=cities, selected_city=city, selected_status=status, selected_amenities=amenities, amenities=amenities)

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
        username = form.username.data
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            user = cur.fetchone()
            if user:
                token = generate_token(username)
                reset_url = url_for('reset_password_confirm', token=token, _external=True)
                flash(_(f'Ссылка для сброса пароля: {reset_url} (Действительна 1 час). В реальном приложении это отправляется по email.'), 'success')
            else:
                flash(_('Имя пользователя не найдено'), 'danger')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password_confirm(token):
    username = confirm_token(token)
    if not username:
        flash(_('Недействительная или просроченная ссылка'), 'danger')
        return redirect(url_for('login'))

    form = ResetPasswordConfirmForm()
    if form.validate_on_submit():
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE users SET password = ? WHERE username = ?", 
                        (generate_password_hash(form.new_password.data), username))
            conn.commit()
        flash(_('Пароль успешно изменен'), 'success')
        return redirect(url_for('login'))
    return render_template('reset_password_confirm.html', form=form)

@app.route('/change_language/<lang>')
def change_language(lang):
    if lang in ['ru', 'pl']:
        session['lang'] = lang
    return redirect(request.referrer or url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash(_('Выход выполнен'), 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    with app.app_context():
        with get_db() as conn:
            with app.open_resource('database.sql', mode='r') as f:
                conn.cursor().executescript(f.read())
            conn.commit()
    app.run(debug=True)