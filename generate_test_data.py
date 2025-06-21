from faker import Faker
import random
from app import db, User, Apartment
from werkzeug.security import generate_password_hash
from predict import predict_price

fake = Faker('ru_RU')

def generate_test_data(num_users=5, num_apartments_per_user=3):
    # Создаем пользователей
    for _ in range(num_users):
        username = fake.user_name()
        while User.query.filter_by(username=username).first():
            username = fake.user_name()
        user = User(
            username=username,
            password_hash=generate_password_hash('password123'),
            email=fake.email()
        )
        db.session.add(user)
    db.session.commit()

    # Список возможных значений
    cities = ['Бишкек', 'Ош', 'Джалал-Абад']
    amenities_list = ['Парковка', 'Балкон', 'Лифт', 'Мебель', 'Кондиционер', 'Охрана']
    streets = ['пр. Чуй', 'ул. Советская', 'ул. Ленина', 'пр. Манаса', 'ул. Токтогула']

    # Создаем квартиры
    users = User.query.all()
    for user in users:
        for _ in range(num_apartments_per_user):
            city = random.choice(cities)
            amenities = random.sample(amenities_list, random.randint(0, len(amenities_list)))
            rooms = random.randint(1, 4)
            area = round(random.uniform(30, 150), 1)
            floor = random.randint(1, 20)
            total_floors = random.randint(floor, 25)
            
            # Предсказываем цену
            predicted_price = predict_price({
                'city': city,
                'rooms': rooms,
                'area': area,
                'floor': floor,
                'total_floors': total_floors,
                'amenities': amenities
            })

            apartment = Apartment(
                city=city,
                address=f"{random.choice(streets)} {random.randint(1, 200)}",
                rooms=rooms,
                area=area,
                floor=floor,
                total_floors=total_floors,
                price=predicted_price,
                description=fake.text(max_nb_chars=200),
                amenities=','.join(amenities),
                image='https://source.unsplash.com/300x200/?apartment',
                user_id=user.id,
                is_active=True
            )
            db.session.add(apartment)
    db.session.commit()
    print(f"Generated {num_users} users and {num_users * num_apartments_per_user} apartments.")

if __name__ == '__main__':
    from app import app
    with app.app_context():
        generate_test_data()