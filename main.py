import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, Boolean, TIMESTAMP
from datetime import datetime, timedelta

# ---------------------------
# Параметры задачи
# ---------------------------

# Гравитационная константа
GM = 3.986004418e14  # м³/с²

# Начальные условия спутника
x0 = 7003.008789e3
y0 = -12206.626953e3
z0 = 21280.765625e3
vx0 = 0.7835417e3
vy0 = 2.8042530e3
vz0 = 1.3525150e3

y0_vec = [x0, y0, z0, vx0, vy0, vz0]

# Параметры интегрирования
t_span = (0, 86400)
t_eval = np.arange(0, 86401, 60)  # каждые 60 секунд

# Координаты приёмного пункта (широта, долгота)
lat_deg = 55.812843
lon_deg = 37.494503
R_earth = 6378137.0  # радиус Земли в м

# Перевод в радианы
lat_rad = np.deg2rad(lat_deg)
lon_rad = np.deg2rad(lon_deg)

# Перевод ПП в ИНСК (XYZ)
pp_x = R_earth * np.cos(lat_rad) * np.cos(lon_rad)
pp_y = R_earth * np.cos(lat_rad) * np.sin(lon_rad)
pp_z = R_earth * np.sin(lat_rad)

r_pp_vec = np.array([pp_x, pp_y, pp_z])

# Минимальный угол подъема h0
h0_deg = 5
sin_h0 = np.sin(np.deg2rad(h0_deg))

print(f"Координаты ПП в ИНСК: X={pp_x:.2f} м, Y={pp_y:.2f} м, Z={pp_z:.2f} м")

# ---------------------------
# Функция движения ИСЗ
# ---------------------------

def sat_motion(t, y):
    x, y_, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y_**2 + z**2)
    ax = -GM * x / r**3
    ay = -GM * y_ / r**3
    az = -GM * z / r**3
    return [vx, vy, vz, ax, ay, az]

# ---------------------------
# Интегрирование орбиты
# ---------------------------

sol = solve_ivp(
    sat_motion,
    t_span,
    y0_vec,
    method='RK45',
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-9
)

# ---------------------------
# Проверка видимости
# ---------------------------

visible_flags = []

for i in range(len(sol.t)):
    sat_pos = np.array([sol.y[0, i], sol.y[1, i], sol.y[2, i]])
    rel_vec = sat_pos - r_pp_vec

    norm_rel = np.linalg.norm(rel_vec)
    norm_pp = np.linalg.norm(r_pp_vec)

    cos_theta = np.dot(rel_vec, r_pp_vec) / (norm_rel * norm_pp)

    visible = cos_theta >= sin_h0
    visible_flags.append(visible)

# ---------------------------
# DataFrame траектории
# ---------------------------

df = pd.DataFrame({
    'time_sec': sol.t,
    'x': sol.y[0],
    'y': sol.y[1],
    'z': sol.y[2],
    'vx': sol.y[3],
    'vy': sol.y[4],
    'vz': sol.y[5],
    'visible': visible_flags
})

print(df.head())

# ---------------------------
# Графический вывод траектории
# ---------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# вся траектория
ax.plot(df['x'], df['y'], df['z'], color='gray', alpha=0.5, label='Траектория ИСЗ')

# видимые точки
visible_df = df[df['visible']]
ax.scatter(visible_df['x'], visible_df['y'], visible_df['z'],
           color='red', s=10, label='Зона видимости')

# точка ПП
ax.scatter(pp_x, pp_y, pp_z, color='blue', s=50, label='Приемный пункт')

# рисуем Землю
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)
xe = R_earth * np.outer(np.cos(u), np.sin(v))
ye = R_earth * np.outer(np.sin(u), np.sin(v))
ze = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(xe, ye, ze, color='lightblue', alpha=0.3)

ax.set_xlabel('X, м')
ax.set_ylabel('Y, м')
ax.set_zlabel('Z, м')
ax.set_title('3D траектория ИСЗ')

ax.legend()
plt.show()

# ---------------------------
# Определение интервалов видимости
# ---------------------------

df['visible_shift'] = df['visible'].shift(1, fill_value=False)
intervals = []
in_interval = False
start_time = None

for idx, row in df.iterrows():
    if not in_interval and row['visible']:
        start_time = row['time_sec']
        in_interval = True
    elif in_interval and not row['visible']:
        end_time = row['time_sec']
        intervals.append((start_time, end_time))
        in_interval = False

if in_interval:
    intervals.append((start_time, df['time_sec'].iloc[-1]))

# перевод интервалов в datetime
start_dt = datetime.now()
intervals_dt = [
    (start_dt + timedelta(seconds=float(start)),
     start_dt + timedelta(seconds=float(end)))
    for start, end in intervals
]

print(f"Обнаружено {len(intervals_dt)} интервалов видимости.")

# ---------------------------
# Запись в PostgreSQL
# ---------------------------

# Подключение
engine = create_engine('postgresql://postgres:123@localhost:5432/practice_mai')
metadata = MetaData()

# Определение таблиц

satellites = Table(
    'satellites', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String, nullable=False),
    Column('description', String)
)

receiver_points = Table(
    'receiver_points', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('name', String, nullable=False),
    Column('lat', Float, nullable=False),
    Column('lon', Float, nullable=False),
    Column('x', Float, nullable=False),
    Column('y', Float, nullable=False),
    Column('z', Float, nullable=False)
)

visible = Table(
    'visible', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('d_start', TIMESTAMP, nullable=False),
    Column('d_end', TIMESTAMP, nullable=False),
    Column('visible', Boolean, nullable=False),
    Column('id_pp', Integer, nullable=False),
    Column('id_satellite', Integer, nullable=False)
)

metadata.create_all(engine)

# Вставка данных
with engine.begin() as conn:
    # Спутник
    result = conn.execute(satellites.insert().returning(satellites.c.id), [
        {
            'name': 'Example_Satellite',
            'description': 'Test run'
        }
    ])
    sat_id = result.fetchone()[0]

    # Приемный пункт
    result = conn.execute(receiver_points.insert().returning(receiver_points.c.id), [
        {
            'name': 'Moscow Station',
            'lat': lat_deg,
            'lon': lon_deg,
            'x': pp_x,
            'y': pp_y,
            'z': pp_z
        }
    ])
    pp_id = result.fetchone()[0]

    # Интервалы видимости
    for d_start, d_end in intervals_dt:
        conn.execute(
            visible.insert(),
            [
                {
                    'd_start': d_start,
                    'd_end': d_end,
                    'visible': True,
                    'id_pp': pp_id,
                    'id_satellite': sat_id
                }
            ]
        )
