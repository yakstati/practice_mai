import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta

GM = 3.986004418e14  # м³/с²

def calculate_satellite_trajectory(lat_deg, lon_deg):
    # начальные условия из ОДК ГЛОНАСС
    x0 = 7003.008789e3
    y0 = -12206.626953e3
    z0 = 21280.765625e3
    vx0 = 0.7835417e3
    vy0 = 2.8042530e3
    vz0 = 1.3525150e3

    y0_vec = [x0, y0, z0, vx0, vy0, vz0]

    t_span = (0, 86400) # моделяция на 1 день
    t_eval = np.arange(0, 86401, 60) # шаг - минута

    R_earth = 6378137.0   # для отрисовки Земли
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    pp_x = R_earth * np.cos(lat_rad) * np.cos(lon_rad)  # перевод в XYZ
    pp_y = R_earth * np.cos(lat_rad) * np.sin(lon_rad)
    pp_z = R_earth * np.sin(lat_rad)
    r_pp_vec = np.array([pp_x, pp_y, pp_z])

    sin_h0 = np.sin(np.deg2rad(5))

    def sat_motion(t, y):        # система ДУ     (спросить нужно ли моделировать на исходной системе)
        x, y_, z, vx, vy, vz = y
        r = np.sqrt(x**2 + y_**2 + z**2)
        ax = -GM * x / r**3
        ay = -GM * y_ / r**3
        az = -GM * z / r**3
        return [vx, vy, vz, ax, ay, az]

    sol = solve_ivp(          # интегрирование РК45
        sat_motion,
        t_span,
        y0_vec,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9
    )

    visible_flags = []
    for i in range(len(sol.t)):                  # сохранение интервалов
        sat_pos = np.array([sol.y[0, i], sol.y[1, i], sol.y[2, i]])
        rel_vec = sat_pos - r_pp_vec
        norm_rel = np.linalg.norm(rel_vec)
        norm_pp = np.linalg.norm(r_pp_vec)
        cos_theta = np.dot(rel_vec, r_pp_vec) / (norm_rel * norm_pp)
        visible = cos_theta >= sin_h0
        visible_flags.append(visible)          

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

    # интервалы видимости
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

    start_dt = datetime.now()
    intervals_dt = [
        (start_dt + timedelta(seconds=float(start)),
         start_dt + timedelta(seconds=float(end)))
        for start, end in intervals
    ]

    return df, intervals_dt, (pp_x, pp_y, pp_z)
