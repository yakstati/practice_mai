from sqlalchemy import create_engine, MetaData, Table, Column, Integer, Float, String, Boolean, TIMESTAMP

class DBHandler:
    def __init__(self):
        self.engine = create_engine('postgresql://postgres:123@localhost:5432/practice_mai')
        self.metadata = MetaData()
        self.define_tables()
        self.metadata.create_all(self.engine, checkfirst=True)

    def define_tables(self):
        self.satellites = Table(
            'satellites', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('name', String, nullable=False),
            Column('description', String)
        )
        self.receiver_points = Table(
            'receiver_points', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('name', String, nullable=False),
            Column('lat', Float, nullable=False),
            Column('lon', Float, nullable=False),
            Column('x', Float, nullable=False),
            Column('y', Float, nullable=False),
            Column('z', Float, nullable=False)
        )
        self.visible = Table(
            'visible', self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('d_start', TIMESTAMP, nullable=False),
            Column('d_end', TIMESTAMP, nullable=False),
            Column('visible', Boolean, nullable=False),
            Column('id_pp', Integer, nullable=False),
            Column('id_satellite', Integer, nullable=False)
        )

    def save_to_db(self, df, intervals_dt, lat, lon, pp_xyz):
        with self.engine.begin() as conn:
            result = conn.execute(self.satellites.insert().returning(self.satellites.c.id), [
                {'name': 'Example_Satellite', 'description': 'Test run'}
            ])
            sat_id = result.fetchone()[0]

            result = conn.execute(self.receiver_points.insert().returning(self.receiver_points.c.id), [
                {
                    'name': 'Receiver Point',
                    'lat': lat,
                    'lon': lon,
                    'x': pp_xyz[0],
                    'y': pp_xyz[1],
                    'z': pp_xyz[2]
                }
            ])
            pp_id = result.fetchone()[0]

            for d_start, d_end in intervals_dt:
                conn.execute(self.visible.insert(), [
                    {
                        'd_start': d_start,
                        'd_end': d_end,
                        'visible': True,
                        'id_pp': pp_id,
                        'id_satellite': sat_id
                    }
                ])
