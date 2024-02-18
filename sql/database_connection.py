import pandas.io.sql as psql
import psycopg2
from sqlalchemy import create_engine


class SoccerDatabase:
    """Connection to DB and data extraction."""

    def __init__(self, host, database, user, password, port):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port

        self.conn = psycopg2.connect(
            host=host, database=database, user=user, password=password, port=port
        )

    def create_table(self, query):
        cursor = self.conn.cursor()

        cursor.execute(query=query)

        self.conn.commit()
        cursor.close()

    def query(self, query):
        df = psql.read_sql(query, self.conn)

        return df

    def show_tables(self):
        cursor = self.conn.cursor()

        cursor.execute(
            """
               SELECT table_name FROM information_schema.tables
               WHERE table_schema = 'public'
               """
        )

        tables = [i[0] for i in cursor.fetchall()]

        self.conn.commit()

        cursor.close()

        return tables

    def write_dataframe(self, table_name, df):
        engine = create_engine(
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

        df.to_sql(table_name, engine, if_exists="append", index=False)

    def drop_table(self, table_name):
        cursor = self.conn.cursor()

        # cursor.execute(f"TRUNCATE {table_name}; DELETE FROM {table_name};")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")

        # Commit the changes to the database
        self.conn.commit()

        # Close communication with the PostgreSQL database
        cursor.close()

    def close(self):
        self.conn.close()
