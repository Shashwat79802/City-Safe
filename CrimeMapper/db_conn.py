import psycopg2


def db_connection():
    db_params = {
        "user": "postgres",
        "password": "shashwat",
        "host": "localhost",
        "port": "5432",
        "database": "citysafe"
    }

    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    return connection, cursor
    