import psycopg2
from config import host,user,password,db_name
from querylist import queryInsertTable
import streamlit as st
def db_connection():
    return psycopg2.connect(
        host=host,
        user=user,
        password=password,
        database=db_name
    )
def execute_query(query, *params):
    st.error("На данном сайте не работает БД, если требуется работа с БД, установите проект локально")
    return
    connection = db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)

        connection.commit()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        connection.close()

def insert_data_into_db(data):
    st.error("На данном сайте не работает БД, если требуется работа с БД, установите проект локально")
    return
    connection = db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.executemany(queryInsertTable,data)

        connection.commit()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        connection.close()