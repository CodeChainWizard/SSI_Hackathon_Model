import psycopg2
from psycopg2.extras import Json

def save_prediction(prediction_type, prediction_data):
    conn = psycopg2.connect(database="ai_system", user="admin", password="password", host="localhost", port="5432")
    cursor = conn.cursor()

    query = "INSERT INTO predictions (type, data) VALUES (%s, %s)"
    cursor.execute(query, (prediction_type, Json(prediction_data)))

    conn.commit()
    conn.close()
    print("✅ Prediction saved to DB!")
