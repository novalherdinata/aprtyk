import mysql.connector
import streamlit as st
import psycopg2

@st.cache_resource()
class connect_db():
    def __init__(self):
        # self.connection = psycopg2.connect(
        #     host="209.182.237.44",
        #     user="postgres",
        #     password="SamsungDev12@",
        #     database="rahmatANPR",
        #     port="5433"
        # )
        self.connection = psycopg2.connect(
            host= st.secrets["DB_HOST"],
            user= st.secrets["DB_USERNAME"],
            password= st.secrets["DB_PASSWORD"],
            database= st.secrets["DB_DATABASE"],
            port= st.secrets["DB_PORT"]
        )
        self.cursor = self.connection.cursor()

    def create_record(self,path, folder, filename, class_name):
        sql = "INSERT INTO image_dataset (path, folder, filename, class) VALUES (%s, %s, %s, %s)"
        val = (path,folder, filename, class_name)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record inserted successfully.")
                    
    def read_record(self, filename):
        sql = "SELECT * FROM image_dataset WHERE filename = %s LIMIT 1"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record
    
    def read_compare_data(self, filename):
        sql = "SELECT result FROM results_data WHERE id_data = %s"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record
    
    def read_check_id(self, filename):
        sql = "SELECT id FROM results_data WHERE id_data = %s"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record
    
    def read_compare_data_yolo(self, filename):
        sql = "SELECT resultyolo FROM results_data WHERE id_data = %s"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record
    
    def read_yolo_state(self, filename):
        sql = "SELECT statust FROM results_data WHERE id_data = %s"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record
    
    def read_compare_result(self, filename):
        sql = "SELECT status, plat FROM results_data WHERE id_data = %s"
        self.cursor.execute(sql, (filename,))
        record = self.cursor.fetchone()
        return record

    def update_record(self, id, new_filename, new_class_name):
        sql = "UPDATE image_dataset SET folder = %s, filename = %s, class = %s WHERE id = %s LIMIT 1"
        val = (new_class_name,new_filename, new_class_name, id)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record updated successfully.")

    def update_record_predict(self, id, new_filename, new_class_name):
        sql = "UPDATE results_data SET predicted_at = %s WHERE id_data = %s LIMIT 1"
        val = (new_class_name,new_filename, new_class_name, id)
        self.cursor.execute(sql, val)
        self.connection.commit()
        st.success("Record updated successfully.")
        

    def delete_record(self, id):
        sql = "DELETE FROM image_dataset WHERE id = %s LIMIT 1"
        self.cursor.execute(sql, (id,))
        self.connection.commit()
        st.success("Record deleted successfully.")

    def __del__(self):
        self.cursor.close()
        self.connection.close()
    
