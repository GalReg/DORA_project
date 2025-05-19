from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import joblib

# Константы с путями
DATA_DIR = '/opt/airflow/data'
MODELS_DIR = '/opt/airflow/models'
RESULTS_DIR = '/opt/airflow/results'

def run_dvc_command(cmd):
    """Выполняет DVC команду в рабочей директории"""
    os.system(f'cd /opt/airflow && {cmd}')

def data_processing():
    try:
        # Чтение данных напрямую (без DVC API)
        df = pd.read_csv(f'{DATA_DIR}/raw_dataset.csv')
        
        # Обработка данных
        df = df.dropna()
        processed_path = f'{DATA_DIR}/processed_data.csv'
        df.to_csv(processed_path, index=False)
        
        # Версионирование через DVC
        run_dvc_command(f'dvc add {processed_path}')
        run_dvc_command('git add data/processed_data.csv.dvc data/.gitignore')
        run_dvc_command('git commit -m "Processed data version"')
        run_dvc_command('dvc push')
        return "Data processing completed successfully"
    except Exception as e:
        return f"Error in data_processing: {str(e)}"

def model_training():
    try:
        # Загрузка обработанных данных
        df = pd.read_csv(f'{DATA_DIR}/processed_data.csv')
        
        # Разделение на признаки и целевую переменную
        X = df.drop('PurchaseStatus', axis=1)
        y = df['PurchaseStatus']
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Обучение модели
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Сохранение модели
        model_path = f'{MODELS_DIR}/model.pkl'
        joblib.dump(model, model_path)
        
        # Версионирование модели
        run_dvc_command(f'dvc add {model_path}')
        run_dvc_command('git add models/model.pkl.dvc models/.gitignore')
        run_dvc_command('git commit -m "Model version"')
        run_dvc_command('dvc push')
        return "Model training completed successfully"
    except Exception as e:
        return f"Error in model_training: {str(e)}"

def model_testing():
    try:
        # Загрузка модели и данных
        model = joblib.load(f'{MODELS_DIR}/model.pkl')
        df = pd.read_csv(f'{DATA_DIR}/processed_data.csv')
        
        # Подготовка тестовых данных
        X_test = df.drop('PurchaseStatus', axis=1)
        y_test = df['PurchaseStatus']
        
        # Тестирование
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Сохранение результатов
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f'{RESULTS_DIR}/accuracy.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}")
        
        # Версионирование результатов
        run_dvc_command(f'dvc add {RESULTS_DIR}/accuracy.txt')
        run_dvc_command('git add results/accuracy.txt.dvc results/.gitignore')
        run_dvc_command('git commit -m "Test results version"')
        run_dvc_command('dvc push')
        return f"Model testing completed. Accuracy: {accuracy:.4f}"
    except Exception as e:
        return f"Error in model_testing: {str(e)}"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML pipeline with data processing, training and testing',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'dvc'],
) as dag:

    process_data = PythonOperator(
        task_id='data_processing',
        python_callable=data_processing,
    )

    train_model = PythonOperator(
        task_id='model_training',
        python_callable=model_training,
    )

    test_model = PythonOperator(
        task_id='model_testing',
        python_callable=model_testing,
    )

    process_data >> train_model >> test_model