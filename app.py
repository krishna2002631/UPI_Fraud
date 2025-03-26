import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

class UPIFraudDetectionModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced'
            ))
        ])
        
        self.feature_columns = [
            'transaction_amount', 
            'sender_history_count', 
            'receiver_history_count', 
            'time_of_day', 
            'geo_distance', 
            'device_risk_score'
        ]
        
        self.initialize_model()

    def initialize_model(self):
        sample_data = self.generate_sample_data()
        self.train_model(sample_data)

    @staticmethod
    def generate_sample_data(num_samples=1000):
        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=num_samples),
            'transaction_amount': np.random.uniform(100, 50000, num_samples),
            'sender_lat': np.random.uniform(10, 30, num_samples),
            'sender_lon': np.random.uniform(70, 90, num_samples),
            'receiver_lat': np.random.uniform(10, 30, num_samples),
            'receiver_lon': np.random.uniform(70, 90, num_samples),
            'sender_history_count': np.random.randint(1, 10, num_samples),
            'receiver_history_count': np.random.randint(1, 10, num_samples),
            'is_new_device': np.random.choice([True, False], num_samples),
            'is_fraud': np.random.choice([True, False], num_samples, p=[0.1, 0.9])
        })

    def preprocess_data(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data])
        
        data['time_of_day'] = pd.to_datetime(data.get('timestamp', datetime.datetime.now())).dt.hour
        
        data['geo_distance'] = np.sqrt(
            (data.get('sender_lat', 0) - data.get('receiver_lat', 0))**2 + 
            (data.get('sender_lon', 0) - data.get('receiver_lon', 0))**2
        )
        
        data['device_risk_score'] = np.where(
            data.get('is_new_device', False), 1.5, 1.0
        )
        
        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 1 if col != 'transaction_amount' else 5000
        
        return data[self.feature_columns]

    def train_model(self, transactions_data):
        X = self.preprocess_data(transactions_data)
        y = transactions_data.get('is_fraud', np.random.choice([True, False], len(X)))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.pipeline.fit(X_train, y_train)
        
        y_pred = self.pipeline.predict(X_test)
        performance = classification_report(y_test, y_pred, output_dict=True)
        
        return performance

    def detect_fraud(self, transaction):
        X = self.preprocess_data(transaction)
        
        fraud_prob = self.pipeline.predict_proba(X)[0][1]
        is_fraud = fraud_prob > 0.5
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_prob),
            'features': X.to_dict(orient='records')[0]
        }

def main():
    st.set_page_config(page_title="UPI Fraud Detection", page_icon="💳", layout="wide")
    
    st.title("🛡️ UPI Fraud Detection Dashboard")
    
    model = UPIFraudDetectionModel()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🕵️ Fraud Detection", 
        "📊 Model Performance",
        "📈 Risk Visualization"
    ])
    
    with tab1:
        st.header("Real-time Fraud Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            transaction_amount = st.number_input("Transaction Amount (₹)", min_value=0.0, value=5000.0)
            sender_lat = st.number_input("Sender Latitude", min_value=-90.0, max_value=90.0, value=20.5)
            sender_lon = st.number_input("Sender Longitude", min_value=-180.0, max_value=180.0, value=80.2)
        
        with col2:
            is_new_device = st.checkbox("New Device", value=False)
            receiver_lat = st.number_input("Receiver Latitude", min_value=-90.0, max_value=90.0, value=22.3)
            receiver_lon = st.number_input("Receiver Longitude", min_value=-180.0, max_value=180.0, value=82.1)
        
        if st.button("Detect Fraud"):
            test_transaction = {
                'timestamp': datetime.datetime.now(),
                'transaction_amount': transaction_amount,
                'sender_lat': sender_lat,
                'sender_lon': sender_lon,
                'receiver_lat': receiver_lat,
                'receiver_lon': receiver_lon,
                'is_new_device': is_new_device,
                'sender_history_count': 5,
                'receiver_history_count': 5
            }
            
            result = model.detect_fraud(test_transaction)
            
            if result['is_fraud']:
                st.error(f"🚨 FRAUD ALERT! Fraud Probability: {result['fraud_probability']:.2%}")
            else:
                st.success(f"✅ Transaction Seems Safe. Fraud Probability: {result['fraud_probability']:.2%}")
            
            st.json(result['features'])
    
    with tab2:
        st.header("Model Performance Metrics")
        
        if st.button("Train and Evaluate Model"):
            with st.spinner("Training Model..."):
                transactions_data = UPIFraudDetectionModel.generate_sample_data()
                performance = model.train_model(transactions_data)
            
            st.success("Model Trained Successfully!")
            
            cols = st.columns(3)
            metrics = [
                ('Precision', performance['weighted avg']['precision']),
                ('Recall', performance['weighted avg']['recall']),
                ('F1-Score', performance['weighted avg']['f1-score'])
            ]
            
            for col, (name, value) in zip(cols, metrics):
                col.metric(name, f"{value:.2%}")
    
    with tab3:
        st.header("Risk Visualization")
        
        # Generate sample data for visualization
        sample_data = UPIFraudDetectionModel.generate_sample_data(500)
        
        # Fraud distribution by transaction amount
        st.subheader("Fraud Distribution by Transaction Amount")
        fig1 = px.box(sample_data, x='is_fraud', y='transaction_amount', 
                      title='Transaction Amount Distribution by Fraud Status',
                      labels={'is_fraud': 'Fraud Status', 'transaction_amount': 'Transaction Amount (₹)'})
        st.plotly_chart(fig1)
        
        # Fraud by time of day
        st.subheader("Fraud Occurrences by Time of Day")
        sample_data['hour'] = pd.to_datetime(sample_data['timestamp']).dt.hour
        fraud_by_hour = sample_data[sample_data['is_fraud']].groupby('hour').size().reset_index()
        fraud_by_hour.columns = ['hour', 'fraud_count']
        
        fig2 = px.bar(fraud_by_hour, x='hour', y='fraud_count',
                      title='Fraud Counts by Hour of Day',
                      labels={'hour': 'Hour of Day', 'fraud_count': 'Number of Frauds'})
        st.plotly_chart(fig2)

if __name__ == "__main__":
    main()