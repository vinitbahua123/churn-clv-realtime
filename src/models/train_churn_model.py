"""
Churn Model Training with MLflow Experiment Tracking
Trains multiple models and logs metrics, parameters, and artifacts
"""

import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    """Train and evaluate churn prediction models with MLflow tracking"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = self.config['data']['processed_path']
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        logger.info("✅ MLflow tracking configured")
    
    def load_data(self):
        """Load processed data and split into train/test"""
        logger.info(f"📥 Loading data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        
        # Separate features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        # Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['training']['test_size'],
            random_state=self.config['training']['random_state'],
            stratify=y
        )
        
        logger.info(f"✅ Data loaded: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"   Train churn rate: {y_train.mean():.2%}")
        logger.info(f"   Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Standardize features"""
        logger.info("🔧 Scaling features...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler
        scaler_path = self.model_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"✅ Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled, scaler
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Calculate all evaluation metrics"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"\n   Confusion Matrix:")
        logger.info(f"   TN={tn}, FP={fp}")
        logger.info(f"   FN={fn}, TP={tp}")
        
        return metrics, cm
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train baseline Logistic Regression model"""
        logger.info("\n🎯 Training Logistic Regression (Baseline)...")
        
        with mlflow.start_run(run_name="logistic_regression"):
            # Train model
            model = LogisticRegression(
                random_state=self.config['training']['random_state'],
                max_iter=1000
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics, cm = self.evaluate_model(model, X_test, y_test, "Logistic Regression")
            
            # Log to MLflow
            mlflow.log_params({
                "model_type": "LogisticRegression",
                "solver": "lbfgs",
                "max_iter": 1000
            })
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("✅ Logistic Regression trained and logged")
            
            return model, metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        logger.info("\n🌲 Training Random Forest...")
        
        with mlflow.start_run(run_name="random_forest"):
            # Train model
            model = RandomForestClassifier(
                n_estimators=self.config['training']['n_estimators'],
                max_depth=self.config['training']['max_depth'],
                random_state=self.config['training']['random_state'],
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics, cm = self.evaluate_model(model, X_test, y_test, "Random Forest")
            
            # Log to MLflow
            mlflow.log_params({
                "model_type": "RandomForest",
                "n_estimators": self.config['training']['n_estimators'],
                "max_depth": self.config['training']['max_depth'],
                "n_jobs": -1
            })
            mlflow.log_metrics(metrics)
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("\n📊 Top 5 Important Features:")
            for idx, row in feature_importance.head().iterrows():
                logger.info(f"   {row['feature']}: {row['importance']:.4f}")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info("✅ Random Forest trained and logged")
            
            return model, metrics
    
    def save_best_model(self, models_metrics):
        """Save the best performing model"""
        # Find best model by ROC-AUC
        best_model_name = max(models_metrics, key=lambda x: models_metrics[x][1]['roc_auc'])
        best_model, best_metrics = models_metrics[best_model_name]
        
        logger.info(f"\n🏆 Best Model: {best_model_name}")
        logger.info(f"   ROC-AUC: {best_metrics['roc_auc']:.4f}")
        
        # Save best model
        model_path = self.model_dir / "churn_model.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"✅ Best model saved to {model_path}")
        
        return best_model, best_metrics
    
    def run_training_pipeline(self):
        """Execute full training pipeline"""
        logger.info("🚀 Starting model training pipeline...")
        
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        
        try:
            # Load and prepare data
            X_train, X_test, y_train, y_test = self.load_data()
            X_train_scaled, X_test_scaled, scaler = self.scale_features(X_train, X_test)
            
            # Train models
            models_metrics = {}
            
            lr_model, lr_metrics = self.train_logistic_regression(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            models_metrics['Logistic Regression'] = (lr_model, lr_metrics)
            
            rf_model, rf_metrics = self.train_random_forest(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            models_metrics['Random Forest'] = (rf_model, rf_metrics)
            
            # Save best model
            best_model, best_metrics = self.save_best_model(models_metrics)
            
            logger.info("\n✅ Training pipeline completed successfully!")
            logger.info(f"🎯 View results: mlflow ui")
            
            return best_model, best_metrics
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = ChurnModelTrainer()
    model, metrics = trainer.run_training_pipeline()
