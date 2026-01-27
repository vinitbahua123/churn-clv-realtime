"""
Data Preprocessing Pipeline - Cloud-Ready Architecture
Uses DuckDB for SQL-based transformations (mimics Snowflake/Databricks)
"""

import duckdb
import pandas as pd
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """SQL-based data preprocessing using DuckDB"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = self.config['data']['raw_path']
        self.processed_path = self.config['data']['processed_path']
        self.features_db = self.config['data']['features_path']
        
        # Initialize DuckDB connection
        self.conn = duckdb.connect(self.features_db)
        logger.info("✅ DuckDB connection established")
    
    def load_raw_data(self):
        """Load raw data using SQL"""
        logger.info(f"📥 Loading data from {self.raw_path}")
        
        # Create table from CSV using SQL
        query = f"""
        CREATE OR REPLACE TABLE raw_churn AS 
        SELECT * FROM read_csv_auto('{self.raw_path}')
        """
        self.conn.execute(query)
        
        # Get row count
        count = self.conn.execute("SELECT COUNT(*) FROM raw_churn").fetchone()[0]
        logger.info(f"✅ Loaded {count} rows")
        
        return self.conn.execute("SELECT * FROM raw_churn").df()
    
    def clean_data(self):
        """Clean data using SQL transformations"""
        logger.info("🧹 Cleaning data...")
        
        # SQL-based cleaning (production approach)
        query = """
        CREATE OR REPLACE TABLE cleaned_churn AS
        SELECT 
            * EXCLUDE (customerID, TotalCharges),
            
            -- Fix TotalCharges (handle empty strings)
            CASE 
                WHEN TRIM(TotalCharges) = '' OR TotalCharges IS NULL THEN 0.0
                ELSE CAST(TotalCharges AS DOUBLE)
            END AS TotalCharges
            
        FROM raw_churn
        """
        
        self.conn.execute(query)
        
        # Get stats
        stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(CASE WHEN TotalCharges = 0 THEN 1 END) as zero_charges
            FROM cleaned_churn
        """).fetchone()
        
        logger.info(f"✅ Data cleaned - {stats[0]} rows, {stats[1]} zero charges fixed")
    
    def encode_features(self):
        """Encode categorical features using SQL"""
        logger.info("🔢 Encoding categorical features...")
        
        # Get categorical columns from config
        cat_cols = self.config['features']['categorical_cols']
        
        # Build CASE statements for each categorical column
        case_statements = []
        for col in cat_cols:
            # Get unique values
            unique_vals = self.conn.execute(f"""
                SELECT DISTINCT {col} 
                FROM cleaned_churn 
                WHERE {col} IS NOT NULL 
                ORDER BY {col}
            """).fetchall()
            
            # Create CASE statement
            cases = [f"WHEN {col} = '{val[0]}' THEN {idx}" 
                    for idx, val in enumerate(unique_vals)]
            case_stmt = f"CASE {' '.join(cases)} ELSE 0 END AS {col}_encoded"
            case_statements.append(case_stmt)
        
        # Create encoded table
        encoded_cols = ",\n            ".join(case_statements)
        
        query = f"""
        CREATE OR REPLACE TABLE encoded_churn AS
        SELECT 
            tenure,
            MonthlyCharges,
            TotalCharges,
            SeniorCitizen,
            {encoded_cols},
            CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS Churn
        FROM cleaned_churn
        """
        
        self.conn.execute(query)
        logger.info("✅ Features encoded")
    
    def save_processed_data(self):
        """Export processed data"""
        logger.info(f"💾 Saving processed data to {self.processed_path}")
        
        # Export to CSV
        query = f"""
        COPY encoded_churn TO '{self.processed_path}' 
        (HEADER, DELIMITER ',')
        """
        self.conn.execute(query)
        
        # Get final stats
        df = self.conn.execute("SELECT * FROM encoded_churn").df()
        
        logger.info(f"✅ Saved {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"📊 Target distribution:")
        logger.info(f"   No Churn: {(df['Churn'] == 0).sum()}")
        logger.info(f"   Churn: {(df['Churn'] == 1).sum()}")
        
        return df
    
    def run_pipeline(self):
        """Execute full preprocessing pipeline"""
        logger.info("🚀 Starting preprocessing pipeline...")
        
        # Ensure directories exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/features").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        try:
            self.load_raw_data()
            self.clean_data()
            self.encode_features()
            df = self.save_processed_data()
            
            logger.info("✅ Preprocessing pipeline completed successfully!")
            return df
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            raise
        finally:
            self.conn.close()

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.run_pipeline()
    print("\n📊 Final Dataset Preview:")
    print(df.head())
    print(f"\n✅ Final Shape: {df.shape}")
    print(f"\n📋 Columns: {df.columns.tolist()}")
