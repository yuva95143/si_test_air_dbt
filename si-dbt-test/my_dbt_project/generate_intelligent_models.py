#!/usr/bin/env python3
"""
Enhanced Intelligent Model Generator using Google Gemini AI
Python 3.11.9 Optimized Version
"""
import os
import sys
import pandas as pd
import google.generativeai as genai
from pathlib import Path
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentModelGenerator:
    def __init__(self):
        self.api_key = os.environ.get('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        
        self.data_dir = Path('./data')
        self.models_dir = Path('./models/generated')
        self.staging_dir = Path('./models/staging')
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("IntelligentModelGenerator initialized successfully")

    def analyze_csv_structure(self, csv_file):
        """Analyze CSV structure and create comprehensive data profile"""
        try:
            # Read sample data
            df = pd.read_csv(csv_file, nrows=1000)  # Increased sample size
            
            # Enhanced analysis
            analysis = {
                'filename': csv_file.stem,
                'row_count': len(df),
                'columns': list(df.columns),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                'sample_data': df.head(5).to_dict('records'),
                'null_counts': df.isnull().sum().to_dict(),
                'unique_values': {col: df[col].nunique() for col in df.columns},
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
                'date_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
                'text_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            logger.info(f"Successfully analyzed {csv_file.name}: {analysis['row_count']} rows, {len(analysis['columns'])} columns")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {csv_file}: {e}")
            return None

    def generate_staging_model(self, csv_analysis):
        """Generate staging model SQL"""
        
        prompt = f"""
        Generate a dbt staging SQL model for CSV data with the following requirements:

        **Data Profile:**
        - File: {csv_analysis['filename']}
        - Rows: {csv_analysis['row_count']}
        - Columns: {csv_analysis['columns']}
        - Data Types: {json.dumps(csv_analysis['data_types'], indent=2)}
        - Numeric Columns: {csv_analysis['numeric_columns']}
        - Text Columns: {csv_analysis['text_columns']}

        **Requirements:**
        1. Create a staging model that reads from seeds/{csv_analysis['filename']}.csv
        2. Use appropriate SQL casting for data types
        3. Add data quality checks (null handling, duplicates)
        4. Use consistent naming conventions (snake_case)
        5. Add basic data validation
        6. Include useful comments
        7. Follow dbt best practices

        Return ONLY the SQL code for the dbt model:
        """

        try:
            response = self.model.generate_content(prompt)
            sql_code = response.text.strip()
            
            # Clean SQL code
            if '```
                sql_code = sql_code.split('```sql').split('```
            elif '```' in sql_code:
                sql_code = sql_code.replace('```
                
            return sql_code
            
        except Exception as e:
            logger.error(f"Error generating staging model: {e}")
            return None

    def generate_analytics_model(self, csv_analysis):
        """Generate analytics model SQL"""
        
        prompt = f"""
        Generate a dbt analytics SQL model that transforms the staging data:

        **Source Data:**
        - Staging Model: stg_{csv_analysis['filename']}
        - Columns: {csv_analysis['columns']}
        - Numeric Columns: {csv_analysis['numeric_columns']}

        **Requirements:**
        1. Reference the staging model: {{{{ ref('stg_{csv_analysis['filename']}') }}}}
        2. Add meaningful aggregations and calculations
        3. Include business logic transformations
        4. Add derived columns that provide insights
        5. Group data logically if applicable
        6. Include data quality metrics
        7. Use window functions where beneficial

        Return ONLY the SQL code for the analytics model:
        """

        try:
            response = self.model.generate_content(prompt)
            sql_code = response.text.strip()
            
            # Clean SQL code
            if '```sql' in sql_code:
                sql_code = sql_code.split('``````').strip()
            elif '```
                sql_code = sql_code.replace('```', '').strip()
                
            return sql_code
            
        except Exception as e:
            logger.error(f"Error generating analytics model: {e}")
            return None

    def create_seed_file(self, csv_file):
        """Copy CSV file to seeds directory"""
        try:
            seeds_dir = Path('./seeds')
            seeds_dir.mkdir(exist_ok=True)
            
            import shutil
            seed_file = seeds_dir / csv_file.name
            shutil.copy2(csv_file, seed_file)
            
            logger.info(f"Created seed file: {seed_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating seed file: {e}")
            return False

    def process_csv_files(self):
        """Process all CSV files in data directory"""
        csv_files = list(self.data_dir.glob('*.csv'))
        
        if not csv_files:
            logger.warning("No CSV files found in ./data directory")
            logger.info("MANUAL STEP: Place your CSV files in ~/si-dbt-test/my_dbt_project/data/")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process...")
        
        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")
            
            # Create seed file
            if not self.create_seed_file(csv_file):
                continue
                
            # Analyze CSV structure
            analysis = self.analyze_csv_structure(csv_file)
            if not analysis:
                continue
            
            # Generate staging model
            staging_sql = self.generate_staging_model(analysis)
            if staging_sql:
                staging_name = f"stg_{analysis['filename']}"
                staging_file = self.staging_dir / f"{staging_name}.sql"
                
                with open(staging_file, 'w') as f:
                    f.write(f"-- Staging model for {csv_file.name}\n")
                    f.write(f"-- Generated by Gemini AI on {datetime.now()}\n")
                    f.write(f"-- Python version: {sys.version}\n\n")
                    f.write(staging_sql)
                
                logger.info(f"Generated staging model: {staging_file}")
            
            # Generate analytics model
            analytics_sql = self.generate_analytics_model(analysis)
            if analytics_sql:
                analytics_name = f"analytics_{analysis['filename']}"
                analytics_file = self.models_dir / f"{analytics_name}.sql"
                
                with open(analytics_file, 'w') as f:
                    f.write(f"-- Analytics model for {csv_file.name}\n")
                    f.write(f"-- Generated by Gemini AI on {datetime.now()}\n")
                    f.write(f"-- Python version: {sys.version}\n\n")
                    f.write(analytics_sql)
                
                logger.info(f"Generated analytics model: {analytics_file}")
        
        logger.info("Model generation completed successfully!")

if __name__ == "__main__":
    try:
        generator = IntelligentModelGenerator()
        generator.process_csv_files()
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        print("MANUAL STEP: Add your Gemini API key to ~/.bashrc")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
