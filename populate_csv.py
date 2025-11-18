"""
California Housing Dataset Generator
Generates realistic synthetic data for the California Housing dataset.

This script creates a CSV with 1 million rows of housing data that follows
the statistical distribution of the original California Housing dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousingDataGenerator:
    """
    Generates synthetic California housing data with realistic distributions.
    
    Based on the statistical properties of the original California Housing dataset
    from the 1990 census.
    """
    
    def __init__(self, n_samples: int = 1_000_000, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define realistic parameter ranges based on California Housing dataset
        self.feature_params = {
            'MedInc': {
                'dist': 'lognormal',
                'mean': 1.2,
                'std': 0.45,
                'min': 0.5,
                'max': 15.0
            },
            'HouseAge': {
                'dist': 'uniform',
                'min': 1.0,
                'max': 52.0
            },
            'AveRooms': {
                'dist': 'gamma',
                'shape': 3.0,
                'scale': 1.8,
                'min': 1.0,
                'max': 15.0
            },
            'AveBedrms': {
                'dist': 'gamma',
                'shape': 2.5,
                'scale': 0.4,
                'min': 0.5,
                'max': 5.0
            },
            'Population': {
                'dist': 'lognormal',
                'mean': 7.0,
                'std': 0.7,
                'min': 100.0,
                'max': 6000.0
            },
            'AveOccup': {
                'dist': 'gamma',
                'shape': 2.0,
                'scale': 1.5,
                'min': 1.0,
                'max': 10.0
            },
            'Latitude': {
                'dist': 'truncnorm',
                'mean': 35.6,
                'std': 2.1,
                'min': 32.5,
                'max': 42.0
            },
            'Longitude': {
                'dist': 'truncnorm',
                'mean': -119.5,
                'std': 2.0,
                'min': -124.35,
                'max': -114.31
            }
        }
    
    def _generate_feature(self, feature_name: str, n: int) -> np.ndarray:
        """Generate data for a single feature based on its distribution."""
        params = self.feature_params[feature_name]
        dist = params['dist']
        
        if dist == 'uniform':
            data = np.random.uniform(params['min'], params['max'], n)
        
        elif dist == 'lognormal':
            data = np.random.lognormal(params['mean'], params['std'], n)
            data = np.clip(data, params['min'], params['max'])
        
        elif dist == 'gamma':
            data = np.random.gamma(params['shape'], params['scale'], n)
            data = np.clip(data, params['min'], params['max'])
        
        elif dist == 'truncnorm':
            # Generate truncated normal distribution
            mean, std = params['mean'], params['std']
            lower = (params['min'] - mean) / std
            upper = (params['max'] - mean) / std
            data = np.random.normal(mean, std, n)
            data = np.clip(data, params['min'], params['max'])
        
        return data
    
    def _generate_target(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Generate target variable (MedHouseVal) based on feature relationships.
        
        Creates a non-linear relationship between features and house value
        to simulate real-world complexity.
        """
        # Base price influenced by income (strongest predictor)
        base_price = 0.5 + 0.45 * features_df['MedInc']
        
        # Age effect (slightly U-shaped: very new or historic homes valued more)
        age_effect = 0.02 * np.abs(features_df['HouseAge'] - 25) / 25
        
        # Room premium
        room_effect = 0.15 * np.log1p(features_df['AveRooms'] - 1)
        
        # Occupancy penalty (overcrowding reduces value)
        occupancy_penalty = -0.1 * np.maximum(0, features_df['AveOccup'] - 3)
        
        # Location premium (proximity to coast, major cities)
        # Coastal California (more negative longitude) is more expensive
        location_effect = 0.002 * (features_df['Longitude'] + 120) ** 2
        
        # Southern California premium
        socal_premium = 0.3 * np.exp(-0.05 * (features_df['Latitude'] - 34) ** 2)
        
        # Combine all effects
        target = (
            base_price + 
            age_effect + 
            room_effect + 
            occupancy_penalty + 
            location_effect + 
            socal_premium
        )
        
        # Add realistic noise
        noise = np.random.normal(0, 0.3, len(features_df))
        target = target + noise
        
        # Clip to reasonable range for California housing (in $100k)
        target = np.clip(target, 0.15, 5.0)
        
        # Round to 2 decimal places
        return np.round(target, 2)
    
    def generate(self) -> pd.DataFrame:
        """Generate the complete synthetic dataset."""
        logger.info(f"Generating {self.n_samples:,} samples...")
        
        # Generate features
        data = {}
        for feature in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                       'Population', 'AveOccup', 'Latitude', 'Longitude']:
            logger.info(f"  Generating {feature}...")
            data[feature] = self._generate_feature(feature, self.n_samples)
        
        df = pd.DataFrame(data)
        
        # Round numeric features appropriately
        df['MedInc'] = df['MedInc'].round(2)
        df['HouseAge'] = df['HouseAge'].round(1)
        df['AveRooms'] = df['AveRooms'].round(2)
        df['AveBedrms'] = df['AveBedrms'].round(2)
        df['Population'] = df['Population'].round(0).astype(int)
        df['AveOccup'] = df['AveOccup'].round(2)
        df['Latitude'] = df['Latitude'].round(2)
        df['Longitude'] = df['Longitude'].round(2)
        
        # Generate target variable
        logger.info("  Generating target variable (MedHouseVal)...")
        df['MedHouseVal'] = self._generate_target(df)
        
        logger.info("✓ Dataset generation complete")
        return df
    
    def save(self, df: pd.DataFrame, output_path: str = "housing_data_1M.csv"):
        """Save the generated dataset to CSV."""
        output_path = Path(output_path)
        logger.info(f"Saving to {output_path}...")
        
        df.to_csv(output_path, index=False)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # Size in MB
        logger.info(f"✓ Saved {len(df):,} rows to {output_path}")
        logger.info(f"  File size: {file_size:.2f} MB")
    
    def print_statistics(self, df: pd.DataFrame):
        """Print summary statistics of the generated data."""
        logger.info("\n" + "="*60)
        logger.info("DATASET STATISTICS")
        logger.info("="*60)
        
        print("\nBasic Info:")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        print("\nFeature Statistics:")
        print(df.describe().round(2).to_string())
        
        print("\nMissing Values:")
        print(df.isnull().sum().to_string())
        
        print("\nCorrelation with Target:")
        correlations = df.corr()['MedHouseVal'].sort_values(ascending=False)
        print(correlations.to_string())


def main():
    """Main execution function."""
    # Configuration
    N_SAMPLES = 1_000_000
    OUTPUT_FILE = "housing_data.csv"
    RANDOM_STATE = 42
    
    logger.info("California Housing Data Generator")
    logger.info(f"Target: {N_SAMPLES:,} samples")
    
    generator = HousingDataGenerator(
        n_samples=N_SAMPLES,
        random_state=RANDOM_STATE
    )
    
    df = generator.generate()
    
    generator.save(df, OUTPUT_FILE)
    
    logger.info("Done!!")


if __name__ == "__main__":
    main()