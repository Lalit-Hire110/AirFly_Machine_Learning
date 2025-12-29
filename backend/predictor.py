"""
AirFly Predictor Module
Singleton-based model loader and prediction engine for flight delay forecasting.
"""

import json
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class FlightDelayPredictor:
    """
    Singleton predictor for flight delay forecasting.
    Loads models once on initialization and provides prediction methods.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, model_dir: str = None):
        if cls._instance is None:
            cls._instance = super(FlightDelayPredictor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, model_dir: str = None):
        """
        Initialize predictor with model artifacts.
        
        Args:
            model_dir: Path to model artifacts directory. Defaults to '../AirFly_Machine_Learning/model_artifacts'
        """
        if self._initialized:
            return
            
        if model_dir is None:
            # Find model_artifacts by searching up from current file
            # This works whether we're imported from frontend/, tests/, or run directly
            current = Path(__file__).resolve().parent
            
            # Try parent directory first (when in backend/)
            model_dir = current.parent / "AirFly_Machine_Learning" / "model_artifacts"
            
            # If not found, try current directory (in case we're already at root)
            if not model_dir.exists():
                model_dir = current / "AirFly_Machine_Learning" / "model_artifacts"
            
            # If still not found, search up the tree
            if not model_dir.exists():
                search_path = current
                for _ in range(5):  # Search up to 5 levels
                    test_path = search_path / "AirFly_Machine_Learning" / "model_artifacts"
                    if test_path.exists():
                        model_dir = test_path
                        break
                    search_path = search_path.parent
        
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load model artifacts
        try:
            self.clf = load(self.model_dir / "clf_pipe.joblib")
            self.reg_norm = load(self.model_dir / "normal_reg_pipe.joblib")
            self.reg_heavy = load(self.model_dir / "heavy_reg_pipe.joblib")
            
            with open(self.model_dir / "features.json", "r") as f:
                self.features = json.load(f)
            
            with open(self.model_dir / "metadata.json", "r") as f:
                self.metadata = json.load(f)
            
            with open(self.model_dir / "versions.json", "r") as f:
                self.versions = json.load(f)
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {str(e)}")
        
        self.heavy_threshold = self.features.get("heavy_threshold", 30.0)
        self.numeric_features = self.features["numeric_features"]
        self.categorical_features = self.features["categorical_features"]
        self.all_features = self.numeric_features + self.categorical_features
        
        self._initialized = True
        print(f"✓ Models loaded successfully from {self.model_dir}")
    
    def _validate_input(self, row_dict: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate input dictionary has required features.
        
        Returns:
            (is_valid, error_message)
        """
        missing_features = []
        for feature in self.all_features:
            if feature not in row_dict:
                missing_features.append(feature)
        
        if missing_features:
            return False, f"Missing required features: {', '.join(missing_features)}"
        
        return True, None
    
    def predict_single(self, row_dict: Dict) -> Dict:
        """
        Predict delay for a single flight.
        
        Args:
            row_dict: Dictionary with feature values (must include all required features)
        
        Returns:
            Dictionary with:
                - predicted_delay_min: Final predicted delay in minutes
                - heavy_flag: 1 if heavy delay predicted, 0 otherwise
                - pred_norm: Normal regime prediction
                - pred_heavy: Heavy regime prediction
                - risk_level: "Low", "Medium", or "High"
                - expected_error: Expected MAE for the regime
                - model_version: Model version info
        """
        # Validate input
        is_valid, error_msg = self._validate_input(row_dict)
        if not is_valid:
            raise ValueError(error_msg)
        
        try:
            # Build DataFrame
            df = pd.DataFrame([row_dict])
            X = df[self.all_features]
            
            # Get classifier prediction (0 = normal, 1 = heavy)
            heavy_prob = self.clf.predict(X)[0]
            
            # Get regression predictions (in log1p space)
            pred_norm_log = self.reg_norm.predict(X)[0]
            pred_heavy_log = self.reg_heavy.predict(X)[0]
            
            # Transform back to minutes
            pred_norm = float(np.expm1(pred_norm_log))
            pred_heavy = float(np.expm1(pred_heavy_log))
            
            # Choose final prediction based on classifier
            pred = pred_heavy if heavy_prob == 1 else pred_norm
            pred = max(0.0, pred)  # Ensure non-negative
            
            # Determine risk level
            if pred < 15:
                risk_level = "Low"
            elif pred < 45:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Expected error from metadata
            if heavy_prob == 1:
                expected_error = self.metadata["regression"]["heavy_segment"]["MAE"]
            else:
                expected_error = self.metadata["regression"]["normal_segment"]["MAE"]
            
            return {
                "predicted_delay_min": round(pred, 2),
                "heavy_flag": int(heavy_prob),
                "pred_norm": round(pred_norm, 2),
                "pred_heavy": round(pred_heavy, 2),
                "risk_level": risk_level,
                "expected_error": round(expected_error, 2),
                "model_version": "v1.0"
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict delays for a batch of flights.
        
        Args:
            df: DataFrame with required features
        
        Returns:
            DataFrame with original data plus prediction columns
        """
        # Validate all required features are present
        missing_features = [f for f in self.all_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        try:
            X = df[self.all_features]
            
            # Get predictions
            heavy_probs = self.clf.predict(X)
            pred_norm_logs = self.reg_norm.predict(X)
            pred_heavy_logs = self.reg_heavy.predict(X)
            
            # Transform back to minutes
            pred_norms = np.expm1(pred_norm_logs)
            pred_heavys = np.expm1(pred_heavy_logs)
            
            # Choose final predictions
            predictions = np.where(heavy_probs == 1, pred_heavys, pred_norms)
            predictions = np.maximum(0, predictions)  # Ensure non-negative
            
            # Add prediction columns
            result_df = df.copy()
            result_df['predicted_delay_min'] = predictions.round(2)
            result_df['heavy_flag'] = heavy_probs
            result_df['pred_norm'] = pred_norms.round(2)
            result_df['pred_heavy'] = pred_heavys.round(2)
            
            # Add risk levels
            risk_levels = []
            for pred in predictions:
                if pred < 15:
                    risk_levels.append("Low")
                elif pred < 45:
                    risk_levels.append("Medium")
                else:
                    risk_levels.append("High")
            result_df['risk_level'] = risk_levels
            
            return result_df
            
        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata and evaluation metrics.
        
        Returns:
            Dictionary with model information
        """
        return {
            "metadata": self.metadata,
            "versions": self.versions,
            "features": {
                "numeric_count": len(self.numeric_features),
                "categorical_count": len(self.categorical_features),
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features
            },
            "heavy_threshold": self.heavy_threshold
        }
    
    def get_feature_importance_approximate(self, row_dict: Dict, top_n: int = 6) -> List[Dict]:
        """
        Get approximate feature importance for a single prediction.
        This is a simplified version that shows which features have the most impact.
        
        Args:
            row_dict: Input features
            top_n: Number of top features to return
        
        Returns:
            List of dicts with feature names and importance scores
        """
        # Get baseline prediction
        baseline_pred = self.predict_single(row_dict)
        baseline_delay = baseline_pred["predicted_delay_min"]
        
        importances = []
        
        # For numeric features, perturb by 10% and measure impact
        for feature in self.numeric_features[:10]:  # Limit to first 10 for speed
            if feature in row_dict and row_dict[feature] != 0:
                perturbed = row_dict.copy()
                perturbed[feature] = row_dict[feature] * 1.1  # 10% increase
                
                try:
                    perturbed_pred = self.predict_single(perturbed)
                    impact = abs(perturbed_pred["predicted_delay_min"] - baseline_delay)
                    importances.append({
                        "feature": feature,
                        "importance": impact
                    })
                except:
                    continue
        
        # Sort by importance and return top N
        importances.sort(key=lambda x: x["importance"], reverse=True)
        return importances[:top_n]
    
    def get_required_features(self) -> List[str]:
        """Get list of all required feature names."""
        return self.all_features.copy()


# Convenience function for quick access
def get_predictor(model_dir: str = None) -> FlightDelayPredictor:
    """Get or create the singleton predictor instance."""
    return FlightDelayPredictor(model_dir)
