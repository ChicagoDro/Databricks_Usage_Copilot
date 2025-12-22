"""
Anomaly Detection for Cost Forecasting
Simple, production-ready implementation using statistical methods
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta


@dataclass
class Anomaly:
    """Represents a detected anomaly"""
    date: str
    entity_id: str
    entity_name: str
    entity_type: str
    metric_name: str
    value: float
    expected_value: float
    expected_range: Tuple[float, float]
    severity: str  # "low", "medium", "high", "critical"
    z_score: float
    pct_deviation: float
    context: Dict


class StatisticalAnomalyDetector:
    """
    Simple statistical anomaly detection using Z-score and IQR.
    
    Good for teaching fundamentals and production use.
    No external ML libraries required.
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        min_history_days: int = 14,
        use_rolling_window: bool = True,
        rolling_window_days: int = 7
    ):
        """
        Args:
            z_threshold: Standard deviations for anomaly (3.0 = 99.7% confidence)
            min_history_days: Minimum days needed to establish baseline
            use_rolling_window: Whether to use rolling stats (adapts to trends)
            rolling_window_days: Size of rolling window for statistics
        """
        self.z_threshold = z_threshold
        self.min_history_days = min_history_days
        self.use_rolling_window = use_rolling_window
        self.rolling_window_days = rolling_window_days
    
    def detect_cost_anomalies(
        self,
        df: pd.DataFrame,
        entity_col: str = "entity_id",
        entity_name_col: str = "entity_name",
        date_col: str = "usage_date",
        cost_col: str = "total_cost"
    ) -> List[Anomaly]:
        """
        Detect anomalies in cost data.
        
        Args:
            df: DataFrame with entity_id, usage_date, total_cost columns
            entity_col: Column name for entity identifier
            entity_name_col: Column name for entity name
            date_col: Column name for date
            cost_col: Column name for cost metric
        
        Returns:
            List of Anomaly objects
        """
        if df.empty:
            return []
        
        anomalies = []
        
        # Ensure date is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Group by entity
        for entity_id, entity_df in df.groupby(entity_col):
            entity_df = entity_df.sort_values(date_col).reset_index(drop=True)
            entity_name = entity_df[entity_name_col].iloc[0] if entity_name_col in entity_df else entity_id
            
            # Need enough history
            if len(entity_df) < self.min_history_days:
                continue
            
            # Calculate statistics for each point
            costs = entity_df[cost_col].values
            dates = entity_df[date_col].values
            
            for idx in range(self.min_history_days, len(costs)):
                current_cost = costs[idx]
                current_date = dates[idx]
                
                # Get historical data (exclude current point)
                if self.use_rolling_window:
                    # Use rolling window for adaptive detection
                    start_idx = max(0, idx - self.rolling_window_days - self.min_history_days)
                    historical = costs[start_idx:idx]
                else:
                    # Use all history up to this point
                    historical = costs[:idx]
                
                # Calculate statistics
                mean = np.mean(historical)
                std = np.std(historical)
                
                # Skip if no variance (constant value)
                if std == 0:
                    continue
                
                # Calculate Z-score
                z_score = (current_cost - mean) / std
                
                # Check if anomalous
                if abs(z_score) > self.z_threshold:
                    # Calculate expected range
                    expected_range = (
                        mean - self.z_threshold * std,
                        mean + self.z_threshold * std
                    )
                    
                    # Calculate percentage deviation
                    pct_deviation = ((current_cost - mean) / mean * 100) if mean > 0 else 0
                    
                    # Classify severity
                    severity = self._classify_severity(z_score)
                    
                    # Build context
                    context = {
                        "historical_mean": float(mean),
                        "historical_std": float(std),
                        "historical_min": float(np.min(historical)),
                        "historical_max": float(np.max(historical)),
                        "days_of_history": len(historical),
                        "previous_value": float(costs[idx - 1]) if idx > 0 else None,
                    }
                    
                    anomalies.append(Anomaly(
                        date=pd.Timestamp(current_date).strftime('%Y-%m-%d'),
                        entity_id=str(entity_id),
                        entity_name=str(entity_name),
                        entity_type="job",  # Could be parameterized
                        metric_name="cost",
                        value=float(current_cost),
                        expected_value=float(mean),
                        expected_range=(float(expected_range[0]), float(expected_range[1])),
                        severity=severity,
                        z_score=float(z_score),
                        pct_deviation=float(pct_deviation),
                        context=context
                    ))
        
        return anomalies
    
    def _classify_severity(self, z_score: float) -> str:
        """Map z-score to severity levels"""
        abs_z = abs(z_score)
        if abs_z > 5:
            return "critical"  # 1 in 3.5 million event
        elif abs_z > 4:
            return "high"      # 1 in 31,574 event
        elif abs_z > 3:
            return "medium"    # 1 in 370 event
        else:
            return "low"


class SimpleForecaster:
    """
    Simple time series forecasting using moving averages and trend.
    
    Good for teaching basics before moving to Prophet/ARIMA.
    """
    
    def __init__(
        self,
        seasonal_period: int = 7,  # Weekly seasonality
        trend_window: int = 14
    ):
        """
        Args:
            seasonal_period: Period of seasonality (7 = weekly)
            trend_window: Window for trend calculation
        """
        self.seasonal_period = seasonal_period
        self.trend_window = trend_window
    
    def forecast(
        self,
        df: pd.DataFrame,
        entity_col: str = "entity_id",
        date_col: str = "usage_date",
        value_col: str = "total_cost",
        forecast_days: int = 30
    ) -> pd.DataFrame:
        """
        Generate simple forecasts using moving average + trend.
        
        Args:
            df: Historical data
            entity_col: Column name for entity
            date_col: Column name for date
            value_col: Column name for value to forecast
            forecast_days: Number of days to forecast
        
        Returns:
            DataFrame with forecasted values
        """
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        forecasts = []
        
        for entity_id, entity_df in df.groupby(entity_col):
            entity_df = entity_df.sort_values(date_col).reset_index(drop=True)
            
            # Need enough history for trend + seasonality
            if len(entity_df) < self.trend_window + self.seasonal_period:
                continue
            
            values = entity_df[value_col].values
            last_date = entity_df[date_col].max()
            
            # Calculate trend (linear regression over recent data)
            recent_values = values[-self.trend_window:]
            x = np.arange(len(recent_values))
            trend_slope = np.polyfit(x, recent_values, 1)[0]
            
            # Calculate seasonal factors (average for each day of week/period)
            seasonal_factors = {}
            for i in range(self.seasonal_period):
                seasonal_values = values[i::self.seasonal_period]
                if len(seasonal_values) > 0:
                    seasonal_factors[i] = np.mean(seasonal_values)
            
            # Overall mean for normalization
            overall_mean = np.mean(values[-self.trend_window:])
            
            # Generate forecasts
            for day in range(1, forecast_days + 1):
                forecast_date = last_date + timedelta(days=day)
                
                # Base forecast from recent mean + trend
                base_forecast = overall_mean + (trend_slope * day)
                
                # Apply seasonal adjustment
                seasonal_idx = (len(values) + day) % self.seasonal_period
                if seasonal_idx in seasonal_factors and overall_mean > 0:
                    seasonal_adjustment = seasonal_factors[seasonal_idx] / overall_mean
                    forecast_value = base_forecast * seasonal_adjustment
                else:
                    forecast_value = base_forecast
                
                # Ensure non-negative
                forecast_value = max(0, forecast_value)
                
                forecasts.append({
                    entity_col: entity_id,
                    date_col: forecast_date,
                    value_col: forecast_value,
                    'forecast_type': 'simple_ma_trend',
                    'is_forecast': True
                })
        
        return pd.DataFrame(forecasts)


def calculate_forecast_accuracy(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    date_col: str = "usage_date",
    value_col: str = "total_cost"
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Returns:
        Dict with MAE, MAPE, RMSE metrics
    """
    # Merge actual and forecast
    merged = actual.merge(
        forecast,
        on=date_col,
        suffixes=('_actual', '_forecast')
    )
    
    if merged.empty:
        return {"mae": 0, "mape": 0, "rmse": 0, "n": 0}
    
    actual_vals = merged[f"{value_col}_actual"].values
    forecast_vals = merged[f"{value_col}_forecast"].values
    
    # Mean Absolute Error
    mae = np.mean(np.abs(actual_vals - forecast_vals))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual_vals - forecast_vals) / np.maximum(actual_vals, 1))) * 100
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((actual_vals - forecast_vals) ** 2))
    
    return {
        "mae": float(mae),
        "mape": float(mape),
        "rmse": float(rmse),
        "n": len(merged)
    }


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    sample_data = pd.DataFrame({
        'entity_id': 'JOB-001',
        'entity_name': 'Daily ETL Job',
        'usage_date': dates,
        'total_cost': 100 + np.random.normal(0, 10, 90) + np.sin(np.arange(90) * 2 * np.pi / 7) * 20
    })
    
    # Add an anomaly
    sample_data.loc[50, 'total_cost'] = 300  # Spike
    
    # Detect anomalies
    detector = StatisticalAnomalyDetector(z_threshold=3.0)
    anomalies = detector.detect_cost_anomalies(sample_data)
    
    print(f"Found {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"  {a.date}: ${a.value:.2f} (expected ${a.expected_value:.2f}, "
              f"severity={a.severity}, z={a.z_score:.2f})")
    
    # Generate forecast
    forecaster = SimpleForecaster()
    forecast_df = forecaster.forecast(sample_data, forecast_days=30)
    
    print(f"\nGenerated {len(forecast_df)} forecast points")
    print(forecast_df.head())