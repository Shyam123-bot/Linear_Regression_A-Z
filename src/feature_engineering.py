import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies.
# Subclasses must implement the apply_transformation method.
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass

# Concrete Strategy for Car Region and Car Type Engineering
class CarRegionAndTypeEngineering(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'Region' column based on 'Location', removes unnecessary columns,
        and categorizes cars into tiers based on 'Price'.

        Parameters:
        df (pd.DataFrame): The dataframe containing car details.

        Returns:
        pd.DataFrame: The transformed dataframe with 'Region' and 'Car_Type' columns.
        """
        logging.info("Applying car region mapping and tier categorization.")

        # Define regions
        regions = {
            'Delhi': 'North', 'Jaipur': 'North',
            'Chennai': 'South', 'Coimbatore': 'South', 'Hyderabad': 'South', 'Bangalore': 'South', 'Kochi': 'South',
            'Kolkata': 'East',
            'Mumbai': 'West', 'Pune': 'West', 'Ahmedabad': 'West'
        }
        
        # Map regions
        df["Region"] = df["Location"].replace(regions)

        # Drop unnecessary columns
        df.drop(["Car_Brand", "Model"], axis=1, inplace=True)

        # Categorize cars into tiers based on Price
        df["Car_Type"] = pd.cut(df["Price"], [-np.inf, 5.5, 10.5, 20.5, 45.0, 75.0, np.inf],
                                labels=["Tier1", "Tier2", "Tier3", "Tier4", "Tier5", "Tier6"])

        logging.info("Car region mapping and tier categorization completed.")
        return df
    
class YearToCarAgeTransformation(FeatureEngineeringStrategy):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Converting 'Year' to 'Car_Age'.")
        
        current_year = datetime.now().year
        df["Car_Age"] = current_year - df["Year"]
        
        df.drop(columns=["Year"], inplace=True)
        logging.info("Year conversion completed. 'Year' replaced with 'Car_Age'.")
        return df


# Concrete Strategy for Log Transformation
# ----------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        logging.info("Log transformation completed.")
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy applies standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed


# Concrete Strategy for Min-Max Scaling
# -------------------------------------
# This strategy applies Min-Max scaling to features, scaling them to a specified range, typically [0, 1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        logging.info(
            f"Applying Min-Max scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed.")
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        logging.info("One-hot encoding completed.")
        return df_transformed


# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEngineeringStrategy to apply transformations to a dataset.
class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy.")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)


# Example usage
if __name__ == "__main__":
    # Example dataframe
    df = pd.read_csv("D:\\Github\\Linear_Regression_A-Z\\extracted_data\\after_data_types_fixing.csv")

    # Apply car region and type categorization
    car_region_engineer = FeatureEngineer(CarRegionAndTypeEngineering())
    df_transformed = car_region_engineer.apply_feature_engineering(df)

    # Apply Year to Car_Age transformation
    year_transformer = FeatureEngineer(YearToCarAgeTransformation())
    df = year_transformer.apply_feature_engineering(df)

    # Log Transformation Example
    log_transformer = FeatureEngineer(LogTransformation(features=['Kilometers_Driven', 'Engine', 'Power']]))
    df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    standard_scaler = FeatureEngineer(StandardScaling(features=['Seats', 'Kilometers_Driven']))
    df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    minmax_scaler = FeatureEngineer(MinMaxScaling(features=['Year', 'Mileage','Engine','Power'], feature_range=(0, 1)))
    df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Transmission','Owner_Type','Region','Car_Type']))
    df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    # Save transformed data
    df_transformed.to_csv("D:/Github/Linear_Regression_A-Z/extracted_data/Adding_regions.csv", index=False)

    logging.info("Feature engineering pipeline completed.")

