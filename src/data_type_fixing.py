import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Abstract Base Class for Data Type Fixing Strategy
class DataTypeFixingStrategy(ABC):
    @abstractmethod
    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to fix data types in the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with fixed data types.
        """
        pass


# Concrete Strategy: Convert Object Columns to Numeric
class ConvertObjectToNumericStrategy(DataTypeFixingStrategy):
    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts object columns containing numeric values into proper float types.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with corrected numerical data types.
        """
        logging.info("Converting object columns to numeric where applicable.")

        # Identify relevant columns
        num_values = []
        for colname in df.columns[df.dtypes == 'object']:
            if df[colname].apply(lambda x: isinstance(x, str)).any():  # Ensure column contains strings
                if df[colname].str.endswith(('pl', 'kg', 'CC', 'bhp', 'Lakh')).any():
                    num_values.append(colname)

        logging.info(f"Columns to convert: {num_values}")

        # Function to extract numeric values
        def obj_to_num(n):
            if isinstance(n, str):  # Ensure it's a string before processing
                if n.endswith('kmpl'):
                    return float(n.split('kmpl')[0])
                elif n.endswith('km/kg'):
                    return float(n.split('km/kg')[0])
                elif n.endswith('CC'):
                    return float(n.split('CC')[0])
                elif n.startswith('null'):  # Handle 'null bhp' values
                    return np.nan
                elif n.endswith('bhp'):
                    return float(n.split('bhp')[0])
            return np.nan  # Return NaN for anything else

        # Apply conversion to selected columns
        for colname in num_values:
            df[colname] = df[colname].apply(obj_to_num).replace(0.0, np.nan)

        logging.info("Object-to-numeric conversion completed.")
        return df


# Concrete Strategy: Convert Specified Columns to Category
class ConvertToCategoryStrategy(DataTypeFixingStrategy):
    def __init__(self, categorical_columns=None):
        """
        Initializes the strategy with specific columns to convert to categorical.

        Parameters:
        categorical_columns (list): List of column names to convert.
        """
        self.categorical_columns = categorical_columns or ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type"]

    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts specified columns to category type.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with specified columns converted to category type.
        """
        logging.info(f"Converting columns to category: {self.categorical_columns}")

        for col in self.categorical_columns:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype("category")

        logging.info("Categorical conversion completed.")
        return df


# Concrete Strategy: Extract Car Brands and Models
class ExtractCarBrandAndModelStrategy(DataTypeFixingStrategy):
    def fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts car brands and models from the 'Name' column and corrects inconsistencies.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with 'Car_Brand' and 'Model' columns added.
        """
        logging.info("Extracting car brands and models from the 'Name' column.")

        # Split the 'Name' column into 'Car_Brand' and 'Model'
        df[['Car_Brand', 'Model']] = df['Name'].str.split(n=1, expand=True)

        # Correct inconsistencies in car brands
        df['Car_Brand'] = df['Car_Brand'].replace('Land', 'Land_Rover')
        df['Car_Brand'] = df['Car_Brand'].replace('ISUZU', 'Isuzu')

        logging.info("Car brands and models extracted and inconsistencies corrected.")
        return df


# Context Class for Applying Data Type Fixing Strategies
class DataTypeFixer:
    def __init__(self, strategy: DataTypeFixingStrategy):
        """
        Initializes the DataTypeFixer with a specific strategy.

        Parameters:
        strategy (DataTypeFixingStrategy): The strategy to be used for fixing data types.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataTypeFixingStrategy):
        """
        Changes the strategy for data type fixing.

        Parameters:
        strategy (DataTypeFixingStrategy): The new strategy.
        """
        logging.info("Switching data type fixing strategy.")
        self._strategy = strategy

    def apply_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the data type fixing using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame.

        Returns:
        pd.DataFrame: The DataFrame with fixed data types.
        """
        logging.info("Applying data type fixing strategy.")
        return self._strategy.fix(df)


# Example Usage
if __name__ == "__main__":
    # Load data (Replace with actual path)
    df = pd.read_csv(r"D:\Github\Linear_Regression_A-Z\extracted_data\cleaned_used_cars_data.csv")

    # Initialize DataTypeFixer with object-to-numeric conversion strategy
    data_type_fixer = DataTypeFixer(ConvertObjectToNumericStrategy())
    df_fixed = data_type_fixer.apply_fix(df)

    # Switch strategy to convert specific columns to category
    data_type_fixer.set_strategy(ConvertToCategoryStrategy())
    df_fixed = data_type_fixer.apply_fix(df_fixed)

    # Switch strategy to extract car brands and models
    data_type_fixer.set_strategy(ExtractCarBrandAndModelStrategy())
    df_final = data_type_fixer.apply_fix(df_fixed)

    # Print final data types and save the DataFrame
    print(df_final.dtypes)
    df_final.to_csv("../extracted_data/after_data_types_fixing.csv", index=False)
    print("✅ Dataset saved after fixing data types.")