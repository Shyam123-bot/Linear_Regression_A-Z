from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


# Abstract Base Class for Univariate Analysis Strategy
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str = None):
        """
        Perform univariate analysis on a specific feature or all features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed. If None, analyze all features.

        Returns:
        None: This method visualizes the distribution of the feature(s).
        """
        pass


# Concrete Strategy for Numerical Features
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str = None):
        """
        Plots both a histogram and a box plot for numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed. If None, analyze all numerical features.

        Returns:
        None: Displays histograms and box plots for the specified or all numerical features.
        """
        if feature:
            # Analyze a single feature
            self._plot_single_feature(df, feature)
        else:
            # Analyze all numerical features
            numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
            self._plot_all_features(df, numerical_columns)

    def _plot_single_feature(self, df: pd.DataFrame, feature: str):
        """
        Plots a histogram and a box plot for a single numerical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram and a box plot for the feature.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # Create subplots

        # Histogram for Distribution
        sns.histplot(df[feature], kde=False, bins=30, ax=axes[0])
        axes[0].set_title(f"Histogram of {feature}")
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel("Frequency")

        # Box Plot for Central Tendency & Dispersion
        sns.boxplot(x=df[feature], showmeans=True, color="yellow", ax=axes[1])
        axes[1].set_title(f"Box Plot of {feature}")
        axes[1].set_xlabel(feature)

        plt.tight_layout()  # Adjust layout
        plt.show()

    def _plot_all_features(self, df: pd.DataFrame, numerical_columns: list):
        """
        Plots histograms and box plots for all numerical features.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        numerical_columns (list): List of numerical column names to be analyzed.

        Returns:
        None: Displays histograms and box plots for all numerical features.
        """
        # Plot histograms for all numerical features
        plt.figure(figsize=(17, 75))
        for i in range(len(numerical_columns)):
            plt.subplot(len(numerical_columns), 2, 2 * i + 1)
            sns.histplot(df[numerical_columns[i]], kde=False)
            plt.tight_layout()
            plt.title(numerical_columns[i], fontsize=25)

        plt.show()

        # Plot box plots for all numerical features
        plt.figure(figsize=(15, 35))
        for i in range(len(numerical_columns)):
            plt.subplot(len(numerical_columns), 2, 2 * i + 1)
            sns.boxplot(df[numerical_columns[i]], showmeans=True, color="yellow")
            plt.tight_layout()
            plt.title(numerical_columns[i], fontsize=25)

        plt.show()


# Concrete Strategy for Categorical Features
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str = None):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed. If None, analyze all categorical features.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        if feature:
            # Analyze a single feature
            self._plot_single_feature(df, feature)
        else:
            # Analyze all categorical features
            categorical_columns = df.select_dtypes(include="object").columns.tolist()
            for col in categorical_columns:
                self._plot_single_feature(df, col)

    def _plot_single_feature(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a single categorical feature.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot for the feature.
        """
        plt.figure(figsize=(12, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# Context Class that uses a UnivariateAnalysisStrategy
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str = None):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed. If None, analyze all features.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature)


# Example usage
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("D:/Github/Linear_Regression_A-Z/extracted_data/after_data_types_fixing.csv")

    # Analyzing all numerical features with histograms and box plots
    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_analysis(df)  # Analyze all numerical features

    # Analyzing a specific numerical feature
    analyzer.execute_analysis(df, "Price")  # Example numerical column

    # Analyzing all categorical features with bar plots
    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_analysis(df)  # Analyze all categorical features

    # Analyzing a specific categorical feature
    analyzer.execute_analysis(df, "Location")  # Example categorical column