from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================
# Bivariate Analysis
# ==========================
class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe.
        """
        pass

class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """Plots the relationship between two numerical features using a scatter plot."""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """Plots the relationship between a categorical feature and a numerical feature using a box plot."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()

class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df, feature1, feature2)

# ==========================
# Multivariate Analysis
# ==========================
class MultivariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """Perform a comprehensive multivariate analysis."""
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        pass

class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """Generates and displays a correlation heatmap for numerical features."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """Generates and displays a pair plot for selected features."""
        sns.pairplot(df)
        plt.suptitle("Pair Plot of Selected Features", y=1.02)
        plt.show()

# ==========================
# Example Usage
# ==========================
if __name__ == "__main__":
    # Example usage of Bivariate Analysis
    # df = pd.read_csv('../extracted-data/your_data_file.csv')
    # bivariate_analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    # bivariate_analyzer.execute_analysis(df, 'Gr Liv Area', 'SalePrice')
    
    # bivariate_analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    # bivariate_analyzer.execute_analysis(df, 'Overall Qual', 'SalePrice')
    
    # Example usage of Multivariate Analysis
    # multivariate_analyzer = SimpleMultivariateAnalysis()
    # selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]
    # multivariate_analyzer.analyze(selected_features)
    pass
