# Time Series Sales Forecasting Task

## Objective

Build a machine learning model to forecast product sales across multiple retail locations using historical time series data.

## Task Description

You are provided with historical sales data for various product categories across different retail stores. Your goal is to predict future sales for each store-product combination.

The dataset includes:
- Historical sales transactions with dates
- Store identifiers and product categories
- Promotional information
- Additional contextual data (store metadata, external economic indicators, calendar events)

## Data Structure

### Main Dataset
- **Training data**: Contains historical sales with features including store ID, product category, date, and promotional status
- **Test data**: Future time periods for which predictions are required
- **Target variable**: Unit sales (continuous values possible due to weight-based products)

### Supplementary Information
- Store characteristics and groupings
- Economic indicators relevant to the region
- Calendar events and holidays with special handling for transferred dates
- Transaction volume data

## Key Considerations

- Sales patterns may exhibit multiple seasonalities (daily, weekly, monthly, yearly)
- External economic factors can significantly impact sales
- Promotional activities influence demand
- Some dates may have special characteristics (paydays, events)
- Historical disruptions may have affected sales patterns

## Evaluation

Model performance will be evaluated using Root Mean Squared Logarithmic Error (RMSLE), which:
- Reduces the impact of large outliers
- Treats under-prediction and over-prediction asymmetrically
- Is appropriate for sales data with a wide range of values

## Submission Format

Predictions should be submitted as a CSV file with:
- ID column matching the test set
- Predicted sales values (non-negative)

## Technical Requirements

- Handle time series data properly to avoid temporal leakage
- Consider hierarchical structure (stores × products)
- Account for multiple seasonality patterns
- Process supplementary data appropriately