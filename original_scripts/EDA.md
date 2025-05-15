# EDA results

> For more details check documents under eda folder.

## 1. Data Loading
Shape of the training data (rows, columns): (19219, 35)

### First 5 rows of the data:

| id | X_Minimum | X_Maximum | Y_Minimum | Y_Maximum | Pixels_Areas | X_Perimeter | Y_Perimeter | ... | SigmoidOfAreas | Pastry | Z_Scratch | K_Scatch | Stains | Dirtiness | Bumps | Other_Faults |
|----|-----------|-----------|-----------|-----------|--------------|-------------|-------------|-----|----------------|--------|-----------|----------|--------|-----------|-------|--------------|
| 0  | 584       | 590       | 909972    | 909977    | 16           | 8           | 5           | ... | 0.1417         | 0      | 0         | 0        | 1      | 0         | 0     | 0            |
| 1  | 808       | 816       | 728350    | 728372    | 433          | 20          | 54          | ... | 0.9491         | 0      | 0         | 0        | 0      | 0         | 0     | 1            |
| 2  | 39        | 192       | 2212076   | 2212144   | 11388        | 705         | 420         | ... | 1.0000         | 0      | 0         | 1        | 0      | 0         | 0     | 0            |
| 3  | 781       | 789       | 3353146   | 3353173   | 210          | 16          | 29          | ... | 0.4025         | 0      | 0         | 1        | 0      | 0         | 0     | 0            |
| 4  | 1540      | 1560      | 618457    | 618502    | 521          | 72          | 67          | ... | 0.9998         | 0      | 0         | 0        | 0      | 0         | 0     | 1            |

### Column Names:
```
Index(['id', 'X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum',
       'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity',
       'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer',
       'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness',
       'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index',
       'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas',
       'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index',
       'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains',
       'Dirtiness', 'Bumps', 'Other_Faults'],
      dtype='object')
```

Identified 27 feature columns.  
Identified 7 target columns (defect types): Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults

## 2. Exploratory Data Analysis (EDA)

### 2.a Data Types and Non-Null Values

RangeIndex: 19219 entries, 0 to 19218

| # | Column                 | Non-Null Count | Dtype   |
|---|-----------------------|---------------|---------|
| 0 | id                     | 19219 non-null | int64   |
| 1 | X_Minimum              | 19219 non-null | int64   |
| 2 | X_Maximum              | 19219 non-null | int64   |
| 3 | Y_Minimum              | 19219 non-null | int64   |
| 4 | Y_Maximum              | 19219 non-null | int64   |
| 5 | Pixels_Areas           | 19219 non-null | int64   |
| 6 | X_Perimeter            | 19219 non-null | int64   |
| 7 | Y_Perimeter            | 19219 non-null | int64   |
| 8 | Sum_of_Luminosity      | 19219 non-null | int64   |
| 9 | Minimum_of_Luminosity  | 19219 non-null | int64   |
| 10 | Maximum_of_Luminosity  | 19219 non-null | int64   |
| 11 | Length_of_Conveyer     | 19219 non-null | int64   |
| 12 | TypeOfSteel_A300       | 19219 non-null | int64   |
| 13 | TypeOfSteel_A400       | 19219 non-null | int64   |
| 14 | Steel_Plate_Thickness  | 19219 non-null | int64   |
| 15 | Edges_Index            | 19219 non-null | float64 |
| 16 | Empty_Index            | 19219 non-null | float64 |
| 17 | Square_Index           | 19219 non-null | float64 |
| 18 | Outside_X_Index        | 19219 non-null | float64 |
| 19 | Edges_X_Index          | 19219 non-null | float64 |
| 20 | Edges_Y_Index          | 19219 non-null | float64 |
| 21 | Outside_Global_Index   | 19219 non-null | float64 |
| 22 | LogOfAreas             | 19219 non-null | float64 |
| 23 | Log_X_Index            | 19219 non-null | float64 |
| 24 | Log_Y_Index            | 19219 non-null | float64 |
| 25 | Orientation_Index      | 19219 non-null | float64 |
| 26 | Luminosity_Index       | 19219 non-null | float64 |
| 27 | SigmoidOfAreas         | 19219 non-null | float64 |
| 28 | Pastry                 | 19219 non-null | int64   |
| 29 | Z_Scratch              | 19219 non-null | int64   |
| 30 | K_Scatch               | 19219 non-null | int64   |
| 31 | Stains                 | 19219 non-null | int64   |
| 32 | Dirtiness              | 19219 non-null | int64   |
| 33 | Bumps                  | 19219 non-null | int64   |
| 34 | Other_Faults           | 19219 non-null | int64   |

dtypes: float64(13), int64(22)  
memory usage: 5.1 MB  
Missing values: None

### 2.b Missing Values per Column
No missing values found in the training data.

### 2.c Statistical Summary for Numerical Features

| Feature               | count   | mean            | std             | min        | 25%             | 50%             | 75%             | max             |
|-----------------------|---------|-----------------|-----------------|------------|-----------------|-----------------|-----------------|-----------------|
| X_Minimum             | 19219.0 | 7.098547e+02    | 5.315442e+02    | 0.0000     | 49.00000        | 7.770000e+02    | 1.152000e+03    | 1.705000e+03    |
| X_Maximum             | 19219.0 | 7.538576e+02    | 4.998366e+02    | 4.0000     | 214.00000       | 7.960000e+02    | 1.165000e+03    | 1.713000e+03    |
| Y_Minimum             | 19219.0 | 1.849756e+06    | 1.903554e+06    | 6712.0000  | 657468.00000    | 1.398169e+06    | 2.368032e+06    | 1.298766e+07    |
| Y_Maximum             | 19219.0 | 1.846605e+06    | 1.896295e+06    | 6724.0000  | 657502.00000    | 1.398179e+06    | 2.362511e+06    | 1.298769e+07    |
| Pixels_Areas          | 19219.0 | 1.683988e+03    | 3.730320e+03    | 6.0000     | 89.00000        | 1.680000e+02    | 6.530000e+02    | 1.526550e+05    |
| X_Perimeter           | 19219.0 | 9.565466e+01    | 1.778214e+02    | 2.0000     | 15.00000        | 2.500000e+01    | 6.400000e+01    | 7.553000e+03    |
| Y_Perimeter           | 19219.0 | 6.412410e+01    | 1.010542e+02    | 1.0000     | 14.00000        | 2.300000e+01    | 6.100000e+01    | 9.030000e+02    |
| Sum_of_Luminosity     | 19219.0 | 1.918467e+05    | 4.420247e+05    | 250.0000   | 9848.00000      | 1.823800e+04    | 6.797800e+04    | 1.159141e+07    |
| Minimum_of_Luminosity | 19219.0 | 8.480842e+01    | 2.880034e+01    | 0.0000     | 70.00000        | 9.000000e+01    | 1.050000e+02    | 1.960000e+02    |
| Maximum_of_Luminosity | 19219.0 | 1.286474e+02    | 1.419698e+01    | 39.0000    | 124.00000       | 1.270000e+02    | 1.350000e+02    | 2.530000e+02    |
| Length_of_Conveyer    | 19219.0 | 1.459351e+03    | 1.455687e+02    | 1227.0000  | 1358.00000      | 1.364000e+03    | 1.652000e+03    | 1.794000e+03    |
| TypeOfSteel_A300      | 19219.0 | 4.026744e-01    | 4.904490e-01    | 0.0000     | 0.00000         | 0.000000e+00    | 1.000000e+00    | 1.000000e+00    |
| TypeOfSteel_A400      | 19219.0 | 5.963370e-01    | 4.906442e-01    | 0.0000     | 0.00000         | 1.000000e+00    | 1.000000e+00    | 1.000000e+00    |
| Steel_Plate_Thickness | 19219.0 | 7.621312e+01    | 5.393196e+01    | 40.0000    | 40.00000        | 6.900000e+01    | 8.000000e+01    | 3.000000e+02    |
| Edges_Index           | 19219.0 | 3.529394e-01    | 3.189760e-01    | 0.0000     | 0.05860         | 2.385000e-01    | 6.561000e-01    | 9.952000e-01    |
| Empty_Index           | 19219.0 | 4.093095e-01    | 1.241435e-01    | 0.0000     | 0.31750         | 4.135000e-01    | 4.946000e-01    | 9.275000e-01    |
| Square_Index          | 19219.0 | 5.745204e-01    | 2.594359e-01    | 0.0083     | 0.37575         | 5.454000e-01    | 8.182000e-01    | 1.000000e+00    |
| Outside_X_Index       | 19219.0 | 3.060936e-02    | 4.730194e-02    | 0.0015     | 0.00660         | 9.500000e-03    | 1.910000e-02    | 6.651000e-01    |
| Edges_X_Index         | 19219.0 | 6.147495e-01    | 2.223913e-01    | 0.0144     | 0.45160         | 6.364000e-01    | 7.857000e-01    | 1.000000e+00    |
| Edges_Y_Index         | 19219.0 | 8.316521e-01    | 2.209660e-01    | 0.1050     | 0.65520         | 9.643000e-01    | 1.000000e+00    | 1.000000e+00    |
| Outside_Global_Index  | 19219.0 | 5.918986e-01    | 4.820500e-01    | 0.0000     | 0.00000         | 1.000000e+00    | 1.000000e+00    | 1.000000e+00    |
| LogOfAreas            | 19219.0 | 2.473475e+00    | 7.605751e-01    | 0.7782     | 1.94940         | 2.227900e+00    | 2.814900e+00    | 4.554300e+00    |
| Log_X_Index           | 19219.0 | 1.312667e+00    | 4.678477e-01    | 0.3010     | 1.00000         | 1.146100e+00    | 1.431400e+00    | 2.997300e+00    |
| Log_Y_Index           | 19219.0 | 1.389737e+00    | 4.055493e-01    | 0.0000     | 1.07920         | 1.322200e+00    | 1.707600e+00    | 4.033300e+00    |
| Orientation_Index     | 19219.0 | 1.027423e-01    | 4.876805e-01    | -0.9884    | -0.27270        | 1.111000e-01    | 5.294000e-01    | 9.917000e-01    |
| Luminosity_Index      | 19219.0 | -1.383818e-01   | 1.203440e-01    | -0.8850    | -0.19250        | -1.426000e-01   | -8.400000e-02   | 6.421000e-01    |
| SigmoidOfAreas        | 19219.0 | 5.719022e-01    | 3.322186e-01    | 0.1190     | 0.25320         | 4.729000e-01    | 9.994000e-01    | 1.000000e+00    |

Numerical features don't have missing values. Minimum and maximum values for most features seem plausible given their definitions (e.g., coordinates, binary flags, indices between 0-1 or specific ranges like Orientation_Index). Features like Pixels_Areas, X_Perimeter, Y_Perimeter, and Sum_of_Luminosity have max values significantly larger than their 75th percentile (e.g., Pixels_Areas 75th percentile is 653, max is 152655). This suggests the presence of some extreme values (potential outliers) or that these features have highly skewed distributions. No values appear "impossible" (like negative areas). Many features exhibit significant right skewness, where the mean is considerably larger than the median (50%). This is prominent in:
* Pixels_Areas (mean 1683 vs. median 168)
* X_Perimeter (mean 95 vs. median 25)
* Y_Perimeter (mean 64 vs. median 23)
* Sum_of_Luminosity (mean 1.9e5 vs. median 1.8e4)
* Outside_X_Index (mean 0.03 vs. median 0.0095)
* Even the log-transformed area features (LogOfAreas, Log_X_Index, Log_Y_Index) still show some right skew.

Other features show weaker skews (e.g., Y_Minimum, Y_Maximum, Steel_Plate_Thickness are right-skewed; X_Minimum, X_Maximum, Minimum_of_Luminosity are slightly left-skewed). Empty_Index, Edges_X_Index, Orientation_Index, Luminosity_Index appear relatively symmetrical (mean close to median). Edges_Y_Index and Outside_Global_Index show some left skew or concentration at higher values (median is 1.0 for Outside_Global_Index).

There's a vast difference in the scales (ranges and standard deviations) of the features. For example, Y_Minimum and Y_Maximum have standard deviations in the millions (e.g., 1.9e6) and values up to 1.29e7. Sum_of_Luminosity also has a very large scale. In contrast, indices like Edges_Index (std ~0.32) or binary TypeOfSteel_ flags (std ~0.49) operate on a much smaller scale (typically 0 to 1). LogOfAreas has a std of ~0.76, while Pixels_Areas has a std of ~3730.

The significant variation in feature scales strongly suggests that feature scaling like Normalization will be crucial.

### 2.d Target Variable Distribution (Defect Types)

| Defect Type  | Count |
|--------------|-------|
| Other_Faults | 6558  |
| Bumps        | 4763  |
| K_Scatch     | 3432  |
| Pastry       | 1466  |
| Z_Scratch    | 1150  |
| Stains       | 568   |
| Dirtiness    | 485   |

Number of samples with no defects: 818  
Total number of samples: 19219  
Number of samples with at least one defect: 18401

### 2.e Feature Visualizations

Check following plots to understand distributions and relationships between features: feature_histograms.png, feature_target_correlation_heatmap.png, feature_feature_correlation_heatmap.png, under eda folder.

High positive (red) or negative (blue) correlations between features might indicate multicollinearity (redundancy). Features highly correlated with target variables are potentially good predictors. Low correlation doesn't necessarily mean a feature is useless (non-linear relationships might exist).

**TODO**: Document Findings: As per the assignment, document these findings. Use tables (like the output of .describe() or .isnull().sum()) and embed the generated plots (histograms, count plots, heatmaps) in your report. Add textual analysis explaining what each plot/table shows (e.g., "Figure X shows the distribution of feature Y, which appears right-skewed", "Table Z indicates significant variation in feature scales, suggesting normalization will be necessary", "The target classes are imbalanced, with 'Other_Faults' being the most frequent and 'Stains' the least frequent").

**TODO**: Histograms: Understand the shape of each feature's distribution (e.g., skewed, normal-like).
Heatmaps: Identify highly correlated features (potential redundancy) and features strongly correlated with specific defects (potential predictors).

### 2.f Initial Preprocessing Considerations (Based on EDA)
Debug: min_std_val = 0.047301941353806255, max_range_ratio_positive = 1705000000.0, range_to_std_ratio = 42.47520700932078
- Feature scales seem relatively comparable or scaling might not be strictly required, but still often beneficial.
- No missing values detected. No imputation needed based on this check.
- Data types seem appropriate (mostly numerical features). No obvious categorical encoding needed unless identified otherwise.
- Target variables are multi-label binary indicators.
- Class imbalance observed in target variables. May need techniques like resampling (SMOTE, over/undersampling) or using appropriate metrics (Precision, Recall, F1, AUC) during modeling.