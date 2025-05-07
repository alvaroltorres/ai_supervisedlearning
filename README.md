# AI Supervised Learning

## Dataset: Steel Plate Defect Prediction

Dataset used was generated from a deep learning model trained on the Steel Plates Faults dataset from UCI.
> https://www.kaggle.com/competitions/playground-series-s4e3/overview

The original dataset of steel plate faults is classified into 7 different types. The goal was to train machine learning for automatic pattern recognition.
> https://archive.ics.uci.edu/dataset/198/steel+plates+faults

### Target variables:

The target variables are 7 types of defects: Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps and Other_Faults.

### Features:

**I. Bounding Box & Basic Geometry:**
These describe the rectangle that encloses the defect.

1.  **`X_Minimum`**: The smallest X-coordinate of the defect's bounding box.
2.  **`X_Maximum`**: The largest X-coordinate of the defect's bounding box.
3.  **`Y_Minimum`**: The smallest Y-coordinate of the defect's bounding box.
4.  **`Y_Maximum`**: The largest Y-coordinate of the defect's bounding box.
5.  **`Pixels_Areas`**: The total number of pixels that make up the defect (its area).

**II. Perimeter:**
These describe the length of the defect's boundary.

6.  **`X_Perimeter`**: The length of the defect's boundary along the X-axis (horizontal extent).
7.  **`Y_Perimeter`**: The length of the defect's boundary along the Y-axis (vertical extent).

**III. Luminosity (Brightness/Intensity):**
These relate to the brightness values of the pixels within the defect.

8.  **`Sum_of_Luminosity`**: The sum of the intensity values of all pixels within the defect.
9.  **`Minimum_of_Luminosity`**: The intensity value of the darkest pixel within the defect.
10. **`Maximum_of_Luminosity`**: The intensity value of the brightest pixel within the defect.
11. **`Luminosity_Index`**: A normalized measure of the average luminosity of the defect, often comparing it to the background or overall plate luminosity. It indicates if the defect is generally darker or lighter.

**IV. Plate & Conveyer Characteristics:**

12. **`Length_of_Conveyer`**: The length of the conveyer belt section from which the image was captured.
13. **`TypeOfSteel_A300`**: A binary indicator (0 or 1) if the steel is of type A300.
14. **`TypeOfSteel_A400`**: A binary indicator (0 or 1) if the steel is of type A400. (If `TypeOfSteel_A300` is 1, this will be 0, and vice-versa).
15. **`Steel_Plate_Thickness`**: The thickness of the steel plate.

**V. Shape, Form, and Location Indices (Derived Ratios & Measures):**
These are normalized values (between 0 and 1) that describe the defect's shape, compactness, and position.

16. **`Edges_Index`**: A measure of the "compactness" or "edginess." It often relates the perimeter to the area. A higher value might mean a more complex or less compact shape.
17. **`Empty_Index`**: The proportion of the defect's bounding box that is *not* filled by the defect itself. It measures how much "empty space" is within the box.
18. **`Square_Index`**: A measure of how close the defect's bounding box is to a perfect square.
19. **`Outside_X_Index`**: The proportion of the defect's area that lies outside a central X-region of the plate or its overall X-span.
20. **`Edges_X_Index`**: Ratio of the defect's perimeter in the X-direction relative to its overall width.
21. **`Edges_Y_Index`**: Ratio of the defect's perimeter in the Y-direction relative to its overall height.
22. **`Outside_Global_Index`**: A value indicating if the defect is close to the global edges of the plate (0 if not, 0.5 if near a corner, 1.0 if near one edge).
23. **`Orientation_Index`**: Indicates the primary orientation of the defect. For instance, 0 might mean aligned with the rolling direction, 1 perpendicular, and 0.5 for other orientations. This helps distinguish between elongated defects along different axes.

**VI. Log-Transformed Features:**
Log transformations are used to handle skewed data and make distributions more symmetrical.

24. **`LogOfAreas`**: The natural logarithm of `Pixels_Areas`.
25. **`Log_X_Index`**: The natural logarithm of the defect's width (likely `X_Maximum - X_Minimum`).
26. **`Log_Y_Index`**: The natural logarithm of the defect's height (likely `Y_Maximum - Y_Minimum`).

**VII. Sigmoid Transformed Feature:**

27. **`SigmoidOfAreas`**: The `Pixels_Areas` value transformed by a sigmoid function. This squashes the area values into a specific range (often 0 to 1) and can help in modeling.

### Supervised learning problem:

Our objective is to predict the probability of each of the 7 binary targets on a given random steel plate that has 27 features.

### Instructions on how to use program:

If you haven't the necessary libraries installed run:
> pip install pandas numpy matplotlib seaborn scikit-learn

To run the EDA:
> python eda.py

Project developed by:
- Tomás Ferreira de Oliveira
- Diogo Miguel Fernandes Ferreira
- Álvaro Luís Dias Amaral Alvim Torres
