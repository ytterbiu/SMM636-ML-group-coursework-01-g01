# Heart Disease Prediction Shiny App

This project implements tree-based machine learning models to predict the
presence of heart disease using the Cleveland dataset. This was part of a group
project at Bayes Business School (Term 2 2025-26) for module SMM636 Machine
Learning.

> [!NOTE]
> Live applications are available via:
> [Basic application](https://3enji.shinyapps.io/smm636-a01-tree-based-methods/)
> [More specialised decision tree](https://3enji.shinyapps.io/smm636-a01-tree-based-methods-standalone-tab4/)

## Models Used

1. Decision Trees — a single, readable set of rules.
2. Random Forests — many trees trained on bootstrapped samples (“wisdom of
   crowds”).
3. XGBoost — boosted trees trained sequentially, with explanations supported
   using **SHAP** values.

## What the app includes

- Presentation-style flow: starts with the problem and ends with the practical
  takeaway, rather than just dumping outputs.
- Interactive visuals: lets users play with the idea of splits and trees before
  moving to the fitted models.
- Tuning controls: sliders for tree complexity, number of trees (`ntree`), and
  boosting rounds to name a few.
- Model evaluation: ROC curves and confusion matrices for test performance, plus
  SHAP summary / beeswarm plots for XGBoost interpretation.
- UI built with `bslib`: simple, responsive layout suitable for a
  “client-facing” demo.

## Dataset

Cleveland Heart Disease Dataset (UCI / Kaggle)

## How to Run

1. Download the repository
2. Open `app.R` in RStudio
3. Ensure the dataset file is in the same folder
4. Click "Run App"

## Requirements

Install dependencies with:

```r
install.packages(c(
  "shiny",
  "bslib",
  "dplyr",
  "ggplot2",
  "DT",
  "rpart",
  "rpart.plot",
  "randomForest",
  "pROC",
  "tidymodels",
  "xgboost",
  "DiagrammeR",
  "shapviz"
))
```



