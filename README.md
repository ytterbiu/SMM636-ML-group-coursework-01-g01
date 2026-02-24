# Heart Disease Prediction Shiny App

This project implements tree-based machine learning models to predict the presence of heart disease using the Cleveland dataset.

## Models Used
- Decision Tree
- Random Forest

## Features
- Interactive model selection
- Adjustable model parameters
- Visualisation of decision trees
- Variable importance analysis
- Performance evaluation using a confusion matrix

## Dataset
Cleveland Heart Disease Dataset (UCI / Kaggle)

## How to Run

1. Download the repository
2. Open `app.R` in RStudio
3. Ensure the dataset file is in the same folder
4. Click "Run App"

## Requirements

Install required packages in R:

install.packages(c(
  "shiny",
  "rpart",
  "rpart.plot",
  "randomForest",
  "ggplot2",
  "dplyr",
  "corrplot"
))

}



#RUN SHINY APPLICATION
shinyApp(ui = ui, server = server)
