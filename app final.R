#GCW1 — Shiny App: Tree-Based Methods for Heart Disease
#Dataset: Cleveland Heart Disease (UCI/Kaggle)

#LOAD LIBRARIES
library(shiny)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
library(dplyr)
library(corrplot)

# LOAD AND CLEAN DATA
heart <- read.csv(
  "processed.cleveland.data.csv",
  header = FALSE,
  na.strings = "?"
)

colnames(heart) <- c(
  "age","sex","cp","trestbps","chol",
  "fbs","restecg","thalach","exang",
  "oldpeak","slope","ca","thal","target"
)

#Convert target to binary
heart$target <- ifelse(heart$target > 0, 1, 0)

#Remove initial missing values
heart <- na.omit(heart)

#CONVERT VARIABLES
#Categorical variables
factor_vars <- c(
  "sex","cp","fbs","restecg",
  "exang","slope","ca","thal","target"
)
heart[factor_vars] <- lapply(heart[factor_vars], as.factor)

# Numeric variables (safe conversion)
numeric_vars <- c("age","trestbps","chol","thalach","oldpeak")
for (col in numeric_vars) {
  heart[[col]] <- suppressWarnings(as.numeric(as.character(heart[[col]])))
}

#Remove rows that became NA during conversion
heart <- na.omit(heart)

#TRAIN / TEST SPLIT
set.seed(123)
train_index <- sample(seq_len(nrow(heart)), size = 0.7 * nrow(heart))
train <- heart[train_index, ]
test  <- heart[-train_index, ]

#SHINY UI
ui <- fluidPage(
  titlePanel("Heart Disease Prediction Using Tree-Based Methods"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Model Controls"),
      
      selectInput(
        "model_type",
        "Choose Model:",
        choices = c("Decision Tree", "Random Forest")
      ),
      
      sliderInput(
        "cp",
        "Tree Complexity (Decision Tree)",
        min = 0.001,
        max = 0.1,
        value = 0.01,
        step = 0.005
      ),
      
      numericInput(
        "ntree",
        "Number of Trees (Random Forest)",
        value = 100,
        min = 50,
        max = 500
      )
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel(
          "Model Visualisation",
          plotOutput("modelPlot")
        ),
        
        tabPanel(
          "Model Performance",
          verbatimTextOutput("performance")
        ),
        
        tabPanel(
          "Explanation",
          h4("How Tree-Based Methods Work"),
          p("Decision trees classify patients using a series of simple rules. Each split separates patients based on features such as chest pain type, age, blood pressure, or cholesterol. By following the path from the root to a leaf node, you can understand which factors lead to higher or lower risk of heart disease."),
          p("Random forests improve on single decision trees by combining many trees. Each tree votes on the outcome, which increases accuracy, reduces overfitting, and stabilizes predictions."),
          
          h4("Key Predictors of Heart Disease"),
          verbatimTextOutput("insights")
        )
      )
    )
  )
)

#SHINY SERVER
server <- function(input, output) {
  
#Decision Tree Model
  tree_model <- reactive({
    rpart(
      target ~ .,
      data = train,
      method = "class",
      control = rpart.control(cp = input$cp)
    )
  })
  
#Random Forest Model
  rf_model <- reactive({
    randomForest(
      target ~ .,
      data = train,
      ntree = input$ntree,
      importance = TRUE
    )
  })
  

#MODEL VISUALISATION
  output$modelPlot <- renderPlot({
    if (input$model_type == "Decision Tree") {
      rpart.plot(
        tree_model(),
        type = 2,
        extra = 104,
        fallen.leaves = TRUE,
        main = "Decision Tree for Heart Disease"
      )
    } else {
      varImpPlot(
        rf_model(),
        main = "Variable Importance (Random Forest)"
      )
    }
  })
  
#MODEL PERFORMANCE
  output$performance <- renderPrint({
    if (input$model_type == "Decision Tree") {
      preds <- predict(tree_model(), test, type = "class")
    } else {
      preds <- predict(rf_model(), test)
    }
    
    cm <- table(Predicted = preds, Actual = test$target)
    
    cat("Confusion Matrix:\n\n")
    print(cm)
    
    accuracy <- sum(diag(cm)) / sum(cm)
    cat("\nOverall Accuracy:", round(accuracy, 3))
  })
  
#EXPLANATION

  output$insights <- renderPrint({
    if (input$model_type == "Random Forest") {
      imp <- importance(rf_model())
      imp <- imp[complete.cases(imp), , drop = FALSE] # remove NA rows safely
      top_vars <- names(sort(imp[, "MeanDecreaseGini"], decreasing = TRUE))[1:3]
      
      cat("The three most important predictors are:\n\n")
      print(top_vars)
      cat("\nThese variables have the strongest influence on the model's predictions.")
    } else {
      cat("Variable importance is only available for Random Forest.\n\n")
      cat("For Decision Trees, you can interpret important features by examining the tree structure in the 'Model Visualisation' tab. Features near the top of the tree generally have the largest influence on the prediction.")
    }
  })
}

# RUN SHINY APP
shinyApp(ui = ui, server = server)