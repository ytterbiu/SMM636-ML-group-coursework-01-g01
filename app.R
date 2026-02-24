# app.R
# ==========================================
# GCW1 — BEST FINAL VERSION (R only, marking-proof)
# Tree-Based Methods for Heart Disease (Explainer-style)
# Inspired by: mlu-explain decision tree + jbkunst educational Shiny
#
# HOW TO RUN:
# 1) Put processed.cleveland.data.csv in the SAME folder as this app.R
# 2) Run: shiny::runApp()
# ==========================================

# -------------------------
# 0) LIBRARIES
# -------------------------
library(shiny)
library(bslib)
library(dplyr)
library(ggplot2)
library(DT)

library(rpart)
library(rpart.plot)
library(randomForest)
library(pROC)

# -------------------------
# 1) DATA: FIND + LOAD + CLEAN
# -------------------------
get_data_path <- function() {
  candidates <- c(
    "processed.cleveland.data.csv",
    "processed.cleveland.data.csv.csv",
    "processed.cleveland.data"
  )
  
  app_dir <- tryCatch(normalizePath(dirname(sys.frame(1)$ofile)), error = function(e) NULL)
  search_dirs <- unique(na.omit(c(app_dir, getwd())))
  
  for (d in search_dirs) {
    for (f in candidates) {
      p <- file.path(d, f)
      if (file.exists(p)) return(p)
    }
  }
  
  stop(
    "Dataset file not found.\n",
    "Place 'processed.cleveland.data.csv' in the same folder as app.R.\n",
    "Checked folders:\n- ",
    paste(search_dirs, collapse = "\n- "),
    call. = FALSE
  )
}

load_heart_data <- function(path) {
  heart <- read.csv(path, header = FALSE, na.strings = "?")
  
  colnames(heart) <- c(
    "age","sex","cp","trestbps","chol",
    "fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","target"
  )
  
  # Binary target: 0=no disease, 1=disease
  heart$target <- ifelse(heart$target > 0, 1, 0)
  
  # Safe numeric conversion
  numeric_vars <- c("age","trestbps","chol","thalach","oldpeak")
  for (col in numeric_vars) {
    heart[[col]] <- suppressWarnings(as.numeric(as.character(heart[[col]])))
  }
  
  # Factors (including target)
  factor_vars <- c("sex","cp","fbs","restecg","exang","slope","ca","thal","target")
  heart[factor_vars] <- lapply(heart[factor_vars], function(x) as.factor(x))
  
  heart <- na.omit(heart)
  heart$target <- factor(heart$target, levels = c("0","1"))
  
  heart
}

heart <- load_heart_data(get_data_path())

# -------------------------
# 2) TRAIN / TEST (STRATIFIED to avoid ROC errors)
# -------------------------
set.seed(123)
idx0 <- which(heart$target == "0")
idx1 <- which(heart$target == "1")

train0 <- sample(idx0, size = floor(0.70 * length(idx0)))
train1 <- sample(idx1, size = floor(0.70 * length(idx1)))

train_index <- c(train0, train1)
train <- heart[train_index, ]
test  <- heart[-train_index, ]

# -------------------------
# 3) METRICS + SAFE ROC
# -------------------------
fmt <- function(x, digits = 3) ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))

compute_metrics <- function(actual, pred_class, pred_prob) {
  actual <- factor(actual, levels = c("0","1"))
  pred_class <- factor(pred_class, levels = c("0","1"))
  
  cm <- table(Predicted = pred_class, Actual = actual)
  acc <- sum(diag(cm)) / sum(cm)
  
  sens <- if (sum(cm[, "1"]) > 0) cm["1","1"] / sum(cm[, "1"]) else NA
  spec <- if (sum(cm[, "0"]) > 0) cm["0","0"] / sum(cm[, "0"]) else NA
  
  roc_obj <- NULL
  auc_val <- NA
  if (length(unique(actual)) == 2 && length(unique(pred_prob)) > 1) {
    roc_obj <- tryCatch(
      pROC::roc(actual, as.numeric(pred_prob), quiet = TRUE, levels = c("0","1"), direction = "<"),
      error = function(e) NULL
    )
    if (!is.null(roc_obj)) auc_val <- as.numeric(pROC::auc(roc_obj))
  }
  
  list(cm = cm, acc = acc, sens = sens, spec = spec, roc = roc_obj, auc = auc_val)
}

safe_plot_roc <- function(roc_obj, auc_val, title_text) {
  if (is.null(roc_obj) || is.na(auc_val)) {
    plot.new()
    text(
      0.5, 0.5,
      "ROC not available.\n(Needs both classes in test set and non-constant probabilities.)",
      cex = 1
    )
    title(title_text)
    return(invisible())
  }
  plot(roc_obj, legacy.axes = TRUE, main = paste0(title_text, " (AUC=", fmt(auc_val), ")"))
  abline(a = 0, b = 1, lty = 2)
}

# -------------------------
# 4) EXPLAINER HELPERS (mlu-explain vibe)
# -------------------------
feature_dictionary <- function() {
  data.frame(
    feature = c("age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"),
    meaning = c(
      "Age (years)",
      "Sex (0=female, 1=male)",
      "Chest pain type (categorical code)",
      "Resting blood pressure (mm Hg)",
      "Serum cholesterol (mg/dl)",
      "Fasting blood sugar >120 mg/dl (1=true, 0=false)",
      "Resting ECG result (categorical code)",
      "Maximum heart rate achieved",
      "Exercise induced angina (1=yes, 0=no)",
      "ST depression induced by exercise relative to rest",
      "Slope of the peak exercise ST segment (categorical)",
      "Number of major vessels (0–3) colored by fluoroscopy",
      "Thalassemia test result (categorical code)"
    ),
    stringsAsFactors = FALSE
  )
}

pretty_rule <- function(x) {
  x <- gsub("<", " < ", x)
  x <- gsub(">", " > ", x)
  x <- gsub("=", " = ", x)
  x <- gsub("\\s+", " ", x)
  trimws(x)
}

get_tree_path_rules <- function(model, one_row) {
  p <- path.rpart(model, newdata = one_row)
  rules <- unlist(p)
  if (length(rules) >= 1 && grepl("root", rules[1], ignore.case = TRUE)) rules <- rules[-1]
  pretty_rule(rules)
}

# -------------------------
# 5) UI (Explainer layout)
# -------------------------
theme_gc <- bs_theme(
  version = 5,
  bootswatch = "flatly",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter")
)

ui <- page_navbar(
  title = "Heart Disease — Tree-Based Methods (Explainer)",
  theme = theme_gc,
  
  nav_panel(
    "Overview",
    layout_column_wrap(
      width = 1,
      card(
        card_header("Goal"),
        p("This app explains how decision trees and random forests can predict heart disease risk."),
        p("Use the decision tree to see transparent rules (like a flowchart), and the random forest to see how combining many trees improves performance.")
      ),
      layout_column_wrap(
        width = 1/3,
        value_box("Rows (cleaned)", nrow(heart)),
        value_box("Features", ncol(heart) - 1),
        value_box("Positive rate (target=1)", round(mean(as.numeric(as.character(heart$target))), 3))
      ),
      card(card_header("Feature dictionary"), DTOutput("dict_tbl"))
    )
  ),
  
  nav_panel(
    "Explore the data",
    layout_column_wrap(
      width = 1,
      card(
        card_header("Dataset preview"),
        div(style = "height: 420px; overflow-y: auto;", DTOutput("data_tbl"))
      ),
      layout_column_wrap(
        width = 1/2,
        card(card_header("Target balance"), plotOutput("plot_target", height = 280)),
        card(card_header("Correlation (numeric only)"), plotOutput("plot_corr", height = 280))
      )
    )
  ),
  
  nav_panel(
    "Decision tree (explainer)",
    layout_sidebar(
      sidebar = sidebar(
        h5("Tree controls"),
        sliderInput("tree_cp", "Complexity (cp)", min = 0.001, max = 0.08, value = 0.01, step = 0.001),
        sliderInput("tree_maxdepth", "Max depth", min = 1, max = 10, value = 4, step = 1),
        sliderInput("tree_minsplit", "Min split", min = 2, max = 60, value = 20, step = 1),
        hr(),
        h5("Explanation controls"),
        sliderInput("threshold", "Classification threshold (probability for class=1)", min = 0.05, max = 0.95, value = 0.50, step = 0.05),
        helpText("Lower threshold → more positives caught (higher sensitivity). Higher threshold → fewer false positives (higher specificity).")
      ),
      layout_column_wrap(
        width = 1,
        card(card_header("Decision tree"), plotOutput("plot_tree", height = 560))
      ),
      layout_column_wrap(
        width = 1/2,
        card(card_header("Performance (Tree)"), verbatimTextOutput("tree_perf")),
        card(card_header("ROC curve (Tree)"), plotOutput("plot_tree_roc", height = 320))
      )
    )
  ),
  
  nav_panel(
    "Walk a patient (step-by-step)",
    layout_sidebar(
      sidebar = sidebar(
        h5("Patient inputs (match dataset coding)"),
        numericInput("p_age", "Age", value = 55, min = 1, max = 120),
        selectInput("p_sex", "Sex (0=female, 1=male)", choices = levels(train$sex), selected = levels(train$sex)[1]),
        selectInput("p_cp", "Chest pain type (cp)", choices = levels(train$cp), selected = levels(train$cp)[1]),
        numericInput("p_trestbps", "Resting BP (trestbps)", value = 130, min = 50, max = 260),
        numericInput("p_chol", "Cholesterol (chol)", value = 240, min = 50, max = 700),
        selectInput("p_fbs", "Fasting blood sugar >120 (fbs)", choices = levels(train$fbs), selected = levels(train$fbs)[1]),
        selectInput("p_restecg", "Resting ECG (restecg)", choices = levels(train$restecg), selected = levels(train$restecg)[1]),
        numericInput("p_thalach", "Max heart rate (thalach)", value = 150, min = 50, max = 260),
        selectInput("p_exang", "Exercise angina (exang)", choices = levels(train$exang), selected = levels(train$exang)[1]),
        numericInput("p_oldpeak", "Oldpeak", value = 1.0, min = 0, max = 10, step = 0.1),
        selectInput("p_slope", "ST slope (slope)", choices = levels(train$slope), selected = levels(train$slope)[1]),
        selectInput("p_ca", "Major vessels (ca)", choices = levels(train$ca), selected = levels(train$ca)[1]),
        selectInput("p_thal", "Thal (thal)", choices = levels(train$thal), selected = levels(train$thal)[1]),
        actionButton("btn_explain", "Explain my path", class = "btn-primary"),
        hr(),
        helpText("This tab shows the exact rules applied by the decision tree (like a guided explanation).")
      ),
      layout_column_wrap(
        width = 1,
        card(card_header("Prediction summary"), verbatimTextOutput("patient_summary")),
        card(card_header("Step-by-step decision path (rules)"), verbatimTextOutput("patient_path"))
      )
    )
  ),
  
  nav_panel(
    "Random forest",
    layout_sidebar(
      sidebar = sidebar(
        h5("Forest controls"),
        sliderInput("rf_ntree", "Number of trees (ntree)", min = 50, max = 800, value = 300, step = 25),
        sliderInput(
          "rf_mtry", "mtry (features per split)",
          min = 1, max = ncol(train) - 1,
          value = max(1, floor(sqrt(ncol(train) - 1))),
          step = 1
        )
      ),
      layout_column_wrap(
        width = 1/2,
        card(card_header("Variable importance (Top 10)"), plotOutput("plot_rf_imp", height = 380)),
        card(card_header("Performance (RF)"), verbatimTextOutput("rf_perf"))
      ),
      layout_column_wrap(
        width = 1,
        card(card_header("ROC curve (RF)"), plotOutput("plot_rf_roc", height = 320))
      )
    )
  )
)

# -------------------------
# 6) SERVER
# -------------------------
server <- function(input, output, session) {
  
  # ---- Feature dictionary
  output$dict_tbl <- DT::renderDT({
    DT::datatable(
      feature_dictionary(),
      options = list(pageLength = 8, dom = "tip"),
      rownames = FALSE
    )
  }, server = FALSE)
  
  # ---- Data preview
  output$data_tbl <- DT::renderDT({
    DT::datatable(
      heart,
      options = list(pageLength = 10, scrollX = TRUE),
      rownames = FALSE
    )
  }, server = FALSE)
  
  output$plot_target <- renderPlot({
    ggplot(heart, aes(x = target)) +
      geom_bar() +
      labs(x = "Target (0=no disease, 1=disease)", y = "Count") +
      theme_minimal(base_size = 13)
  })
  
  output$plot_corr <- renderPlot({
    num_df <- heart %>% select(where(is.numeric))
    if (ncol(num_df) < 2) return(NULL)
    cmat <- cor(num_df, use = "pairwise.complete.obs")
    
    df_long <- as.data.frame(as.table(cmat))
    colnames(df_long) <- c("Var1", "Var2", "Corr")
    
    ggplot(df_long, aes(Var1, Var2, fill = Corr)) +
      geom_tile() +
      coord_equal() +
      labs(x = "", y = "") +
      theme_minimal(base_size = 12) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })
  
  # ---- Decision Tree Model
  tree_model <- reactive({
    rpart(
      target ~ .,
      data = train,
      method = "class",
      control = rpart.control(
        cp = input$tree_cp,
        maxdepth = input$tree_maxdepth,
        minsplit = input$tree_minsplit
      )
    )
  })
  
  output$plot_tree <- renderPlot({
    rpart.plot(
      tree_model(),
      type = 2,
      extra = 104,
      fallen.leaves = TRUE,
      main = "Decision Tree (flowchart of rules)"
    )
  })
  
  # ---- Tree performance (threshold adjustable)
  tree_perf_obj <- reactive({
    prob1 <- predict(tree_model(), test, type = "prob")[, "1"]
    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    pred_class <- factor(pred_class, levels = c("0","1"))
    compute_metrics(test$target, pred_class, prob1)
  })
  
  output$tree_perf <- renderPrint({
    m <- tree_perf_obj()
    cat("Threshold:", input$threshold, "\n\n")
    cat("Confusion Matrix:\n\n")
    print(m$cm)
    cat("\nAccuracy:", fmt(m$acc),
        "\nSensitivity:", fmt(m$sens),
        "\nSpecificity:", fmt(m$spec),
        "\nAUC:", fmt(m$auc), "\n")
  })
  
  output$plot_tree_roc <- renderPlot({
    m <- tree_perf_obj()
    safe_plot_roc(m$roc, m$auc, "Decision Tree ROC")
  })
  
  # =========================
  # PATIENT WALKTHROUGH (FIXED: never blank)
  # =========================
  
  # Default outputs (so the tab never loads blank)
  output$patient_summary <- renderPrint({
    cat("Enter patient details on the left, then click 'Explain my path'.\n\n")
    cat("You will see:\n")
    cat("- Probability of disease (class=1)\n")
    cat("- Predicted class using the threshold slider\n")
    cat("- The exact decision rules the tree applied\n")
  })
  
  output$patient_path <- renderPrint({
    cat("No path yet.\n")
    cat("Click 'Explain my path' to see the step-by-step rules.\n")
  })
  
  build_patient_row <- function() {
    one <- train[1, , drop = FALSE]
    
    one$age <- as.numeric(input$p_age)
    one$sex <- factor(input$p_sex, levels = levels(train$sex))
    one$cp  <- factor(input$p_cp, levels = levels(train$cp))
    one$trestbps <- as.numeric(input$p_trestbps)
    one$chol <- as.numeric(input$p_chol)
    one$fbs <- factor(input$p_fbs, levels = levels(train$fbs))
    one$restecg <- factor(input$p_restecg, levels = levels(train$restecg))
    one$thalach <- as.numeric(input$p_thalach)
    one$exang <- factor(input$p_exang, levels = levels(train$exang))
    one$oldpeak <- as.numeric(input$p_oldpeak)
    one$slope <- factor(input$p_slope, levels = levels(train$slope))
    one$ca <- factor(input$p_ca, levels = levels(train$ca))
    one$thal <- factor(input$p_thal, levels = levels(train$thal))
    
    one
  }
  
  patient_result <- reactiveVal(NULL)
  
  observeEvent(input$btn_explain, {
    one <- build_patient_row()
    
    prob1 <- predict(tree_model(), one, type = "prob")[, "1"]
    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    rules <- get_tree_path_rules(tree_model(), one)
    
    patient_result(list(prob1 = prob1, pred_class = pred_class, rules = rules))
  })
  
  output$patient_summary <- renderPrint({
    res <- patient_result()
    if (is.null(res)) {
      cat("Enter patient details on the left, then click 'Explain my path'.\n")
      return()
    }
    
    cat("Decision Tree Prediction\n")
    cat("------------------------\n")
    cat("Probability of disease (class=1):", fmt(res$prob1), "\n")
    cat("Threshold used:", input$threshold, "\n")
    cat("Predicted class:", res$pred_class, "(1=higher risk, 0=lower risk)\n\n")
    cat("How to interpret:\n")
    cat("- The tree asks questions in order.\n")
    cat("- Early questions (near the top) usually matter most.\n")
  })
  
  output$patient_path <- renderPrint({
    res <- patient_result()
    if (is.null(res)) {
      cat("No path yet. Click 'Explain my path'.\n")
      return()
    }
    
    cat("Step-by-step rule path\n")
    cat("----------------------\n")
    if (length(res$rules) == 0) {
      cat("No rules extracted.\n")
      cat("Try lowering cp or increasing max depth so the tree has more splits.\n")
    } else {
      for (i in seq_along(res$rules)) cat(sprintf("%d) %s\n", i, res$rules[i]))
    }
  })
  
  # ---- Random Forest
  rf_model <- reactive({
    randomForest(
      target ~ .,
      data = train,
      ntree = input$rf_ntree,
      mtry  = input$rf_mtry,
      importance = TRUE
    )
  })
  
  rf_perf_obj <- reactive({
    prob1 <- predict(rf_model(), test, type = "prob")[, "1"]
    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    pred_class <- factor(pred_class, levels = c("0","1"))
    compute_metrics(test$target, pred_class, prob1)
  })
  
  output$rf_perf <- renderPrint({
    m <- rf_perf_obj()
    cat("Threshold:", input$threshold, "\n\n")
    cat("Confusion Matrix:\n\n")
    print(m$cm)
    cat("\nAccuracy:", fmt(m$acc),
        "\nSensitivity:", fmt(m$sens),
        "\nSpecificity:", fmt(m$spec),
        "\nAUC:", fmt(m$auc), "\n\n")
    cat("OOB error (training RF):\n")
    print(tail(rf_model()$err.rate, 1))
  })
  
  # RF importance (fixed: no slice_head / n() issues)
  output$plot_rf_imp <- renderPlot({
    imp <- importance(rf_model())
    if (is.null(dim(imp))) {
      plot.new()
      text(0.5, 0.5, "Importance not available.")
      return(invisible())
    }
    
    imp_df <- data.frame(
      Feature = rownames(imp),
      MeanDecreaseGini = imp[, "MeanDecreaseGini"],
      stringsAsFactors = FALSE
    )
    imp_df <- imp_df[order(imp_df$MeanDecreaseGini, decreasing = TRUE), ]
    imp_df <- head(imp_df, 10)
    
    ggplot(imp_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
      geom_col() +
      coord_flip() +
      labs(x = "", y = "Mean Decrease Gini", title = "Top 10 predictors (Random Forest)") +
      theme_minimal(base_size = 13)
  })
  
  output$plot_rf_roc <- renderPlot({
    m <- rf_perf_obj()
    safe_plot_roc(m$roc, m$auc, "Random Forest ROC")
  })
}

shinyApp(ui, server)