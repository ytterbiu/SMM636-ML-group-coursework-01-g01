# app.R
# ==========================================
# GCW1 — FINAL (cleaner + more presentable Overview)
# Tree-Based Methods for Heart Disease (Explainer-style)
# R only, marking-proof, robust error handling
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

  app_dir <- tryCatch(
    normalizePath(dirname(sys.frame(1)$ofile)),
    error = function(e) NULL
  )
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
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
  )

  heart$target <- ifelse(heart$target > 0, 1, 0)

  numeric_vars <- c("age", "trestbps", "chol", "thalach", "oldpeak")
  for (col in numeric_vars) {
    heart[[col]] <- suppressWarnings(as.numeric(as.character(heart[[col]])))
  }

  factor_vars <- c(
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
    "target"
  )
  heart[factor_vars] <- lapply(heart[factor_vars], function(x) as.factor(x))

  heart <- na.omit(heart)
  heart$target <- factor(heart$target, levels = c("0", "1"))
  heart
}

heart <- load_heart_data(get_data_path())

# -------------------------
# 2) TRAIN / TEST (STRATIFIED)
# -------------------------
set.seed(123)
idx0 <- which(heart$target == "0")
idx1 <- which(heart$target == "1")

train0 <- sample(idx0, size = floor(0.70 * length(idx0)))
train1 <- sample(idx1, size = floor(0.70 * length(idx1)))

train_index <- c(train0, train1)
train <- heart[train_index, ]
test <- heart[-train_index, ]

# -------------------------
# 3) METRICS + SAFE ROC
# -------------------------
fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
}

compute_metrics <- function(actual, pred_class, pred_prob) {
  actual <- factor(actual, levels = c("0", "1"))
  pred_class <- factor(pred_class, levels = c("0", "1"))

  cm <- table(Predicted = pred_class, Actual = actual)
  acc <- sum(diag(cm)) / sum(cm)

  sens <- if (sum(cm[, "1"]) > 0) cm["1", "1"] / sum(cm[, "1"]) else NA
  spec <- if (sum(cm[, "0"]) > 0) cm["0", "0"] / sum(cm[, "0"]) else NA

  roc_obj <- NULL
  auc_val <- NA
  if (length(unique(actual)) == 2 && length(unique(pred_prob)) > 1) {
    roc_obj <- tryCatch(
      pROC::roc(
        actual,
        as.numeric(pred_prob),
        quiet = TRUE,
        levels = c("0", "1"),
        direction = "<"
      ),
      error = function(e) NULL
    )
    if (!is.null(roc_obj)) auc_val <- as.numeric(pROC::auc(roc_obj))
  }

  list(
    cm = cm,
    acc = acc,
    sens = sens,
    spec = spec,
    roc = roc_obj,
    auc = auc_val
  )
}

safe_plot_roc <- function(roc_obj, auc_val, title_text) {
  if (is.null(roc_obj) || is.na(auc_val)) {
    plot.new()
    text(
      0.5,
      0.5,
      "ROC not available.\n(Needs both classes + non-constant probabilities.)",
      cex = 1
    )
    title(title_text)
    return(invisible())
  }
  plot(
    roc_obj,
    legacy.axes = TRUE,
    main = paste0(title_text, " (AUC=", fmt(auc_val), ")")
  )
  abline(a = 0, b = 1, lty = 2)
}

# -------------------------
# 4) EXPLAINER HELPERS
# -------------------------
feature_dictionary <- function() {
  data.frame(
    feature = c(
      "age",
      "sex",
      "cp",
      "trestbps",
      "chol",
      "fbs",
      "restecg",
      "thalach",
      "exang",
      "oldpeak",
      "slope",
      "ca",
      "thal"
    ),
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
      "Slope of peak exercise ST segment (categorical)",
      "Major vessels (0–3) colored by fluoroscopy",
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

get_tree_path_rules_safe <- function(model, one_row) {
  tryCatch(
    {
      p <- path.rpart(model, newdata = one_row)
      rules <- unlist(p)
      if (length(rules) >= 1 && grepl("root", rules[1], ignore.case = TRUE)) {
        rules <- rules[-1]
      }
      pretty_rule(rules)
    },
    error = function(e) character(0)
  )
}

get_prob1_safe <- function(prob_matrix) {
  cn <- colnames(prob_matrix)
  if (!is.null(cn) && "1" %in% cn) {
    return(prob_matrix[, "1"])
  }
  if (ncol(prob_matrix) >= 2) {
    return(prob_matrix[, ncol(prob_matrix)])
  }
  as.numeric(prob_matrix[, 1])
}

# -------------------------
# 5) UI (NEATER OVERVIEW)
# -------------------------
theme_gc <- bs_theme(
  version = 5,
  bootswatch = "cosmo",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter"),
  # You can still override specific colors from the 'cosmo' theme:
  primary = "#EA4C89",
  bg = "#ffffff",
  fg = "#333333"
)

ui <- page_navbar(
  title = "Heart Disease — Tree-Based Methods",
  theme = theme_gc,

  # ================
  # OVERVIEW
  # ================
  nav_panel(
    "Overview",
    layout_column_wrap(
      width = 1,

      card(
        class = "shadow-sm",
        style = "padding: 20px;",
        h3(
          "Heart Disease Risk Prediction Dashboard",
          style = "margin-top: 0; font-weight: 600;"
        ),
        p(
          "This interactive dashboard demonstrates how tree-based machine learning models can be used to predict the presence of heart disease."
        ),
        p(
          "The application compares a ",
          tags$b("Decision Tree"),
          " (transparent, rule-based) with a ",
          tags$b("Random Forest"),
          " (ensemble method improving stability)."
        ),
        tags$hr(style = "margin-top: 15px; margin-bottom: 15px;"),
        p(
          tags$b("Objective: "),
          "Evaluate interpretability, predictive performance, and the trade-off between sensitivity and specificity using classification thresholds."
        )
      ),

      layout_column_wrap(
        width = 1 / 3,
        value_box("Observations", nrow(heart)),
        value_box("Predictor Variables", ncol(heart) - 1),
        value_box(
          "Prevalence (Disease = 1)",
          paste0(round(mean(as.numeric(as.character(heart$target))), 3))
        )
      ),

      layout_column_wrap(
        width = 1 / 2,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Methodological Focus")),
          tags$ul(
            tags$li("Rule-based model interpretation (Decision Tree splits)."),
            tags$li(
              "Ensemble learning and variable importance (Random Forest)."
            ),
            tags$li(
              "Threshold adjustment to examine classification trade-offs."
            ),
            tags$li(
              "Evaluation via Accuracy, Sensitivity, Specificity and AUC."
            )
          )
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("How to Navigate the Dashboard")),
          tags$ol(
            tags$li("Explore the dataset and variable distributions."),
            tags$li("Adjust tree parameters and observe structural changes."),
            tags$li("Use 'Walk a patient' to interpret rule-based decisions."),
            tags$li("Compare performance with Random Forest metrics.")
          )
        )
      ),

      card(
        class = "shadow-sm",
        card_header(tags$strong("Variable Definitions")),
        DTOutput("dict_tbl")
      )
    )
  ), # ✅ IMPORTANT COMMA HERE (you were missing this)

  # ================
  # The problem
  # ================
  nav_panel(
    "Case Study Background",
    layout_sidebar(
      sidebar = sidebar(
        h5("Patient inputs"),
        actionButton("btn_explain", "Explain my path", class = "btn-primary"),
        hr(),
        helpText(
          "Click the button to generate the prediction and the exact decision rules used by the tree."
        )
      ),
      layout_column_wrap(
        width = 1,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Prediction summary")),
          uiOutput("patient_summary_ui")
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("Step-by-step decision path")),
          uiOutput("patient_path_ui")
        )
      )
    )
  ),

  # ================
  # DATA
  # ================
  nav_panel(
    "Explore the data",
    layout_column_wrap(
      width = 1,
      card(
        class = "shadow-sm",
        card_header(tags$strong("Dataset preview")),
        div(style = "height: 420px; overflow-y: auto;", DTOutput("data_tbl"))
      ),
      layout_column_wrap(
        width = 1 / 2,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Target balance")),
          plotOutput("plot_target", height = 280)
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("Correlation (numeric only)")),
          plotOutput("plot_corr", height = 280)
        )
      )
    )
  ),

  # ================
  # TREE
  # ================
  nav_panel(
    "Decision tree",
    layout_sidebar(
      sidebar = sidebar(
        h5("Tree controls"),
        sliderInput(
          "tree_cp",
          "Complexity (cp)",
          min = 0.001,
          max = 0.08,
          value = 0.01,
          step = 0.001
        ),
        sliderInput(
          "tree_maxdepth",
          "Max depth",
          min = 1,
          max = 10,
          value = 4,
          step = 1
        ),
        sliderInput(
          "tree_minsplit",
          "Min split",
          min = 2,
          max = 60,
          value = 20,
          step = 1
        ),
        hr(),
        h5("Decision rule threshold"),
        sliderInput(
          "threshold",
          "Probability cutoff for class=1",
          min = 0.05,
          max = 0.95,
          value = 0.50,
          step = 0.05
        ),
        helpText(
          "Lower cutoff = more positives predicted (higher sensitivity)."
        )
      ),
      layout_column_wrap(
        width = 1,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Decision tree (flowchart of rules)")),
          plotOutput("plot_tree", height = 560)
        )
      ),
      layout_column_wrap(
        width = 1 / 2,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Tree performance")),
          verbatimTextOutput("tree_perf")
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("ROC curve")),
          plotOutput("plot_tree_roc", height = 320)
        )
      )
    )
  ),

  # ================
  # PATIENT WALK
  # ================
  nav_panel(
    "Walk a patient",
    layout_sidebar(
      sidebar = sidebar(
        h5("Patient inputs"),
        numericInput("p_age", "Age", value = 55, min = 1, max = 120),
        selectInput(
          "p_sex",
          "Sex (0=female, 1=male)",
          choices = levels(train$sex),
          selected = levels(train$sex)[1]
        ),
        selectInput(
          "p_cp",
          "Chest pain type (cp)",
          choices = levels(train$cp),
          selected = levels(train$cp)[1]
        ),
        numericInput(
          "p_trestbps",
          "Resting BP (trestbps)",
          value = 130,
          min = 50,
          max = 260
        ),
        numericInput(
          "p_chol",
          "Cholesterol (chol)",
          value = 240,
          min = 50,
          max = 700
        ),
        selectInput(
          "p_fbs",
          "Fasting blood sugar >120 (fbs)",
          choices = levels(train$fbs),
          selected = levels(train$fbs)[1]
        ),
        selectInput(
          "p_restecg",
          "Resting ECG (restecg)",
          choices = levels(train$restecg),
          selected = levels(train$restecg)[1]
        ),
        numericInput(
          "p_thalach",
          "Max heart rate (thalach)",
          value = 150,
          min = 50,
          max = 260
        ),
        selectInput(
          "p_exang",
          "Exercise angina (exang)",
          choices = levels(train$exang),
          selected = levels(train$exang)[1]
        ),
        numericInput(
          "p_oldpeak",
          "Oldpeak",
          value = 1.0,
          min = 0,
          max = 10,
          step = 0.1
        ),
        selectInput(
          "p_slope",
          "ST slope (slope)",
          choices = levels(train$slope),
          selected = levels(train$slope)[1]
        ),
        selectInput(
          "p_ca",
          "Major vessels (ca)",
          choices = levels(train$ca),
          selected = levels(train$ca)[1]
        ),
        selectInput(
          "p_thal",
          "Thal (thal)",
          choices = levels(train$thal),
          selected = levels(train$thal)[1]
        ),
        actionButton("btn_explain", "Explain my path", class = "btn-primary"),
        hr(),
        helpText(
          "Click the button to generate the prediction and the exact decision rules used by the tree."
        )
      ),
      layout_column_wrap(
        width = 1,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Prediction summary")),
          uiOutput("patient_summary_ui")
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("Step-by-step decision path")),
          uiOutput("patient_path_ui")
        )
      )
    )
  ),

  # ================
  # RF
  # ================
  nav_panel(
    "Random forest",
    layout_sidebar(
      sidebar = sidebar(
        h5("Forest controls"),
        sliderInput(
          "rf_ntree",
          "Number of trees (ntree)",
          min = 50,
          max = 800,
          value = 300,
          step = 25
        ),
        sliderInput(
          "rf_mtry",
          "mtry (features per split)",
          min = 1,
          max = ncol(train) - 1,
          value = max(1, floor(sqrt(ncol(train) - 1))),
          step = 1
        )
      ),
      layout_column_wrap(
        width = 1 / 2,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Variable importance (Top 10)")),
          plotOutput("plot_rf_imp", height = 380)
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("Random forest performance")),
          verbatimTextOutput("rf_perf")
        )
      ),
      layout_column_wrap(
        width = 1,
        card(
          class = "shadow-sm",
          card_header(tags$strong("ROC curve (Random Forest)")),
          plotOutput("plot_rf_roc", height = 320)
        )
      )
    )
  ),

  # ================
  # XTBoosted
  # ================
  nav_panel(
    "XT Boosted",
    layout_sidebar(
      sidebar = sidebar(
        h5("Patient inputs"),
        actionButton("btn_explain", "Explain my path", class = "btn-primary"),
        hr(),
        helpText(
          "Click the button to generate the prediction and the exact decision rules used by the tree."
        )
      ),
      layout_column_wrap(
        width = 1,
        card(
          class = "shadow-sm",
          card_header(tags$strong("Prediction summary")),
          uiOutput("patient_summary_ui")
        ),
        card(
          class = "shadow-sm",
          card_header(tags$strong("Step-by-step decision path")),
          uiOutput("patient_path_ui")
        )
      )
    )
  )
)

# -------------------------
# 6) SERVER
# -------------------------
server <- function(input, output, session) {
  output$dict_tbl <- DT::renderDT(
    {
      DT::datatable(
        feature_dictionary(),
        options = list(pageLength = 8, dom = "tip"),
        rownames = FALSE
      )
    },
    server = FALSE
  )

  output$data_tbl <- DT::renderDT(
    {
      DT::datatable(
        heart,
        options = list(pageLength = 10, scrollX = TRUE),
        rownames = FALSE
      )
    },
    server = FALSE
  )

  output$plot_target <- renderPlot({
    ggplot(heart, aes(x = target)) +
      geom_bar() +
      labs(x = "Target (0=no disease, 1=disease)", y = "Count") +
      theme_minimal(base_size = 13)
  })

  output$plot_corr <- renderPlot({
    num_df <- heart %>% select(where(is.numeric))
    if (ncol(num_df) < 2) {
      return(NULL)
    }
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

  # ---- Decision tree
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
      main = ""
    )
  })

  tree_perf_obj <- reactive({
    prob_mat <- predict(tree_model(), test, type = "prob")
    prob1 <- get_prob1_safe(prob_mat)

    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    pred_class <- factor(pred_class, levels = c("0", "1"))
    compute_metrics(test$target, pred_class, prob1)
  })

  output$tree_perf <- renderPrint({
    m <- tree_perf_obj()
    cat("Threshold:", input$threshold, "\n\n")
    cat("Confusion Matrix:\n\n")
    print(m$cm)
    cat(
      "\nAccuracy:",
      fmt(m$acc),
      "\nSensitivity:",
      fmt(m$sens),
      "\nSpecificity:",
      fmt(m$spec),
      "\nAUC:",
      fmt(m$auc),
      "\n"
    )
  })

  output$plot_tree_roc <- renderPlot({
    m <- tree_perf_obj()
    safe_plot_roc(m$roc, m$auc, "Decision Tree ROC")
  })

  # ---- Patient walkthrough (robust)
  build_patient_row <- function() {
    one <- train[1, , drop = FALSE]

    one$age <- as.numeric(input$p_age)
    one$sex <- factor(input$p_sex, levels = levels(train$sex))
    one$cp <- factor(input$p_cp, levels = levels(train$cp))
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
    tryCatch(
      {
        one <- build_patient_row()

        prob_mat <- predict(tree_model(), one, type = "prob")
        prob1 <- get_prob1_safe(prob_mat)

        pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
        rules <- get_tree_path_rules_safe(tree_model(), one)

        patient_result(list(
          prob1 = prob1,
          pred_class = pred_class,
          rules = rules
        ))
      },
      error = function(e) {
        showNotification(
          paste("Could not compute patient explanation:", e$message),
          type = "error",
          duration = 8
        )
        patient_result(list(
          prob1 = NA,
          pred_class = "NA",
          rules = character(0)
        ))
      }
    )
  })

  output$patient_summary_ui <- renderUI({
    res <- patient_result()
    if (is.null(res)) {
      return(tags$p(
        "Enter details and click ",
        tags$b("Explain my path"),
        " to generate the explanation."
      ))
    }

    label <- if (is.na(res$prob1)) {
      "Not available"
    } else if (res$pred_class == "1") {
      "Higher risk (class=1)"
    } else {
      "Lower risk (class=0)"
    }

    tags$div(
      tags$p(tags$b("Probability of disease (class=1): "), fmt(res$prob1)),
      tags$p(tags$b("Threshold used: "), input$threshold),
      tags$p(tags$b("Predicted class: "), label)
    )
  })

  output$patient_path_ui <- renderUI({
    res <- patient_result()
    if (is.null(res)) {
      return(tags$p("No path yet. Click 'Explain my path'."))
    }

    if (length(res$rules) == 0) {
      return(tags$div(
        tags$p("No rules extracted."),
        tags$p(
          "Try lowering cp or increasing max depth so the tree has more splits."
        )
      ))
    }

    tags$ol(lapply(res$rules, tags$li))
  })

  # ---- Random forest
  rf_model <- reactive({
    randomForest(
      target ~ .,
      data = train,
      ntree = input$rf_ntree,
      mtry = input$rf_mtry,
      importance = TRUE
    )
  })

  rf_perf_obj <- reactive({
    prob_mat <- predict(rf_model(), test, type = "prob")
    prob1 <- get_prob1_safe(prob_mat)

    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    pred_class <- factor(pred_class, levels = c("0", "1"))
    compute_metrics(test$target, pred_class, prob1)
  })

  output$rf_perf <- renderPrint({
    m <- rf_perf_obj()
    cat("Threshold:", input$threshold, "\n\n")
    cat("Confusion Matrix:\n\n")
    print(m$cm)
    cat(
      "\nAccuracy:",
      fmt(m$acc),
      "\nSensitivity:",
      fmt(m$sens),
      "\nSpecificity:",
      fmt(m$spec),
      "\nAUC:",
      fmt(m$auc),
      "\n\n"
    )
    cat("OOB error (training RF):\n")
    print(tail(rf_model()$err.rate, 1))
  })

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

    ggplot(
      imp_df,
      aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)
    ) +
      geom_col() +
      coord_flip() +
      labs(
        x = "",
        y = "Mean Decrease Gini",
        title = "Top 10 predictors (Random Forest)"
      ) +
      theme_minimal(base_size = 13)
  })

  output$plot_rf_roc <- renderPlot({
    m <- rf_perf_obj()
    safe_plot_roc(m$roc, m$auc, "Random Forest ROC")
  })
}

shinyApp(ui, server)
