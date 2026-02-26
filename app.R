# app.R
# ==========================================
# GCW1 — FINAL
# Tree-Based Methods for Heart Disease (Explainer-style)
# Patient-specific decision tree diagram with highlighted path (ALWAYS)
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

library(tidymodels)
library(xgboost)

library(DiagrammeR)

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
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
    "exang","oldpeak","slope","ca","thal","target"
  )
  
  heart$target <- ifelse(heart$target > 0, 1, 0)
  
  numeric_vars <- c("age","trestbps","chol","thalach","oldpeak")
  for (col in numeric_vars) {
    heart[[col]] <- suppressWarnings(as.numeric(as.character(heart[[col]])))
  }
  
  factor_vars <- c("sex","cp","fbs","restecg","exang","slope","ca","thal","target")
  heart[factor_vars] <- lapply(heart[factor_vars], function(x) as.factor(x))
  
  heart <- na.omit(heart)
  heart$target <- factor(heart$target, levels = c("0","1"))
  heart
}

heart <- load_heart_data(get_data_path())

heart.recipe <- heart |>
  recipe(target ~ .) |>
  step_dummy(all_nominal_predictors())

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
test  <- heart[-train_index, ]

# DMatrix for xgboost
Dtrain <- heart.recipe |> prep() |> bake(train) |> select(!target) |> xgb.DMatrix()
Dtest <- heart.recipe |> prep() |> bake(test) |> select(!target) |> xgb.DMatrix()

# -------------------------
# 3) METRICS + SAFE ROC
# -------------------------
fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
}

compute_metrics <- function(actual, pred_class, pred_prob) {
  actual <- factor(actual, levels = c("0","1"))
  pred_class <- factor(pred_class, levels = c("0","1"))
  
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
        levels = c("0","1"),
        direction = "<"
      ),
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
      "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
      "exang","oldpeak","slope","ca","thal"
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

get_prob1_safe <- function(prob_matrix) {
  cn <- colnames(prob_matrix)
  if (!is.null(cn) && "1" %in% cn) return(prob_matrix[, "1"])
  if (ncol(prob_matrix) >= 2) return(prob_matrix[, ncol(prob_matrix)])
  as.numeric(prob_matrix[, 1])
}

info_icon <- function(text) {
  tags$span(
    class = "info-btn",
    `data-bs-toggle` = "tooltip",
    `data-bs-placement` = "right",
    title = text,
    "i"
  )
}

plot_cm_heatmap <- function(cm, title_text = "") {
  df <- as.data.frame(cm)
  colnames(df) <- c("Predicted", "Actual", "Count")
  
  ggplot(df, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = Count), size = 5) +
    labs(title = title_text, x = "Actual", y = "Predicted") +
    theme_minimal(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold"),
      panel.grid = element_blank()
    )
}

# ---- Tree utilities
tree_has_splits <- function(model) {
  if (is.null(model) || is.null(model$frame)) return(FALSE)
  any(model$frame$var != "<leaf>")
}

get_leaf_node_id <- function(model, one_row) {
  as.integer(predict(model, one_row, type = "where"))
}

node_ancestors <- function(node_id) {
  out <- integer(0)
  while (!is.na(node_id) && node_id >= 1) {
    out <- c(out, node_id)
    if (node_id == 1) break
    node_id <- floor(node_id / 2)
  }
  unique(out)
}

plot_patient_tree_highlight <- function(model, one_row, main_title = "Patient-specific Decision Tree") {
  if (is.null(model) || is.null(model$frame)) {
    plot.new()
    text(0.5, 0.5, "Model not available.")
    return(invisible())
  }
  
  node_ids <- as.integer(rownames(model$frame))
  leaf <- tryCatch(get_leaf_node_id(model, one_row), error = function(e) NA_integer_)
  path_nodes <- if (!is.na(leaf)) node_ancestors(leaf) else integer(0)
  
  box_cols <- rep("#EFEFEF", length(node_ids))
  names(box_cols) <- as.character(node_ids)
  if (length(path_nodes) > 0) {
    hit <- intersect(as.character(path_nodes), names(box_cols))
    box_cols[hit] <- "#FFE3EE"
  }
  
  rpart.plot::prp(
    model,
    type = 2,
    extra = 104,
    fallen.leaves = TRUE,
    box.col = box_cols,
    branch.col = "#8A1C3D",
    shadow.col = "gray85",
    main = main_title
  )
  
  legend(
    "topleft",
    legend = c("Patient path", "Other nodes"),
    fill = c("#FFE3EE", "#EFEFEF"),
    border = c("#8A1C3D", "gray70"),
    bty = "n",
    cex = 0.9
  )
  invisible()
}

# -------------------------
# 5) UI
# -------------------------
theme_gc <- bs_theme(
  version = 5,
  bootswatch = "cosmo",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter"),
  primary = "#8A1C3D",
  bg = "#ffffff",
  fg = "#333333"
)

ui <- page_navbar(
  title = "Heart Disease — Tree-Based Methods",
  theme = theme_gc,
  
  header = tags$head(
    tags$style(
      HTML(
        "
        .navbar, .navbar .navbar-brand, .navbar .nav-link { font-weight: 600; }
        .btn-primary { font-weight: 600; }
        .info-btn{
          display:inline-flex;
          align-items:center;
          justify-content:center;
          width:18px;
          height:18px;
          margin-left:8px;
          border-radius:999px;
          background: var(--bs-primary);
          color: #fff;
          font-size:12px;
          line-height:18px;
          cursor: pointer;
          user-select: none;
        }
        .info-btn:hover{ filter: brightness(0.95); }
        "
      )
    ),
    tags$script(
      HTML(
        "
        document.addEventListener('DOMContentLoaded', function () {
          var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle=\"tooltip\"]'));
          tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
          });
        });
        "
      )
    )
  ),
  
  nav_panel(
    "Overview",
    layout_column_wrap(
      width = 1,
      card(
        class = "shadow-sm",
        style = "padding: 20px;",
        h3("Heart Disease Risk Prediction Dashboard", style = "margin-top: 0; font-weight: 600;"),
        p("This interactive dashboard demonstrates how tree-based machine learning models can be used to predict the presence of heart disease."),
        p("The application compares a ", tags$b("Decision Tree"), " (transparent, rule-based) with a ", tags$b("Random Forest"), " (ensemble method improving stability)."),
        tags$hr(style = "margin-top: 15px; margin-bottom: 15px;"),
        p(tags$b("Objective: "), "Evaluate interpretability, predictive performance, and the trade-off between sensitivity and specificity using classification thresholds.")
      ),
      layout_column_wrap(
        width = 1 / 3,
        value_box("Observations", nrow(heart)),
        value_box("Predictor Variables", ncol(heart) - 1),
        value_box("Prevalence (Disease = 1)", paste0(round(mean(as.numeric(as.character(heart$target))), 3)))
      ),
      card(class = "shadow-sm", card_header(tags$strong("Variable Definitions")), DTOutput("dict_tbl"))
    )
  ),
  
  # ✅ RESTORED: Case Study Background (with same patient-specific tree diagram)
  nav_panel(
    "Case Study Background",
    layout_sidebar(
      sidebar = sidebar(
        h5("Patient inputs"),
        actionButton("btn_explain_bg", "Explain my path", class = "btn-primary"),
        hr(),
        helpText("Click the button to generate the prediction and the highlighted decision tree path for the example patient.")
      ),
      layout_column_wrap(
        width = 1,
        card(class = "shadow-sm", card_header(tags$strong("Prediction summary")), uiOutput("patient_summary_ui_bg")),
        card(class = "shadow-sm", card_header(tags$strong("Patient-specific decision tree (highlighted path)")),
             plotOutput("plot_patient_tree_bg", height = 560))
      )
    )
  ),
  
  nav_panel(
    "Explore the data",
    layout_column_wrap(
      width = 1,
      card(class = "shadow-sm", card_header(tags$strong("Dataset preview")),
           div(style = "height: 420px; overflow-y: auto;", DTOutput("data_tbl"))),
      layout_column_wrap(
        width = 1 / 2,
        card(class = "shadow-sm", card_header(tags$strong("Target balance")), plotOutput("plot_target", height = 280)),
        card(class = "shadow-sm", card_header(tags$strong("Correlation (numeric only)")), plotOutput("plot_corr", height = 280))
      )
    )
  ),
  
  nav_panel(
    "Decision tree",
    layout_sidebar(
      sidebar = sidebar(
        h5("Tree controls"),
        sliderInput("tree_cp",
                    tagList("Complexity (cp)", info_icon("Controls pruning. Higher cp makes the tree simpler. Lower cp allows more splits.")),
                    min = 0.001, max = 0.08, value = 0.01, step = 0.001),
        sliderInput("tree_maxdepth",
                    tagList("Max depth", info_icon("Higher depth allows more splitting but can overfit.")),
                    min = 1, max = 10, value = 4, step = 1),
        sliderInput("tree_minsplit",
                    tagList("Min split", info_icon("Minimum observations required to split. Higher values reduce splits.")),
                    min = 2, max = 60, value = 20, step = 1),
        hr(),
        h5("Decision rule threshold"),
        sliderInput("threshold", "Probability cutoff for class=1", min = 0.05, max = 0.95, value = 0.50, step = 0.05),
        helpText("Lower cutoff = more positives predicted (higher sensitivity).")
      ),
      layout_column_wrap(
        width = 1,
        card(class = "shadow-sm", card_header(tags$strong("Decision tree")),
             plotOutput("plot_tree", height = 560))
      ),
      layout_column_wrap(
        width = 1 / 2,
        card(class = "shadow-sm", card_header(tags$strong("Tree performance")), uiOutput("tree_perf_ui")),
        card(class = "shadow-sm", card_header(tags$strong("ROC curve")), plotOutput("plot_tree_roc", height = 320))
      )
    )
  ),
  
  nav_panel(
    "Walk a patient",
    layout_sidebar(
      sidebar = sidebar(
        h5("Patient inputs"),
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
        helpText("This will show a decision tree with your patient path highlighted.")
      ),
      layout_column_wrap(
        width = 1,
        card(class = "shadow-sm", card_header(tags$strong("Prediction summary")), uiOutput("patient_summary_ui")),
        card(class = "shadow-sm", card_header(tags$strong("Patient-specific decision tree (highlighted path)")),
             plotOutput("plot_patient_tree", height = 560))
      )
    )
  ),
  
  nav_panel(
    "Random forest",
    layout_sidebar(
      sidebar = sidebar(
        h5("Forest controls"),
        sliderInput("rf_ntree",
                    tagList("Number of trees (ntree)", info_icon("More trees usually improves stability but increases run time.")),
                    min = 50, max = 800, value = 300, step = 25),
        sliderInput("rf_mtry",
                    tagList("mtry (features per split)", info_icon("How many predictors are sampled at each split.")),
                    min = 1, max = ncol(train) - 1, value = max(1, floor(sqrt(ncol(train) - 1))), step = 1)
      ),
      layout_column_wrap(
        width = 1 / 2,
        card(class = "shadow-sm", card_header(tags$strong("Variable importance (Top 10)")), plotOutput("plot_rf_imp", height = 380)),
        card(class = "shadow-sm", card_header(tags$strong("Random forest performance")), uiOutput("rf_perf_ui"))
      ),
      layout_column_wrap(
        width = 1,
        card(class = "shadow-sm", card_header(tags$strong("ROC curve (Random Forest)")), plotOutput("plot_rf_roc", height = 320))
      )
    )
  ),
  
  nav_panel(
    "XG Boosted",
    layout_sidebar(
      sidebar = sidebar(
        h5("XGBoost controls"),
        sliderInput("xgb_num_boost_round", "Number of Boosting rounds", min = 1, max = 10, value = 3, step = 1),
        sliderInput("xgb_max_depth", "Max Depth", min = 1, max = 10, value = 3, step = 1),
        sliderInput("xgb_eta", "Learning ", min = 0, max = 1, value = 1, step = 0.05),
        actionButton("btn_train_xgb", "Train XGBoost", class = "btn-primary"),
        hr(),
        sliderInput("xgb_tree_index", "Tree index to display", min = 1, max = 10, value = 1, step = 1),
        hr(),
        h5("Decision rule threshold"),
        sliderInput("xg_threshold", "Probability cutoff for class=1", min = 0.05, max = 0.95, value = 0.50, step = 0.05),
        helpText("Lower cutoff = more positives predicted (higher sensitivity).")
      ),
      layout_column_wrap(
        width = 1,
        card(class = "shadow-sm", card_header(tags$strong("Selected XGBoost Tree")),
            grVizOutput("plot_xgb_tree"))
            #plotOutput("plot_xgb_tree", height = 560))
      ),
      layout_column_wrap(
        width = 1 / 2,
        card(class = "shadow-sm", card_header(tags$strong(" performance")), verbatimTextOutput("xg_tree_perf")),
        card(class = "shadow-sm", card_header(tags$strong("ROC curve")), plotOutput("plot_xg_tree_roc", height = 320))
      )
    )
  )
)

# -------------------------
# 6) SERVER
# -------------------------
server <- function(input, output, session) {
  
  output$dict_tbl <- DT::renderDT({
    DT::datatable(feature_dictionary(), options = list(pageLength = 8, dom = "tip"), rownames = FALSE)
  }, server = FALSE)
  
  output$data_tbl <- DT::renderDT({
    DT::datatable(heart, options = list(pageLength = 10, scrollX = TRUE), rownames = FALSE)
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
  
  # ---- Main Decision Tree (controlled by sliders)
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
  
  # ---- Explainer Tree (used ONLY to ensure a meaningful patient path)
  explainer_tree_model <- reactive({
    rpart(
      target ~ .,
      data = train,
      method = "class",
      control = rpart.control(
        cp = 0.001,
        maxdepth = 6,
        minsplit = 8
      )
    )
  })
  
  output$plot_tree <- renderPlot({
    rpart.plot(tree_model(), type = 2, extra = 104, fallen.leaves = TRUE, main = "")
  })
  
  tree_perf_obj <- reactive({
    prob_mat <- predict(tree_model(), test, type = "prob")
    prob1 <- get_prob1_safe(prob_mat)
    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    pred_class <- factor(pred_class, levels = c("0", "1"))
    compute_metrics(test$target, pred_class, prob1)
  })
  
  output$tree_perf_ui <- renderUI({
    m <- tree_perf_obj()
    metrics_tbl <- tags$table(
      class = "table table-sm table-striped align-middle",
      tags$tbody(
        tags$tr(tags$th("Threshold"), tags$td(fmt(input$threshold, digits = 2))),
        tags$tr(tags$th("Accuracy"), tags$td(fmt(m$acc))),
        tags$tr(tags$th("Sensitivity"), tags$td(fmt(m$sens))),
        tags$tr(tags$th("Specificity"), tags$td(fmt(m$spec))),
        tags$tr(tags$th("AUC"), tags$td(fmt(m$auc)))
      )
    )
    tags$div(metrics_tbl, tags$div(style = "margin-top: 8px;", plotOutput("plot_tree_cm_heat", height = 280)))
  })
  
  output$plot_tree_cm_heat <- renderPlot({
    m <- tree_perf_obj()
    plot_cm_heatmap(m$cm, "Decision Tree — Confusion Matrix Heatmap")
  })
  
  output$plot_tree_roc <- renderPlot({
    m <- tree_perf_obj()
    safe_plot_roc(m$roc, m$auc, "Decision Tree ROC")
  })
  
  # ---- Patient row builder
  build_patient_row <- function() {
    one <- train[1, , drop = FALSE]
    one$age      <- as.numeric(input$p_age)
    one$sex      <- factor(input$p_sex, levels = levels(train$sex))
    one$cp       <- factor(input$p_cp, levels = levels(train$cp))
    one$trestbps <- as.numeric(input$p_trestbps)
    one$chol     <- as.numeric(input$p_chol)
    one$fbs      <- factor(input$p_fbs, levels = levels(train$fbs))
    one$restecg  <- factor(input$p_restecg, levels = levels(train$restecg))
    one$thalach  <- as.numeric(input$p_thalach)
    one$exang    <- factor(input$p_exang, levels = levels(train$exang))
    one$oldpeak  <- as.numeric(input$p_oldpeak)
    one$slope    <- factor(input$p_slope, levels = levels(train$slope))
    one$ca       <- factor(input$p_ca, levels = levels(train$ca))
    one$thal     <- factor(input$p_thal, levels = levels(train$thal))
    one
  }
  
  # ---- Walk a patient (user inputs)
  patient_result <- reactiveVal(NULL)
  
  observeEvent(input$btn_explain, {
    tryCatch({
      one <- build_patient_row()
      
      m_use <- tree_model()
      used_explainer <- FALSE
      if (!tree_has_splits(m_use)) {
        m_use <- explainer_tree_model()
        used_explainer <- TRUE
      }
      
      prob_mat <- predict(m_use, one, type = "prob")
      prob1 <- get_prob1_safe(prob_mat)
      pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
      
      patient_result(list(
        one = one,
        model_used = m_use,
        prob1 = as.numeric(prob1),
        pred_class = as.character(pred_class),
        used_explainer = used_explainer
      ))
    }, error = function(e) {
      showNotification(paste("Could not compute patient explanation:", e$message),
                       type = "error", duration = 8)
      patient_result(NULL)
    })
  })
  
  output$patient_summary_ui <- renderUI({
    res <- patient_result()
    if (is.null(res)) {
      return(tags$p("Enter details and click ", tags$b("Explain my path"),
                    " to generate the prediction and the highlighted decision tree path."))
    }
    
    label <- if (is.na(res$prob1)) {
      "Not available"
    } else if (res$pred_class == "1") {
      "Higher risk (class=1)"
    } else {
      "Lower risk (class=0)"
    }
    
    note <- if (isTRUE(res$used_explainer)) {
      tags$div(
        class = "alert alert-warning p-2 mt-2",
        tags$strong("Note: "),
        "Your slider-selected tree was too simple (no splits). An explainer tree was used so the patient path can be shown."
      )
    } else NULL
    
    tags$div(
      tags$table(
        class = "table table-sm table-striped align-middle",
        tags$tbody(
          tags$tr(tags$th("Probability of disease (class=1)"), tags$td(fmt(res$prob1))),
          tags$tr(tags$th("Threshold used"), tags$td(fmt(input$threshold, digits = 2))),
          tags$tr(tags$th("Predicted class"), tags$td(label))
        )
      ),
      note
    )
  })
  
  output$plot_patient_tree <- renderPlot({
    res <- patient_result()
    if (is.null(res)) {
      plot.new()
      text(0.5, 0.5,
           "No patient run yet.\nClick 'Explain my path' to show the tree with your path highlighted.",
           cex = 1)
      return(invisible())
    }
    
    plot_patient_tree_highlight(
      model = res$model_used,
      one_row = res$one,
      main_title = "Patient-specific Decision Tree (highlighted path)"
    )
  })
  
  # ---- Case Study Background (example patient = first row of train)
  patient_result_bg <- reactiveVal(NULL)
  
  observeEvent(input$btn_explain_bg, {
    tryCatch({
      one <- train[1, , drop = FALSE]
      
      m_use <- tree_model()
      used_explainer <- FALSE
      if (!tree_has_splits(m_use)) {
        m_use <- explainer_tree_model()
        used_explainer <- TRUE
      }
      
      prob_mat <- predict(m_use, one, type = "prob")
      prob1 <- get_prob1_safe(prob_mat)
      pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
      
      patient_result_bg(list(
        one = one,
        model_used = m_use,
        prob1 = as.numeric(prob1),
        pred_class = as.character(pred_class),
        used_explainer = used_explainer
      ))
    }, error = function(e) {
      showNotification(paste("Could not compute case study explanation:", e$message),
                       type = "error", duration = 8)
      patient_result_bg(NULL)
    })
  })
  
  output$patient_summary_ui_bg <- renderUI({
    res <- patient_result_bg()
    if (is.null(res)) {
      return(tags$p("Click ", tags$b("Explain my path"),
                    " to generate the example patient prediction and highlighted path."))
    }
    
    label <- if (is.na(res$prob1)) {
      "Not available"
    } else if (res$pred_class == "1") {
      "Higher risk (class=1)"
    } else {
      "Lower risk (class=0)"
    }
    
    note <- if (isTRUE(res$used_explainer)) {
      tags$div(
        class = "alert alert-warning p-2 mt-2",
        tags$strong("Note: "),
        "Your slider-selected tree was too simple (no splits). An explainer tree was used so the path can be shown."
      )
    } else NULL
    
    tags$div(
      tags$table(
        class = "table table-sm table-striped align-middle",
        tags$tbody(
          tags$tr(tags$th("Probability of disease (class=1)"), tags$td(fmt(res$prob1))),
          tags$tr(tags$th("Threshold used"), tags$td(fmt(input$threshold, digits = 2))),
          tags$tr(tags$th("Predicted class"), tags$td(label))
        )
      ),
      note
    )
  })
  
  output$plot_patient_tree_bg <- renderPlot({
    res <- patient_result_bg()
    if (is.null(res)) {
      plot.new()
      text(0.5, 0.5,
           "Click 'Explain my path' to generate the example patient tree path.",
           cex = 1)
      return(invisible())
    }
    
    plot_patient_tree_highlight(
      model = res$model_used,
      one_row = res$one,
      main_title = "Example Patient — Decision Tree (highlighted path)"
    )
  })
  
  # ---- Random forest
  rf_model <- reactive({
    randomForest(target ~ ., data = train,
                 ntree = input$rf_ntree, mtry = input$rf_mtry, importance = TRUE)
  })
  
  rf_perf_obj <- reactive({
    prob_mat <- predict(rf_model(), test, type = "prob")
    prob1 <- get_prob1_safe(prob_mat)
    pred_class <- ifelse(prob1 >= input$threshold, "1", "0")
    pred_class <- factor(pred_class, levels = c("0", "1"))
    compute_metrics(test$target, pred_class, prob1)
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
    
    ggplot(imp_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
      geom_col() +
      coord_flip() +
      labs(x = "", y = "Mean Decrease Gini", title = "Top 10 predictors (Random Forest)") +
      theme_minimal(base_size = 13)
  })
  
  output$rf_perf_ui <- renderUI({
    m <- rf_perf_obj()
    oob <- tryCatch(tail(rf_model()$err.rate, 1), error = function(e) NULL)
    
    metrics_tbl <- tags$table(
      class = "table table-sm table-striped align-middle",
      tags$tbody(
        tags$tr(tags$th("Threshold"), tags$td(fmt(input$threshold, digits = 2))),
        tags$tr(tags$th("Accuracy"), tags$td(fmt(m$acc))),
        tags$tr(tags$th("Sensitivity"), tags$td(fmt(m$sens))),
        tags$tr(tags$th("Specificity"), tags$td(fmt(m$spec))),
        tags$tr(tags$th("AUC"), tags$td(fmt(m$auc))),
        tags$tr(tags$th("OOB error (training RF)"),
                tags$td(if (is.null(oob)) "NA" else fmt(as.numeric(oob[1, "OOB"]), digits = 3)))
      )
    )
    
    tags$div(
      metrics_tbl,
      tags$div(style = "margin-top: 8px;", plotOutput("plot_rf_cm_heat", height = 280))
    )
  })
  
  output$plot_rf_cm_heat <- renderPlot({
    m <- rf_perf_obj()
    plot_cm_heatmap(m$cm, "Random Forest — Confusion Matrix Heatmap")
  })
  
  output$plot_rf_roc <- renderPlot({
    m <- rf_perf_obj()
    safe_plot_roc(m$roc, m$auc, "Random Forest ROC")
  })
  
  # ---- XG Boosted (rpart-based as per your original section)
  build_xgb_model <- function(
    tree_depth = 3,
    learn_rate = 1,
    trees = 3
  ){
    xgb.model <- boost_tree(
        mode = 'classification',
        engine = 'xgboost',
        tree_depth = tree_depth,
        learn_rate = learn_rate,
        trees = trees,
      )

    xgb.wf <- workflow() |> 
      add_recipe(heart.recipe) |> 
      add_model(xgb.model)

    xgb.wf |> fit(train)
  }

  xgb.fit <- reactiveVal(build_xgb_model())

  observeEvent(input$btn_train_xgb, {
    tryCatch({
      
      xgb.fit(
        build_xgb_model(
          tree_depth = input$xgb_max_depth,
          learn_rate = input$xgb_eta,
          trees = input$xgb_num_boost_round
        )
      )

    }, error = function(e) {
      showNotification(paste("Could not train XGBoost model:", e$message),
                       type = "error", duration = 8)
      xgb.fit(NULL)
    })
  })
  
  output$plot_xgb_tree <- renderGrViz({
    
    xgb.fit.obj <- extract_fit_engine(xgb.fit())

    xgb.plot.tree(xgb.fit.obj,
      tree_idx = input$xgb_tree_index,
      with_stats=TRUE)
  })
  
  xg_perf_obj <- reactive({
    # prob_mat <- predict(xg_tree_model(), test, type = "prob")
    # prob1 <- get_prob1_safe(prob_mat)
    # pred_class <- ifelse(prob1 >= input$xg_threshold, "1", "0")
    # pred_class <- factor(pred_class, levels = c("0", "1"))
    # compute_metrics(test$target, pred_class, prob1)
    NULL
  })
  
  output$xg_tree_perf <- renderPrint({
    # m <- xg_perf_obj()
    # cat("Threshold:", input$xg_threshold, "\n\n")
    # cat("Confusion Matrix:\n\n")
    # print(m$cm)
    # cat("\nAccuracy:", fmt(m$acc),
    #     "\nSensitivity:", fmt(m$sens),
    #     "\nSpecificity:", fmt(m$spec),
    #     "\nAUC:", fmt(m$auc), "\n")
    print(xgb.fit())
  })
  
  output$plot_xg_tree_roc <- renderPlot({
    # m <- xg_perf_obj()
    # safe_plot_roc(m$roc, m$auc, "XG Boosted ROC")
    return(invisible())
  })
}

shinyApp(ui, server)