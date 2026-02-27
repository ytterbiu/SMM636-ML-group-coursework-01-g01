# app.R
# ============================================================================ #
# GCW1 — FINAL (Guided Presentation Version) ====
# ============================================================================ #
# Libraries ====
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
library(shapviz)

# ============================================================================ #
# Data: find, load, clean ====
get_data_path <- function() {
  candidates <- c("processed.cleveland.data.csv", "processed.cleveland.data")
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
  stop("Dataset file not found.")
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
heart.recipe <- heart |>
  recipe(target ~ .) |>
  step_dummy(all_nominal_predictors())

# ============================================================================ #
# Train/Test ====
set.seed(123)
idx0 <- which(heart$target == "0")
idx1 <- which(heart$target == "1")
train0 <- sample(idx0, size = floor(0.70 * length(idx0)))
train1 <- sample(idx1, size = floor(0.70 * length(idx1)))
train_index <- c(train0, train1)
train <- heart[train_index, ]
test <- heart[-train_index, ]

# ============================================================================ #
## Helper functions ----
fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, digits = digits, format = "f"))
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

plot_cm_heatmap <- function(cm, title_text = "") {
  df <- as.data.frame(cm)
  colnames(df) <- c("Predicted", "Actual", "Count")
  ggplot(df, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = Count), size = 5) +
    scale_fill_gradient(low = "white", high = "#8A1C3D") +
    labs(
      title = title_text,
      x = "Actual (0=Healthy, 1=Disease)",
      y = "Predicted"
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "none")
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
safe_plot_roc <- function(roc_obj, auc_val, title_text) {
  if (is.null(roc_obj) || is.na(auc_val)) {
    plot.new()
    text(0.5, 0.5, "ROC not available.", cex = 1)
    return(invisible())
  }

  # Use ggroc to plot False Positive Rate (0 to 1) vs True Positive Rate (0 to 1)
  g <- pROC::ggroc(roc_obj, legacy.axes = TRUE) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      color = "gray50"
    ) +
    theme_minimal(base_size = 14) +
    labs(
      title = paste0(title_text, " (AUC = ", fmt(auc_val), ")"),
      x = "False Positive Rate (1 - Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) # Perfectly squares and locks the axes

  return(g)
}

# ============================================================================ #
# UI ====
# ============================================================================ #
theme_gc <- bs_theme(version = 5, bootswatch = "cosmo", primary = "#8A1C3D")

ui <- page_navbar(
  id = "main_nav",
  title = "Tree-Based Methods: Predicting Heart Disease",
  theme = theme_gc,

  header = tags$head(
    tags$style(HTML(
      "
      body { padding-bottom: 80px; } 
      .tab-pane { padding-bottom: 90px !important; } /* fixes the scroll cut-off */
      .nav-footer { position: fixed; bottom: 0; left: 0; width: 100%; background: #f8f9fa; 
                    padding: 15px 20px; border-top: 1px solid #ddd; z-index: 1000; 
                    display: flex; justify-content: space-between; }
      .info-btn { display:inline-flex; align-items:center; justify-content:center; width:18px; 
                  height:18px; margin-left:8px; border-radius:999px; background: var(--bs-primary); 
                  color: #fff; font-size:12px; cursor: pointer; }
    "
    )),
    tags$script(HTML(
      "
      document.addEventListener('DOMContentLoaded', function () {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle=\"tooltip\"]'));
        tooltipTriggerList.map(function (t) { return new bootstrap.Tooltip(t); });
      });
    "
    ))
  ),

  # ========================================================================== #
  ## Tab 1: The problem ----
  nav_panel(
    title = "1. The Problem",
    value = "tab1",
    layout_column_wrap(
      width = 1 / 2, # Splits the screen exactly 50/50

      # LEFT SIDE: Text
      card(
        h3("Predicting Heart Disease: A Case Study"),
        p(
          "In clinical settings, identifying patients at high risk of heart disease quickly and accurately is critical."
        ),
        p(
          "This presentation demonstrates how ",
          tags$b("Tree-Based Machine Learning"),
          " can be used to solve this problem. We use historical health data from Cleveland to build a decision tree that learns the 'rules' needed to predict whether an individual is suffering from heart disease or not."
        ),

        tags$hr(),

        h5("What are Tree-Based Methods?"),
        p(
          "Tree-based methods are a powerful but intuitive way to classify data. They work by asking a sequence of simple 'yes or no' questions about the data (e.g., 'Is the patient older than 50?', 'Is their maximum heart rate below 150?')."
        ),
        p(
          "Each answer splits the data into smaller, more specific groups until a final classification - such as 'Healthy' or 'Disease' - is reached. The result looks just like an upside-down tree or a flowchart."
        ),

        tags$hr(),

        h5("Why Tree-Based Methods?"),
        tags$ul(
          tags$li(
            "They are highly interpretable (you can follow the logic step-by-step)."
          ),
          tags$li("They mimic human decision-making processes."),
          tags$li(
            "Advanced versions (like Random Forests or 'XGBoost') offer state-of-the-art predictive accuracy."
          )
        )
      ),

      # RIGHT SIDE: Image Graphic
      card(
        h5("How a Decision Tree Works"),
        tags$div(
          style = "text-align: left; padding: 5px;",

          # description above the image
          tags$p(
            "This is an example of a decision tree used to ",
            tags$em("classify"),
            " the type of a tree (like Oak, Cherry, or Apple) based on its height and diameter. We start at the top of the tree and apply the first rule - in this case, checking if the diameter is less than or equal to 0.45m. This step-by-step sorting continues until we hit a final 'leaf' at the end of the branch, providing the exact classification."
          ),

          # image call (Fixed for centering)
          tags$img(
            src = "decision_tree_transparent.png",
            alt = "Illustration of a basic decision tree classifying tree types",
            style = "display: block; margin-left: auto; margin-right: auto; max-width: 100%; max-height: 400px; height: auto; margin-bottom: 15px;"
          ),

          # italicized note about training/testing below the image
          tags$p(
            style = "font-size: 0.9em; color: #555; text-align: center;",
            tags$i(
              "Note: This decision tree is built from an existing set of data. It is then tested and used on separate, new data."
            )
          ),

          tags$p(
            "In machine learning, we split our data into 'training' and 'testing' partitions that do not overlap. The training dataset is used to build the tree, and we evaluate how well the tree performs (how accurately it classifies) using the testing data to ensure it can accurately classify new information."
          )
        )
      )
    )
  ),

  # ========================================================================== #
  ## Tab 2: Data explore ----
  nav_panel(
    title = "2. The Data",
    value = "tab2",

    # top row: summary value boxes
    layout_column_wrap(
      width = 1 / 3,
      value_box(
        "Observations (Patients)",
        nrow(heart),
        theme = "primary",
        p("total records in our dataset")
      ),
      value_box(
        "Predictor Variables",
        ncol(heart) - 1,
        theme = "info",
        p("available clinical measurements")
      ),
      value_box(
        "Prevalence (Disease = 1)",
        paste0(
          round(mean(as.numeric(as.character(heart$target))) * 100, 1),
          "%"
        ),
        theme = "danger",
        p("proportion of patients with heart disease")
      )
    ),

    # middle row: explanations and data dictionary
    layout_column_wrap(
      width = 1 / 2,

      # left side: the problem & teaching points
      card(
        h4("From simple examples to real-world complexity"),
        p(
          "Unlike the basic 'tree type' example, real-world data is much more complex. In this case study, we are trying to predict a ",
          tags$b("target variable"),
          " (whether a patient has heart disease or not) using 13 different pieces of patient information",
          tags$b("(predictor variables)"),
          "."
        ),

        # tags$hr(),

        h5("The value of tree-based methods here"),
        p(
          "Real-world data is rarely perfect or 'clean'. Tree-based methods are valuable because they are uniquely equipped to handle these complexities:"
        ),
        tags$ul(
          tags$li(
            tags$b("Handling mixed data types: "),
            "Our dataset contains both ",
            tags$i("quantitative"),
            " (numeric, like Age or Cholesterol) and ",
            tags$i("qualitative"),
            " (categorical, like Chest Pain type). Trees can process both naturally without needing complex mathematical transformations."
          ),
          tags$li(
            tags$b("Built-in feature selection: "),
            "If a specific clinical measurement doesn't help separate healthy patients from sick ones, the tree simply won't use it. It filters out the noise automatically."
          ),
          tags$li(
            tags$b("Capturing complex interactions: "),
            "Trees easily capture nuanced rules, such as: ",
            tags$em(
              "'High cholesterol might only be a strong risk indicator IF the patient is also over 60 years old.'"
            )
          )
        )
      ),

      # right side: grouped data dictionary
      card(
        h4("The 13 predictors explained"),
        p(
          "The Cleveland Heart Disease data contains several clinical measurements we can use to predict whether somebody has heart disease, including:"
        ),

        accordion(
          open = FALSE, # keeps them closed by default for a clean look
          accordion_panel(
            "Demographics & Symptoms",
            tags$ul(
              tags$li(tags$b("Age:"), " patient's age in years (Numeric)"),
              tags$li(
                tags$b("Sex:"),
                " gender (0 = Female, 1 = Male) (Nominal)"
              ),
              tags$li(
                tags$b("cp (Chest Pain):"),
                " type of chest pain (0 = typical angina, 1 = atypical, 2 = non-anginal, 3 = asymptomatic) (Nominal)"
              )
            )
          ),
          accordion_panel(
            "Clinical Vitals",
            tags$ul(
              tags$li(
                tags$b("trestbps:"),
                " resting blood pressure in mm/Hg (Numeric)"
              ),
              tags$li(tags$b("chol:"), " serum cholesterol in mg/dl (Numeric)"),
              tags$li(
                tags$b("fbs:"),
                " fasting blood sugar > 120 mg/dl (0 = False, 1 = True) (Nominal)"
              )
            )
          ),
          accordion_panel(
            "Test Results (ECG & Exercise)",
            tags$ul(
              tags$li(
                tags$b("restecg:"),
                " resting ECG results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy) (Nominal)"
              ),
              tags$li(
                tags$b("thalach:"),
                " maximum heart rate achieved (Numeric)"
              ),
              tags$li(
                tags$b("exang:"),
                " exercise-induced angina (0 = No, 1 = Yes) (Nominal)"
              ),
              tags$li(
                tags$b("oldpeak:"),
                " exercise-induced ST-depression relative to rest (Numeric)"
              ),
              tags$li(
                tags$b("slope:"),
                " slope of peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping) (Nominal)"
              ),
              tags$li(
                tags$b("ca:"),
                " number of major vessels (0–3) colored by fluoroscopy (Nominal)"
              ),
              tags$li(
                tags$b("thal:"),
                " thalassemia blood disorder (1 = normal, 2 = fixed defect, 3 = reversible defect) (Nominal)"
              )
            )
          )
        )
      )
    )
  ),

  # ========================================================================== #
  ## Tab 3: Step-by-step ----
  nav_panel(
    title = "3. Building a Tree",
    value = "tab3",

    # top instructions and dynamic value boxes
    layout_sidebar(
      sidebar = sidebar(
        h5("Growing a Tree"),
        helpText(
          "Move the slider to see how the algorithm divides the data step-by-step, learning new rules with each layer.",
          br(),
          tags$em(
            "Note: all scores are calculated here for the training dataset."
          )
        ),
        sliderInput(
          "step_depth",
          "Tree depth level:",
          min = 1,
          max = 5,
          value = 1,
          step = 1,
          animate = animationOptions(interval = 5000)
        )
      ),

      # top row: dynamic stats based on the slider
      uiOutput("step_metrics_ui"),

      layout_column_wrap(
        width = 1 / 2,

        # left side: the traditional tree diagram
        card(
          class = "shadow-sm",
          card_header(tags$strong("The decision rules")),

          # instructions on how to read the tree
          tags$div(
            style = "background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9em;",
            tags$b("How to read the boxes:"),
            tags$ul(
              style = "margin-bottom: 0; padding-left: 20px;",
              tags$li(
                tags$b("Color: "),
                "Blue = Predicted Healthy, Red = Predicted Disease."
              ),
              tags$li(
                tags$b("Top text: "),
                "The final prediction for patients in this group (0 or 1)."
              ),
              tags$li(
                tags$b("Middle text: "),
                "The probability of having heart disease in this group."
              ),
              tags$li(
                tags$b("Bottom text: "),
                "The percentage of all patients that fall into this group."
              )
            )
          ),

          plotOutput("plot_step_tree", height = 400)
        ),

        # right side: the 2d problem space with shaded regions
        card(
          class = "shadow-sm",
          card_header(tags$strong("The 2D problem space (Age vs Heart Rate)")),
          plotOutput("plot_step_2d", height = 400),

          # contextual note requested by user
          tags$p(
            style = "font-size: 0.9em; color: #555; text-align: center; margin-top: 10px;",
            tags$i(
              "Note: This is only a 2D slice of our data. While this simple example segments patients using just two variables, a full machine learning model filters based upon all the other clinical data not shown here, creating a complex, multi-dimensional set of rules."
            )
          )
        )
      )
    )
  ),

  # ========================================================================== #
  ## Tab 4: Full decision tree ----
  nav_panel(
    title = "4. Full Decision Tree",
    value = "tab4",

    layout_sidebar(
      sidebar = sidebar(
        h5("1. Patient Profile"),
        helpText(
          tags$b("Feature Selection in action: "),
          "Even though we provided 13 variables, the optimal tree only needs these 5 key metrics to make its predictions. Adjust them to see the path change!"
        ),

        # updated inputs to match the tree's actual splits
        selectInput(
          "p_cp",
          "Chest Pain Type (cp)",
          choices = c(
            "Typical Angina" = "0",
            "Atypical" = "1",
            "Non-anginal" = "2",
            "Asymptomatic" = "3"
          ),
          selected = "3"
        ),
        selectInput(
          "p_ca",
          "Major Vessels Colored (ca)",
          choices = c("0" = "0", "1" = "1", "2" = "2", "3" = "3"),
          selected = "0"
        ),
        selectInput(
          "p_thal",
          "Thalassemia (thal)",
          choices = c(
            "Normal" = "1",
            "Fixed Defect" = "2",
            "Reversible Defect" = "3"
          ),
          selected = "1"
        ),
        numericInput(
          "p_oldpeak",
          "ST Depression (oldpeak)",
          value = 1.0,
          min = 0,
          max = 6.2,
          step = 0.1
        ),
        selectInput(
          "p_slope",
          "ST Slope (slope)",
          choices = c("Upsloping" = "0", "Flat" = "1", "Downsloping" = "2"),
          selected = "1"
        ),

        actionButton(
          "btn_predict",
          "Predict Patient Path & Condition",
          class = "btn-primary"
        ),
        uiOutput("predicted_condition_ui"),

        tags$hr(),

        h5("2. Model Complexity"),
        helpText(
          "Trees can grow too complex and 'overfit' the training data. Pruning helps it generalise to new patients."
        ),
        checkboxInput(
          "auto_cp",
          "Use optimal complexity (Auto-prune)",
          value = TRUE
        ),

        # this slider only appears if the checkbox above is UNTICKED
        conditionalPanel(
          condition = "!input.auto_cp",
          sliderInput(
            "tree_cp",
            "Pruning strength (lower = more complex tree):",
            min = 0.001,
            max = 0.04,
            value = 0.01,
            step = 0.005
          ),
          helpText(
            "This control is the model’s complexity parameter: higher values prune more resulting in a simpler tree."
          )
        ),
      ),

      # top row: testing data performance metrics
      uiOutput("tab4_metrics_ui"),

      layout_column_wrap(
        width = 1, # single column for a nice wide tree
        card(
          class = "shadow-sm",
          card_header(tags$strong("The full clinical decision tree")),
          plotOutput("plot_full_tree", height = 500),

          tags$p(
            style = "font-size: 0.9em; color: #555; text-align: center; margin-top: 10px;",
            tags$i(
              "Note: This tree was built using the Training Data, but the accuracy scores above are calculated using the unseen Testing Data to see how well the tree works on new data."
            )
          )
        )
      )
    )
  ),

  # ========================================================================== #
  # ========================================================================== #
  ## Tab 5: Ensembles ----
  nav_panel(
    title = "5. Ensembles",
    value = "tab5",

    navset_card_underline(
      ### subtab 1 ----
      # --- Subtab 1: RF Explanation ---
      nav_panel(
        "1. What is a Random Forest?",
        layout_column_wrap(
          width = 1 / 2,
          card(
            h4("The wisdom of crowds"),
            p(
              "A Decision Tree is great, but it has a flaw - it can easily over-memorise the training data (overfitting) and become unstable. If you change the data slightly, the tree changes completely."
            ),
            p(
              "A ",
              tags$b("Random Forest"),
              " solves this by building hundreds of different decision trees. To make sure the trees aren't all identical, the algorithm uses two tricks:"
            ),
            tags$ul(
              tags$li(
                tags$b("Bootstrapping: "),
                "Each tree is trained on a random, slightly different subset of the patients."
              ),
              tags$li(
                tags$b("Feature Sampling: "),
                "At each split, the tree is only allowed to look at a random subset of the clinical variables."
              )
            ),
            p(
              "When a new patient arrives, all 500 trees 'vote' on the diagnosis. The majority wins. This approach typically improves accuracy and stability."
            ),
            tags$div(
              style = "text-align: left; padding-top: 2px;",
              tags$p(
                "In our case study, random forests can help to identify which clinical variables reliably drive predictions across many different trees, highlighting factors that are consistently informative."
              )
            )
          ),
          card(
            h5("Random Forest visualised"),

            tags$div(
              style = "text-align: center;",
              tags$img(
                src = "rf_intro_graphic.png",
                style = "display:block; margin:0 auto; max-width:100%; max-height:350px; height:auto;"
              )
            ),

            tags$p(
              style = "text-align: center; padding: 12px 0 0 0; margin: 0;",
              tags$em(
                "Image from https://www.grammarly.com/blog/ai/what-is-random-forest/ (accessed 27/02/2025)."
              )
            ),
          )
        )
      ),

      ### subtab 2 ----
      # --- Subtab 2: RF Implementation ---
      nav_panel(
        "2. Random Forest in Action",

        layout_sidebar(
          sidebar = sidebar(
            h5("Forest Controls"),
            sliderInput(
              "rf_ntree",
              "Number of trees in forest:",
              min = 50,
              max = 500,
              value = 200,
              step = 50
            ),
            hr(),
            h5("Inspect Individual Trees"),
            helpText(
              "Cycle through the simulated trees that make up the forest to see how different they are from each other."
            ),
            sliderInput(
              "rf_tree_index",
              "Tree Index to View:",
              min = 1,
              max = 20,
              value = 1,
              step = 1
            )
          ),

          # Top row: Comparative stats (Forest vs individual selected tree)
          uiOutput("rf_comparative_metrics_ui"),

          layout_column_wrap(
            width = 1 / 2,
            card(
              class = "shadow-sm",
              uiOutput("rf_tree_header_ui"), # (⌐■_■)
              plotOutput("plot_rf_individual_tree", height = 400)
            ),
            card(
              class = "shadow-sm",
              card_header(tags$strong("Overall Forest variable importance")),
              plotOutput("plot_rf_imp", height = 400),
              p(
                class = "text-muted",
                style = "margin: 0.5rem 1rem 0.75rem 1rem; font-size: 0.9rem; line-height: 1.25;",
                "Because drawing hundreds of trees is hard to read, we summarise them with a bar chart. Variables nearer the top carried the most weight in the model’s decisions."
              )
            )
          )
        )
      ),

      ### subtab 3 ----
      # --- Subtab 3: XGBoost Explanation ---
      nav_panel(
        "3. What is XGBoost?",
        layout_column_wrap(
          width = 1 / 2,
          card(
            h4("Learning from mistakes"),
            p(
              "While a Random Forest builds hundreds of trees independently at the same time, ",
              tags$b("Boosted Trees (XGBoost)"),
              " build trees sequentially - one after another."
            ),
            p(
              "Each new tree looks specifically at the patients that the previous trees misclassified, and tries to correct those specific mistakes. This is similar to a student taking a practice test, seeing what they got wrong, and studying only those topics for the next test."
            ),
            tags$ul(
              tags$li(
                tags$b("Learning Rate: "),
                "Controls how aggressive the corrections are. A smaller rate means it learns slower but avoids over-correcting."
              ),
              tags$li(
                tags$b("Max Depth: "),
                "Boosted trees are usually kept very shallow (called 'stumps') so they don't overfit."
              )
            ),
            p(
              "XGBoost is often considered the most powerful algorithm for tabular real-world data."
            )
          ),
          card(
            h5("XGBoost algorithm visualised"),
            tags$div(
              style = "text-align: center; padding: 20px;",
              tags$img(
                src = "xgb_intro_graphic.png",
                style = "max-width: 100%; max-height: 550px; "
              )
            )
          )
        )
      ),

      ### subtab 4 ----
      # --- Subtab 4: XGBoost Implementation ---
      nav_panel(
        "4. XGBoost in Action",
        layout_sidebar(
          sidebar = sidebar(
            h5("XGBoost controls"),
            sliderInput(
              "xgb_num_boost_round",
              "Number of Boosting rounds (Trees):",
              min = 1,
              max = 20,
              value = 10,
              step = 1
            ),
            sliderInput(
              "xgb_max_depth",
              "Max Tree Depth:",
              min = 1,
              max = 4,
              value = 3,
              step = 1
            ),
            sliderInput(
              "xgb_eta",
              "Learning Rate:",
              min = 0.05,
              max = 1,
              value = 0.3,
              step = 0.05
            ),
            actionButton(
              "btn_train_xgb",
              "Train XGBoost",
              class = "btn-primary"
            ),
            hr(),
            sliderInput(
              "xgb_tree_index",
              "Tree index to display:",
              min = 1,
              max = 10,
              value = 1,
              step = 1
            ),
            hr(),
            h5("Decision rule threshold"),
            sliderInput(
              "xg_threshold",
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
            width = 1 / 2,
            card(
              class = "shadow-sm",
              card_header(tags$strong("Selected XGBoost Tree")),
              grVizOutput("plot_xgb_tree")
            ),
            card(
              class = "shadow-sm",
              card_header(tags$strong("XGBoost Performance")),
              uiOutput("xgb_perf_ui")
            )
          ),
          layout_column_wrap(
            width = 1 / 2,
            card(
              class = "shadow-sm",
              card_header(tags$strong("Feature Importance")),
              plotOutput("xgb_feature_imp")
            ),
            card(
              class = "shadow-sm",
              card_header(tags$strong("ROC Curve")),
              plotOutput("plot_xg_tree_roc", height = 320)
            )
          )
        )
      )
    )
  ),

  # ========================================================================== #
  ## Tab 6: Takeaways ----
  nav_panel(
    title = "6. Takeaways",
    value = "tab6",
    layout_column_wrap(
      width = 1 / 2, # Splits the screen exactly 50/50

      # LEFT SIDE: Text summarizing the models
      card(
        h3("Which model is the right choice?"),
        p(
          "In data analytics, there is rarely a single 'perfect' algorithm. As we have demonstrated with this clinical dataset, choosing a model requires balancing ",
          tags$b("accuracy"),
          " with ",
          tags$b("interpretability."),
        ),

        tags$hr(),

        h5("1. Decision Trees"),
        p(tags$b("The clear communicator.")),
        tags$ul(
          tags$li(
            "Highly transparent and easy to explain to stakeholders or regulators."
          ),
          tags$li("Mimics human decision-making processes."),
          tags$li(
            tags$span(
              style = "color: #e74c3c; font-weight: bold;",
              "Limitation: "
            ),
            "Prone to overfitting and generally has the lowest predictive accuracy of the three."
          )
        ),

        # tags$hr(),

        h5("2. Random Forests"),
        p(tags$b("The stable workhorse.")),
        tags$ul(
          tags$li("Excellent out-of-the-box predictive accuracy."),
          tags$li(
            "Highly resistant to overfitting due to 'the wisdom of crowds' (averaging hundreds of independent trees)."
          ),
          tags$li(
            tags$span(
              style = "color: #e74c3c; font-weight: bold;",
              "Limitation: "
            ),
            "It is a 'black box'. We can see which variables are important overall, but we cannot easily draw a single flowchart for a specific patient."
          )
        ),
        #
        # tags$hr(),

        h5("3. XGBoost"),
        p(tags$b("The precision instrument.")),
        tags$ul(
          tags$li(
            "Often provides state-of-the-art, best-in-class predictive accuracy."
          ),
          tags$li(
            "Learns sequentially from its own mistakes to optimize performance."
          ),
          tags$li(
            tags$span(
              style = "color: #e74c3c; font-weight: bold;",
              "Limitation: "
            ),
            "Requires careful mathematical tuning to prevent it from memorizing the noise in the data. Also a 'black box'."
          )
        )
      ),

      # RIGHT SIDE: The Final Pitch
      card(
        h5("Our Approach"),
        tags$div(
          style = "text-align: left; padding: 5px;",

          tags$p(
            "As actuaries and data professionals, we do not just apply complex algorithms blindly. We partner with you to understand your specific business needs:"
          ),

          tags$ul(
            tags$li(
              "If your priority is ",
              tags$b("regulatory compliance and transparency"),
              " (e.g., explaining exactly why a policy was priced a certain way), we will leverage optimized ",
              tags$b("Decision Trees"),
              "."
            ),
            tags$li(
              "If your priority is ",
              tags$b("pure predictive power"),
              " (e.g., catching as many fraudulent claims as possible), we will deploy highly tuned ",
              tags$b("Ensembles like XGBoost"),
              "."
            )
          ),

          tags$br(),

          tags$div(
            style = "background-color: #f8f9fa; padding: 15px; border-left: 4px solid #8A1C3D; border-radius: 4px;",
            tags$p(
              style = "font-weight: bold; font-size: 1.1em; color: #333; margin-bottom: 0;",
              "We balance cutting-edge accuracy with transparent, interpretable models."
            )
          ),

          tags$br(),
          tags$br(),

          tags$p(
            style = "font-size: 1.0em; color: #555; text-align: center;",
            tags$i("Thank you for exploring this interactive case study.")
          )
        )
      )
    )
  ),

  ## Footer navigation ----
  # --- FOOTER NAVIGATION ---
  footer = tags$div(
    class = "nav-footer",
    actionButton("btn_prev", "← Previous Step", class = "btn-secondary"),
    actionButton("btn_next", "Next Step →", class = "btn-primary")
  )
)

# ============================================================================ #
# Server ====
# ============================================================================ #
server <- function(input, output, session) {
  # Navigation Logic
  tab_order <- c("tab1", "tab2", "tab3", "tab4", "tab5", "tab6")

  observeEvent(input$btn_next, {
    curr <- input$main_nav
    idx <- which(tab_order == curr)
    if (length(idx) == 1 && idx < length(tab_order)) {
      updateNavbarPage(session, "main_nav", selected = tab_order[idx + 1])
    }
  })

  observeEvent(input$btn_prev, {
    curr <- input$main_nav
    idx <- which(tab_order == curr)
    if (length(idx) == 1 && idx > 1) {
      updateNavbarPage(session, "main_nav", selected = tab_order[idx - 1])
    }
  })

  # ========================================================================== #
  ## Tab 2: Data explore server logic ----

  # render the target balance bar chart
  output$plot_target <- renderPlot({
    ggplot(heart, aes(x = target, fill = target)) +
      geom_bar() +
      scale_fill_manual(values = c("0" = "#4A90E2", "1" = "#8A1C3D")) +
      labs(x = "Target (0 = Healthy, 1 = Disease)", y = "Patient Count") +
      theme_minimal(base_size = 14) +
      theme(legend.position = "none")
  })

  # ========================================================================== #
  ## Tab 3: Step-by-step server logic ----

  # build a model using only age and thalach restricted by the depth slider
  step_model <- reactive({
    req(input$step_depth)
    rpart(
      target ~ age + thalach,
      data = train,
      method = "class",
      control = rpart.control(
        maxdepth = input$step_depth,
        cp = -1,
        minsplit = 2
      )
    )
  })

  # dynamic value boxes for the top of tab 3
  output$step_metrics_ui <- renderUI({
    req(step_model())

    # get predictions for current tree depth
    preds <- predict(step_model(), train, type = "class")
    actual <- train$target

    # calculate overall accuracy
    acc <- sum(preds == actual) / nrow(train)

    # calculate 'confidence' metrics
    healthy_idx <- which(preds == "0")
    conf_healthy <- if (length(healthy_idx) > 0) {
      sum(actual[healthy_idx] == "0") / length(healthy_idx)
    } else {
      0
    }

    disease_idx <- which(preds == "1")
    conf_disease <- if (length(disease_idx) > 0) {
      sum(actual[disease_idx] == "1") / length(disease_idx)
    } else {
      0
    }

    # percentages of population
    pct_healthy <- length(healthy_idx) / nrow(train)
    pct_disease <- length(disease_idx) / nrow(train)

    layout_column_wrap(
      width = 1 / 3,
      value_box(
        "Current Accuracy",
        paste0(round(acc * 100, 1), "%"),
        theme = "success",
        p("how often the current rules are correct")
      ),
      value_box(
        "Predicted Healthy (NPV)",
        paste0(round(pct_healthy * 100, 1), "%"),
        # custom theme to match the background blue exactly
        theme = value_box_theme(bg = "#4A90E2", fg = "white"),
        p(paste0(
          "(",
          round(conf_healthy * 100, 1),
          "% of these were actually healthy)"
        ))
      ),
      value_box(
        "Predicted Disease (PPV)",
        paste0(round(pct_disease * 100, 1), "%"),
        theme = "danger",
        p(paste0(
          "(",
          round(conf_disease * 100, 1),
          "% of these were actually sick)"
        ))
      )
    )
  })

  # render the traditional tree diagram
  output$plot_step_tree <- renderPlot({
    m <- step_model()
    # colors matching the prediction: 0 (Healthy) = Blue, 1 (Disease) = Red
    box_colors <- ifelse(m$frame$yval == 1, "#4A90E2", "#c3436a")

    rpart.plot(
      m,
      type = 2,
      extra = 104,
      fallen.leaves = FALSE,
      main = "",
      box.col = box_colors,
      shadow.col = "gray90"
    )
  })

  # render the 2d scatter plot with dynamic decision boundaries
  output$plot_step_2d <- renderPlot({
    m <- step_model()

    grid_data <- expand.grid(
      age = seq(
        min(train$age, na.rm = T),
        max(train$age, na.rm = T),
        length.out = 150
      ),
      thalach = seq(
        min(train$thalach, na.rm = T),
        max(train$thalach, na.rm = T),
        length.out = 150
      )
    )

    grid_data$pred_class <- predict(m, newdata = grid_data, type = "class")

    ggplot() +
      geom_raster(
        data = grid_data,
        aes(x = age, y = thalach, fill = pred_class),
        alpha = 0.4
      ) +
      geom_point(
        data = train,
        aes(x = age, y = thalach, color = target),
        size = 2.5,
        shape = 16
      ) +
      scale_color_manual(
        values = c("0" = "#4A90E2", "1" = "#8A1C3D"),
        name = "Actual Status",
        labels = c("Healthy", "Disease")
      ) +
      scale_fill_manual(
        values = c("0" = "#4A90E2", "1" = "#8A1C3D"),
        guide = "none"
      ) +
      theme_minimal(base_size = 14) +
      labs(x = "Age (Years)", y = "Max Heart Rate (thalach)") +
      theme(legend.position = "bottom")
  })

  # FIX: These lines ensure the plots load even before the tab is clicked
  outputOptions(output, "plot_step_tree", suspendWhenHidden = FALSE)
  outputOptions(output, "plot_step_2d", suspendWhenHidden = FALSE)

  # ========================================================================== #
  # ========================================================================== #
  ## Tab 4: Full decision tree server logic ----

  # determine the complexity parameter based on the checkbox
  current_cp <- reactive({
    if (input$auto_cp) {
      0.015 # a hardcoded 'optimal' value that produces a clean, readable tree
    } else {
      req(input$tree_cp)
      input$tree_cp
    }
  })

  # build the full tree using all 13 variables on the training data
  full_tree_model <- reactive({
    rpart(
      target ~ .,
      data = train,
      method = "class",
      control = rpart.control(cp = current_cp())
    )
  })

  # calculate testing metrics for the top value boxes
  output$tab4_metrics_ui <- renderUI({
    m <- full_tree_model()

    # PREDICT ON THE UNSEEN TESTING DATA
    preds <- predict(m, test, type = "class")
    actual <- test$target

    cm <- table(Predicted = preds, Actual = actual)
    acc <- sum(diag(cm)) / sum(cm)

    # sensitivity: true positive rate (how many sick people did we catch?)
    sens <- if (sum(actual == "1") > 0) cm["1", "1"] / sum(actual == "1") else 0
    # specificity: true negative rate (how many healthy people did we clear?)
    spec <- if (sum(actual == "0") > 0) cm["0", "0"] / sum(actual == "0") else 0

    layout_column_wrap(
      width = 1 / 3,
      value_box(
        "Real-World Accuracy",
        paste0(round(acc * 100, 1), "%"),
        theme = "success",
        p("performance on unseen testing data")
      ),
      value_box(
        "Sensitivity (Caught Heart Disease)",
        paste0(round(sens * 100, 1), "%"),
        theme = "danger",
        p("correctly identified heart disease cases")
      ),
      value_box(
        "Specificity (Cleared Healthy)",
        paste0(round(spec * 100, 1), "%"),
        theme = value_box_theme(bg = "#4A90E2", fg = "white"),
        p("correctly identified healthy cases")
      )
    )
  })

  # handle the patient prediction button click
  patient_leaf <- reactiveVal(NULL)
  patient_pred <- reactiveVal(NULL)

  observeEvent(input$btn_predict, {
    # 1. take the first row of training data to get the exact data structure & factor levels
    one <- train[1, , drop = FALSE]

    # 2. override the 5 variables exposed in the UI (the ones the tree actually uses)
    one$cp <- factor(input$p_cp, levels = levels(train$cp))
    one$ca <- factor(input$p_ca, levels = levels(train$ca))
    one$thal <- factor(input$p_thal, levels = levels(train$thal))
    one$slope <- factor(input$p_slope, levels = levels(train$slope))
    one$oldpeak <- input$p_oldpeak

    # 3. safely hold all the ignored variables at median/mode so the model doesn't crash
    one$age <- median(train$age, na.rm = TRUE)
    one$sex <- factor("1", levels = levels(train$sex))
    one$trestbps <- median(train$trestbps, na.rm = TRUE)
    one$chol <- median(train$chol, na.rm = TRUE)
    one$thalach <- median(train$thalach, na.rm = TRUE)
    one$fbs <- factor("0", levels = levels(train$fbs))
    one$restecg <- factor("0", levels = levels(train$restecg))
    one$exang <- factor("0", levels = levels(train$exang))

    # 4. BUG FIX: duplicate the row to bypass the rpart 1-row dimension bug
    two_rows <- rbind(one, one)

    # 5. get the matrix prediction and match its counts to the tree frame
    m <- full_tree_model()
    # predict class
    pred_class <- as.character(predict(m, two_rows, type = "class")[1])
    patient_pred(pred_class)
    pred_mat <- predict(m, two_rows, type = "matrix")

    # columns 2 and 3 of the matrix hold the exact counts of healthy/sick in that specific leaf
    leaf_idx <- which(
      m$frame$var == "<leaf>" &
        m$frame$yval2[, 2] == pred_mat[1, 2] &
        m$frame$yval2[, 3] == pred_mat[1, 3]
    )

    if (length(leaf_idx) > 0) {
      patient_leaf(as.integer(rownames(m$frame)[leaf_idx[1]]))
    } else {
      patient_leaf(NULL)
    }
  })

  # predicted condition output
  output$predicted_condition_ui <- renderUI({
    p <- patient_pred()
    if (is.null(p)) {
      return(NULL)
    }

    label <- if (p == "1") "Heart disease" else "No heart disease"
    col <- if (p == "1") "#c3436a" else "#4A90E2"

    tags$div(
      style = "margin-top:12px;",
      tags$div(style = "font-size:0.95rem; color:#666;", "Predicted condition"),
      tags$div(
        style = sprintf(
          "margin-top:6px; display:inline-block; padding:8px 12px; border-radius:999px;
       font-weight:700; font-size:1.1rem; color:white; background:%s;",
          col
        ),
        label
      )
    )
  })

  # if the tree structure changes (e.g., user changes cp), reset the patient highlight
  observeEvent(full_tree_model(), {
    patient_leaf(NULL)
    patient_pred(NULL)
  })

  # render the full tree, highlighting the path if the button was clicked
  output$plot_full_tree <- renderPlot({
    m <- full_tree_model()
    node_ids <- as.integer(rownames(m$frame))

    # default colors (blue vs red)
    box_colors <- ifelse(m$frame$yval == 1, "#c3436a", "#4A90E2")
    names(box_colors) <- as.character(node_ids)

    # if a patient was predicted, calculate the path and highlight it in gold
    leaf <- patient_leaf()
    if (!is.null(leaf)) {
      path <- integer(0)
      curr <- leaf
      while (!is.na(curr) && curr >= 1) {
        path <- c(path, curr)
        if (curr == 1) {
          break
        }
        curr <- floor(curr / 2)
      }

      # dim all boxes slightly, then turn the patient's path bright gold
      box_colors[] <- "#EFEFEF"
      hits <- intersect(as.character(path), names(box_colors))
      box_colors[hits] <- "#FFD700"
    }

    rpart.plot(
      m,
      type = 2,
      extra = 104,
      fallen.leaves = TRUE,
      main = "",
      box.col = box_colors,
      shadow.col = "gray90"
    )
  })

  outputOptions(output, "plot_full_tree", suspendWhenHidden = FALSE)

  # ========================================================================== #
  ## Tab 5: Ensembles server logic ----

  # Helper function to generate the green/red comparison UI
  render_delta_ui <- function(title_text, forest_val, tree_val, theme_color) {
    diff <- forest_val - tree_val
    color <- if (diff > 0) {
      "#27ae60"
    } else if (diff < 0) {
      "#e74c3c"
    } else {
      "gray"
    }
    arrow <- if (diff > 0) {
      "▲"
    } else if (diff < 0) {
      "▼"
    } else {
      "-"
    }

    value_box(
      title_text,
      paste0(round(forest_val * 100, 1), "%"),
      theme = theme_color,
      tags$div(
        style = paste(
          "color:",
          color,
          "; font-size: 0.9rem; font-weight: bold; background: white; padding: 2px 6px; border-radius: 4px; display: inline-block;"
        ),
        paste0(
          arrow,
          " ",
          round(abs(diff) * 100, 1),
          "% vs selected individual tree"
        )
      )
    )
  }

  # --- RF Logic ---
  rf_model <- reactive({
    randomForest(
      target ~ .,
      data = train,
      ntree = input$rf_ntree,
      importance = TRUE
    )
  })

  # Simulate an individual tree
  rf_individual_tree <- reactive({
    set.seed(input$rf_tree_index)
    boot_idx <- sample(1:nrow(train), replace = TRUE)
    boot_train <- train[boot_idx, ]
    # need model = TRUE below to fix warning
    rpart(
      target ~ .,
      data = boot_train,
      method = "class",
      control = rpart.control(cp = 0.015),
      model = TRUE
    )
  })

  # Dynamic Card Header for the tree
  output$rf_tree_header_ui <- renderUI({
    card_header(tags$strong(paste(
      "Selected individual tree (Tree #",
      input$rf_tree_index,
      ")"
    )))
  })

  output$rf_comparative_metrics_ui <- renderUI({
    prob_forest <- get_prob1_safe(predict(rf_model(), test, type = "prob"))
    m_forest <- compute_metrics(
      test$target,
      ifelse(prob_forest >= 0.5, "1", "0"),
      prob_forest
    )

    prob_tree <- get_prob1_safe(predict(
      rf_individual_tree(),
      test,
      type = "prob"
    ))
    m_tree <- compute_metrics(
      test$target,
      ifelse(prob_tree >= 0.5, "1", "0"),
      prob_tree
    )

    layout_column_wrap(
      width = 1 / 3,
      render_delta_ui(
        "Forest Accuracy",
        m_forest$acc,
        m_tree$acc,
        "success"
      ),
      render_delta_ui(
        "Forest Sensitivity (Caught Heart Disease)",
        m_forest$sens,
        m_tree$sens,
        "danger"
      ),
      render_delta_ui(
        "Forest Specificity (Cleared Healthy)",
        m_forest$spec,
        m_tree$spec,
        value_box_theme(bg = "#4A90E2", fg = "white")
      )
    )
  })

  output$plot_rf_individual_tree <- renderPlot({
    m <- rf_individual_tree()
    box_colors <- ifelse(m$frame$yval == 1, "#c3436a", "#4A90E2")
    # Bugfix
    # removes main = ...
    # adds roundint = FALSE
    rpart.plot(
      m,
      type = 2,
      extra = 104,
      fallen.leaves = TRUE,
      main = "",
      box.col = box_colors,
      roundint = FALSE
    )
  })

  output$plot_rf_imp <- renderPlot({
    # explicit call to randomForest::importance to avoid package conflicts
    imp_df <- as.data.frame(randomForest::importance(rf_model()))
    imp_df$Feature <- rownames(imp_df)
    imp_df <- imp_df[order(imp_df$MeanDecreaseGini, decreasing = TRUE), ]
    ggplot(
      imp_df,
      aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)
    ) +
      geom_col(fill = "#2c3e50") +
      coord_flip() +
      labs(x = "", y = "Importance (mean decrease gini)") +
      theme_minimal()
  })

  # --- XGBoost Logic ---

  # FIXED: Explicitly use set_engine to prevent param leakage warnings
  build_xgb_model <- function(tree_depth = 3, learn_rate = 0.3, trees = 10) {
    xgb.model <- boost_tree(
      mode = 'classification',
      trees = trees,
      tree_depth = tree_depth,
      learn_rate = learn_rate
    ) |>
      set_engine('xgboost')

    xgb.wf <- workflow() |> add_recipe(heart.recipe) |> add_model(xgb.model)
    xgb.wf |> fit(train)
  }

  xgb.fit <- reactiveVal(build_xgb_model())
  xgb.fit.obj <- reactive({
    extract_fit_engine(xgb.fit())
  })

  # # DYNAMICALLY UPDATE SLIDER MAX WHEN BOOSTING ROUNDS CHANGE
  # observeEvent(input$xgb_num_boost_round, {
  #   updateSliderInput(
  #     session,
  #     "xgb_tree_index",
  #     max = input$xgb_num_boost_round
  #   )
  # })

  observeEvent(input$btn_train_xgb, {
    tryCatch(
      {
        xgb.fit(build_xgb_model(
          tree_depth = input$xgb_max_depth,
          learn_rate = input$xgb_eta,
          trees = input$xgb_num_boost_round
        ))
      },
      error = function(e) {
        showNotification(
          paste("Could not train XGBoost model:", e$message),
          type = "error"
        )
      }
    )

    # DYNAMICALLY UPDATE SLIDER MAX WHEN BOOSTING ROUNDS CHANGE
    updateSliderInput(
      session,
      "xgb_tree_index",
      max = input$xgb_num_boost_round
    )
  })

  output$plot_xgb_tree <- renderGrViz({
    req(xgb.fit.obj())

    # 1-indexed trees in XGBoost, capped at max rounds
    idx <- as.integer(min(input$xgb_tree_index, input$xgb_num_boost_round))

    xgboost::xgb.plot.tree(
      model = xgb.fit.obj(),
      tree_idx = idx, 
      with_stats = FALSE
    )
  })

  xgb_perf_obj <- reactive({
    prob_mat <- predict(xgb.fit(), new_data = test, type = 'prob')
    prob1 <- unlist(prob_mat[, 2])
    pred_class <- ifelse(prob1 >= input$xg_threshold, "1", "0")
    compute_metrics(test$target, pred_class, prob1)
  })

  output$xgb_perf_ui <- renderUI({
    m <- xgb_perf_obj()
    metrics_tbl <- tags$table(
      class = "table table-sm table-striped align-middle",
      tags$tbody(
        tags$tr(
          tags$th("Threshold"),
          tags$td(fmt(input$xg_threshold, digits = 2))
        ),
        tags$tr(tags$th("Accuracy"), tags$td(fmt(m$acc))),
        tags$tr(tags$th("Sensitivity"), tags$td(fmt(m$sens))),
        tags$tr(tags$th("Specificity"), tags$td(fmt(m$spec))),
        tags$tr(tags$th("AUC"), tags$td(fmt(m$auc)))
      )
    )
    tags$div(
      metrics_tbl,
      tags$div(
        style = "margin-top: 8px;",
        plotOutput("plot_xgb_cm_heat", height = 200)
      )
    )
  })

  output$plot_xgb_cm_heat <- renderPlot({
    plot_cm_heatmap(xgb_perf_obj()$cm, "XGBoost — Confusion Matrix")
  })

  output$plot_xg_tree_roc <- renderPlot({
    m <- xgb_perf_obj()
    safe_plot_roc(m$roc, m$auc, "XGBoost ROC")
  })

  output$xgb_feature_imp <- renderPlot({
    Xtrain <- heart.recipe |> prep() |> bake(train) |> select(-target)
    shp <- shapviz(xgb.fit.obj(), X_pred = data.matrix(Xtrain), X = Xtrain)
    sv_importance(shp, kind = 'both', fill = '#4A90E2')
  })
}

# Run app ====
shinyApp(ui, server)
