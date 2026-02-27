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
        h5("Growing a tree"),
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

        actionButton("btn_predict", "Predict My Path", class = "btn-primary"),

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
          card_header(tags$strong("The Full Clinical Decision Tree")),
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
  ## Tab 5: Advanced ensembles ----
  nav_panel(
    title = "5. Ensembles (RF & XGBoost)",
    value = "tab5",
    navset_card_underline(
      nav_panel(
        "Random Forest",
        layout_column_wrap(
          width = 1 / 2,
          plotOutput("plot_rf_imp"),
          plotOutput("plot_rf_cm_heat")
        )
      ),
      nav_panel(
        "XGBoost",
        p("Boosted trees build sequentially, learning from previous mistakes."),
        grVizOutput("plot_xgb_tree")
      )
    )
  ),

  # ========================================================================== #
  ## Tab 6: Takeaway / conclusions ----
  nav_panel(
    title = "6. Takeaways",
    value = "tab6",
    card(
      h3("Advantages & Limitations"),
      tags$ul(
        tags$li(
          tags$b("Decision Trees: "),
          "Excellent for communicating to stakeholders. Highly transparent. Prone to overfitting and slightly lower accuracy."
        ),
        tags$li(
          tags$b("Random Forests: "),
          "Greatly improved accuracy and stability by averaging many trees. Harder to draw a single 'path' for a patient."
        ),
        tags$li(
          tags$b("XGBoost: "),
          "Often best-in-class performance. Requires careful tuning to avoid overfitting."
        )
      )
    )
  ),

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

  # if the tree structure changes (e.g., user changes cp), reset the patient highlight
  observeEvent(full_tree_model(), {
    patient_leaf(NULL)
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
  ## Tab 5: Ensembles ----
  rf_model <- reactive({
    randomForest(target ~ ., data = train, ntree = 100, importance = TRUE)
  })
  output$plot_rf_imp <- renderPlot({
    varImpPlot(rf_model(), main = "Feature Importance")
  })
  output$plot_rf_cm_heat <- renderPlot({
    prob1 <- get_prob1_safe(predict(rf_model(), test, type = "prob"))
    m <- compute_metrics(test$target, ifelse(prob1 >= 0.5, "1", "0"), prob1)
    plot_cm_heatmap(m$cm, "RF Confusion Matrix")
  })

  xgb_mod <- boost_tree(
    mode = 'classification',
    engine = 'xgboost',
    trees = 5
  ) |>
    fit(target ~ ., data = prep(heart.recipe) |> bake(train))

  output$plot_xgb_tree <- renderGrViz({
    xgb.plot.tree(extract_fit_engine(xgb_mod), tree_idx = 1)
  })
}

# Run app ====
shinyApp(ui, server)
