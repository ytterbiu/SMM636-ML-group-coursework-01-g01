# LOAD REQUIRED LIBRARIES -----------------------------------------------

library(shiny)
library(ggplot2)
library(tidymodels)
library(tidyr)
library(rpart)
library(ranger)
library(jsonlite)
library(bslib)
library(bsicons)

# LOAD DATA ---------------------------------------------------------------

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
addResourcePath("www", file.path(getwd(), "www"))

h <- read.csv("Heart_disease_cleveland_new.csv")
h$target <- as.factor(h$target)

# VARIABLE PLAIN-ENGLISH LOOKUP ------------------------------------------

var_info <- list(
  age      = list(label = "Age",               icon = "🎂", unit = "years",
                  desc = "The patient's age in years."),
  sex      = list(label = "Sex",               icon = "👤", unit = "(0=F, 1=M)",
                  desc = "Patient's biological sex (0 = Female, 1 = Male)."),
  cp       = list(label = "Chest Pain Type",   icon = "💢", unit = "(0–3)",
                  desc = "Type of chest pain: 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic."),
  trestbps = list(label = "Resting Blood Pressure", icon = "🩺", unit = "mmHg",
                  desc = "Blood pressure measured when the patient is at rest."),
  chol     = list(label = "Cholesterol",       icon = "🧪", unit = "mg/dl",
                  desc = "Serum cholesterol level in mg/dl."),
  fbs      = list(label = "Fasting Blood Sugar", icon = "🍬", unit = "(>120 mg/dl = 1)",
                  desc = "Whether fasting blood sugar > 120 mg/dl (1 = true, 0 = false)."),
  restecg  = list(label = "Resting ECG",       icon = "📈", unit = "(0–2)",
                  desc = "Resting electrocardiogram results (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)."),
  thalach  = list(label = "Max Heart Rate",    icon = "❤️", unit = "bpm",
                  desc = "Maximum heart rate achieved during exercise."),
  exang    = list(label = "Exercise Angina",   icon = "🏃", unit = "(0/1)",
                  desc = "Whether the patient experienced chest pain during exercise (1 = yes, 0 = no)."),
  oldpeak  = list(label = "ST Depression",     icon = "📉", unit = "mm",
                  desc = "ST segment depression induced by exercise relative to rest — indicates stress on the heart."),
  slope    = list(label = "ST Slope",          icon = "📐", unit = "(0–2)",
                  desc = "Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)."),
  ca       = list(label = "Major Vessels",     icon = "🔬", unit = "(0–3)",
                  desc = "Number of major blood vessels coloured by fluoroscopy (0–3). More coloured vessels = more blockage."),
  thal     = list(label = "Thalassemia",       icon = "🧬", unit = "(0–3)",
                  desc = "Blood disorder test result (1 = normal, 2 = fixed defect, 3 = reversible defect).")
)

# HELPER: build plain-English explanation for a node --------------------

node_explanation <- function(var, thresh, is_leaf, majority, n, is_left, parent_var, parent_thresh) {
  if (is_leaf) {
    pct <- round(100 * n / n, 0)  # placeholder; actual % passed from JS
    paste0(
      "Based on all the decisions made above, this group of ", n, " patients ",
      "is predicted to have <strong>", majority, "</strong>. ",
      "This is the final prediction for patients who followed this path through the tree."
    )
  } else {
    info <- var_info[[var]]
    lbl  <- if (!is.null(info)) info$label else var
    desc <- if (!is.null(info)) info$desc  else ""
    paste0(
      "<strong>", lbl, "</strong> — ", desc, " ",
      "The tree splits patients here: those with <em>", lbl, " ≤ ", round(thresh, 2),
      "</em> go left, and those with <em>", lbl, " > ", round(thresh, 2), "</em> go right."
    )
  }
}

# HELPER: rpart → flat node list (JSON-ready) ----------------------------

rpart_to_nodelist <- function(tree) {
  frame    <- tree$frame
  nodes_r  <- as.integer(rownames(frame))
  splits_m <- tree$splits
  
  non_leaf_nodes <- nodes_r[frame[as.character(nodes_r), "var"] != "<leaf>"]
  # Plain named list: node id (as char) -> numeric threshold
  node_thresh <- vector("list", length(nodes_r))
  names(node_thresh) <- as.character(nodes_r)
  if (!is.null(splits_m) && length(non_leaf_nodes) > 0) {
    n_map <- min(length(non_leaf_nodes), nrow(splits_m))
    for (i in seq_len(n_map)) {
      node_thresh[[as.character(non_leaf_nodes[i])]] <- as.numeric(splits_m[i, "index"])
    }
  }
  
  nodelist <- lapply(nodes_r, function(nd) {
    nd_chr   <- as.character(nd)
    var      <- as.character(frame[nd_chr, "var"])
    is_leaf  <- (var == "<leaf>")
    yval     <- as.integer(frame[nd_chr, "yval"])
    majority <- ifelse(yval == 2, "Disease", "No Disease")
    n_obs    <- frame[nd_chr, "n"]
    thresh   <- node_thresh[[nd_chr]]
    depth    <- floor(log2(nd))
    parent   <- if (nd == 1L) NA_integer_ else nd %/% 2L
    lc       <- nd * 2L
    rc       <- nd * 2L + 1L
    has_left  <- lc %in% nodes_r
    has_right <- rc %in% nodes_r
    
    info  <- if (!is_leaf) var_info[[var]] else NULL
    label <- if (!is.null(info)) info$label else var
    icon  <- if (!is.null(info)) info$icon  else "🔍"
    unit  <- if (!is.null(info)) info$unit  else ""
    
    explanation <- if (is_leaf) {
      paste0("Based on all decisions above, these <strong>", n_obs,
             " patients</strong> are predicted to have <strong>", majority, "</strong>.")
    } else {
      vdesc <- if (!is.null(info)) info$desc else ""
      paste0("<strong>", label, "</strong> (", unit, ") — ", vdesc,
             "<br><br>Split rule: patients with <em>", label, " ≤ ", round(thresh, 2),
             "</em> go left &nbsp;|&nbsp; <em>", label, " > ", round(thresh, 2), "</em> go right.")
    }
    
    list(
      id          = nd,
      parent      = if (is.na(parent)) NULL else parent,
      isLeft      = if (nd == 1L) NULL else as.logical(nd %% 2L == 0L),
      var         = var,
      varLabel    = label,
      icon        = icon,
      unit        = unit,
      threshold   = if (is.null(thresh) || is.na(thresh)) NULL else as.numeric(round(thresh, 2)),
      isLeaf      = is_leaf,
      majority    = majority,
      n           = n_obs,
      depth       = depth,
      leftChild   = if (has_left)  lc else NULL,
      rightChild  = if (has_right) rc else NULL,
      explanation = explanation
    )
  })
  
  nodelist
}

# UI ----------------------------------------------------------------------

ui <- page_sidebar(
  title = tags$span(
    tags$span("❤️", style = "margin-right:8px;"),
    "Heart Disease Decision Tree"
  ),
  
  sidebar = sidebar(
    width = 280,
    
    sliderInput("split_ratio", "Training Data Split:",
                min = 0.1, max = 0.99, value = 0.5),
    
    conditionalPanel(
      condition = "input.tabs == 'Decision Tree'",
      sliderInput(
        inputId = "tree_depth",
        label = tooltip(
          trigger = list("Tree Depth:", bsicons::bs_icon("info-circle")),
          "rpart may not change with every depth step due to automatic pruning."
        ),
        min = 1, max = 30, value = 5
      ),
      hr(),
      # Walkthrough controls
      div(class = "walkthrough-controls",
          p(style = "font-weight:600; margin-bottom:8px; font-size:0.9rem;",
            "🚶 Guided Walkthrough"),
          div(style = "display:flex; gap:6px; margin-bottom:8px;",
              actionButton("wt_prev", "◀ Prev",
                           style = "flex:1; font-size:0.8rem; padding:4px;"),
              actionButton("wt_next", "Next ▶",
                           style = "flex:1; font-size:0.8rem; padding:4px;")
          ),
          div(style = "display:flex; gap:6px;",
              actionButton("wt_reset", "↺ Reset",
                           style = "flex:1; font-size:0.8rem; padding:4px;"),
              actionButton("wt_full", "Show All",
                           style = "flex:1; font-size:0.8rem; padding:4px;")
          ),
          textOutput("wt_counter"),
          hr()
      ),
      # Legend
      h6("Legend", style = "font-weight:700;"),
      tags$div(style = "display:flex;flex-direction:column;gap:6px;font-size:0.85rem;",
               tags$div(style = "display:flex;align-items:center;gap:8px;",
                        tags$div(style = "width:14px;height:14px;background:#1a5f7a;border-radius:3px;border:2px solid #57c5e0;"),
                        tags$span("Split node (decision)")),
               tags$div(style = "display:flex;align-items:center;gap:8px;",
                        tags$div(style = "width:14px;height:14px;background:#1e7a4e;border-radius:50%;border:2px solid #4ade80;"),
                        tags$span("Leaf: No Disease")),
               tags$div(style = "display:flex;align-items:center;gap:8px;",
                        tags$div(style = "width:14px;height:14px;background:#7a1e1e;border-radius:50%;border:2px solid #f87171;"),
                        tags$span("Leaf: Disease")),
               tags$div(style = "display:flex;align-items:center;gap:8px;",
                        tags$div(style = "width:14px;height:14px;background:#fbbf24;border-radius:3px;border:2px solid #fbbf24;"),
                        tags$span("Currently highlighted"))
      )
    ),
    
    conditionalPanel(
      condition = "input.tabs == 'Random Forest'",
      sliderInput("rf_trees", label = tooltip(
        trigger = list("Number of Trees:", bsicons::bs_icon("info-circle")),
        "More trees = more stable but slower."),
        min = 50, max = 500, value = 100),
      sliderInput("rf_mtry", label = tooltip(
        trigger = list("Variables per Split (mtry):", bsicons::bs_icon("info-circle")),
        "Features randomly considered at each split."),
        min = 1, max = 13, value = 3),
      sliderInput("rf_min_n", label = tooltip(
        trigger = list("Minimum Node Size:", bsicons::bs_icon("info-circle")),
        "Smallest leaf size."),
        min = 1, max = 20, value = 5)
    )
  ),
  
  tabsetPanel(
    id = "tabs",
    
    # ── Decision Tree Tab ──────────────────────────────────────────────
    tabPanel(
      "Decision Tree",
      
      # Metrics
      card(
        card_header("📊 Model Performance"),
        tableOutput("dt_metrics_table")
      ),
      
      # Tree card
      card(
        card_header("🌳 Interactive Decision Tree — click any node to learn more"),
        
        # Inject CSS + D3 tree
        tags$head(
          tags$style(HTML("
            /* Medical theme */
            body { font-family: 'Segoe UI', sans-serif; }

            #tree-container {
              width: 100%;
              height: 600px;
              background: linear-gradient(135deg, #0a1628 0%, #0d2137 50%, #0a1e2e 100%);
              border-radius: 12px;
              position: relative;
              overflow: hidden;
            }

            #tree-container::before {
              content: '';
              position: absolute;
              top: 0; left: 0; right: 0; bottom: 0;
              background-image:
                radial-gradient(circle at 20% 20%, rgba(87,197,224,0.04) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(74,222,128,0.04) 0%, transparent 50%);
              pointer-events: none;
            }

            /* ECG line decoration */
            #ecg-decoration {
              position: absolute;
              bottom: 12px;
              left: 0; right: 0;
              height: 30px;
              opacity: 0.15;
              pointer-events: none;
            }

            /* Node styles via D3 */
            .tree-node-split rect {
              fill: #1a3a5c;
              stroke: #57c5e0;
              stroke-width: 2px;
              rx: 8; ry: 8;
              filter: drop-shadow(0 4px 12px rgba(87,197,224,0.3));
              cursor: pointer;
              transition: all 0.2s;
            }
            .tree-node-split rect:hover { stroke: #a0e8f8; stroke-width: 3px; }
            .tree-node-split.highlighted rect {
              fill: #2a4a1c;
              stroke: #fbbf24;
              stroke-width: 3px;
              filter: drop-shadow(0 0 16px rgba(251,191,36,0.6));
            }
            .tree-node-split.dimmed rect { opacity: 0.25; }

            .tree-node-leaf-healthy ellipse {
              fill: #0f3d24;
              stroke: #4ade80;
              stroke-width: 2px;
              filter: drop-shadow(0 4px 12px rgba(74,222,128,0.3));
              cursor: pointer;
            }
            .tree-node-leaf-healthy.highlighted ellipse {
              stroke: #fbbf24;
              stroke-width: 3px;
              filter: drop-shadow(0 0 16px rgba(251,191,36,0.6));
            }
            .tree-node-leaf-healthy.dimmed ellipse { opacity: 0.25; }

            .tree-node-leaf-disease ellipse {
              fill: #3d0f0f;
              stroke: #f87171;
              stroke-width: 2px;
              filter: drop-shadow(0 4px 12px rgba(248,113,113,0.3));
              cursor: pointer;
            }
            .tree-node-leaf-disease.highlighted ellipse {
              stroke: #fbbf24;
              stroke-width: 3px;
              filter: drop-shadow(0 0 16px rgba(251,191,36,0.6));
            }
            .tree-node-leaf-disease.dimmed ellipse { opacity: 0.25; }

            .tree-node-hidden { display: none; }

            .node-text { fill: #e2f4fb; font-size: 12px; font-weight: 600; pointer-events: none; }
            .node-subtext { fill: #8ecfdf; font-size: 10px; pointer-events: none; }
            .node-icon { font-size: 14px; pointer-events: none; }
            .node-n { fill: #94a3b8; font-size: 9px; pointer-events: none; }

            .edge-line {
              stroke: #2a4a6a;
              stroke-width: 2px;
              fill: none;
              transition: opacity 0.3s;
            }
            .edge-line.highlighted { stroke: #57c5e0; stroke-width: 2.5px; }
            .edge-line.dimmed { opacity: 0.15; }

            .edge-label {
              fill: #57c5e0;
              font-size: 10px;
              font-weight: 600;
              background: #0d2137;
            }
            .edge-label.dimmed { opacity: 0.15; }

            /* Info panel */
            #node-info-panel {
              position: absolute;
              top: 12px; right: 12px;
              width: 260px;
              background: rgba(10,22,40,0.95);
              border: 1px solid #57c5e0;
              border-radius: 10px;
              padding: 14px 16px;
              color: #e2f4fb;
              font-size: 0.82rem;
              line-height: 1.5;
              display: none;
              backdrop-filter: blur(8px);
              box-shadow: 0 8px 32px rgba(0,0,0,0.5);
              z-index: 10;
              animation: fadeIn 0.2s ease;
            }
            @keyframes fadeIn { from { opacity:0; transform:translateY(-6px); } to { opacity:1; transform:translateY(0); } }
            #node-info-panel .panel-title {
              font-size: 0.95rem;
              font-weight: 700;
              color: #57c5e0;
              margin-bottom: 6px;
              display: flex;
              align-items: center;
              gap: 6px;
            }
            #node-info-panel .panel-badge {
              display: inline-block;
              padding: 2px 8px;
              border-radius: 12px;
              font-size: 0.75rem;
              font-weight: 700;
              margin-bottom: 8px;
            }
            #node-info-panel .badge-split   { background:#1a3a5c; color:#57c5e0; border:1px solid #57c5e0; }
            #node-info-panel .badge-healthy { background:#0f3d24; color:#4ade80; border:1px solid #4ade80; }
            #node-info-panel .badge-disease { background:#3d0f0f; color:#f87171; border:1px solid #f87171; }
            #node-info-panel .panel-n {
              font-size: 0.75rem;
              color: #94a3b8;
              margin-top: 8px;
              border-top: 1px solid #1e3a52;
              padding-top: 6px;
            }
            #node-info-panel .close-btn {
              float: right;
              cursor: pointer;
              color: #94a3b8;
              font-size: 1rem;
              line-height: 1;
            }
            #node-info-panel .close-btn:hover { color: #e2f4fb; }

            .wt-counter-text {
              font-size: 0.8rem;
              color: #64748b;
              margin-top: 6px;
              text-align: center;
            }
          "))
        ),
        
        div(id = "tree-container",
            # SVG will be injected here by D3
            htmlOutput("tree_svg"),
            # Info panel overlay
            div(id = "node-info-panel",
                span(class = "close-btn", onclick = "document.getElementById('node-info-panel').style.display='none'", "✕"),
                div(class = "panel-title", id = "panel-title", "Node Info"),
                div(id = "panel-badge"),
                div(id = "panel-body"),
                div(class = "panel-n", id = "panel-n")
            ),
            # ECG decoration
            tags$svg(id = "ecg-decoration", viewBox = "0 0 1000 30", preserveAspectRatio = "none",
                     tags$path(d = "M0,15 L100,15 L120,15 L130,2 L140,28 L150,15 L160,15 L200,15 L220,15 L230,2 L240,28 L250,15 L260,15 L300,15 L320,15 L330,2 L340,28 L350,15 L360,15 L400,15 L420,15 L430,2 L440,28 L450,15 L460,15 L500,15 L520,15 L530,2 L540,28 L550,15 L560,15 L600,15 L620,15 L630,2 L640,28 L650,15 L660,15 L700,15 L720,15 L730,2 L740,28 L750,15 L760,15 L800,15 L820,15 L830,2 L840,28 L850,15 L860,15 L900,15 L1000,15",
                               stroke = "#57c5e0", `stroke-width` = "1.5", fill = "none")
            )
        ),
        
        # D3 script
        tags$script(src = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"),
        tags$script(HTML("
// ── Tree renderer ───────────────────────────────────────────────────────

var treeData   = null;
var wtStep     = 0;    // current walkthrough step (-1 = show all)
var wtSequence = [];   // ordered node ids for walkthrough

// Called from R when tree data is ready
Shiny.addCustomMessageHandler('updateTree', function(data) {
  treeData   = data.nodes;
  wtStep     = 0;
  wtSequence = buildWalkthrough(treeData);
  Shiny.setInputValue('wt_total', wtSequence.length);
  renderTree();
});

// Called from R walkthrough buttons
Shiny.addCustomMessageHandler('setWtStep', function(step) {
  wtStep = step;
  renderTree();
  Shiny.setInputValue('wt_step_display', wtStep);
});

// Build depth-first walkthrough order
function buildWalkthrough(nodes) {
  var map = {};
  nodes.forEach(function(n) { map[n.id] = n; });
  var order = [];
  function dfs(id) {
    if (!map[id]) return;
    order.push(id);
    var n = map[id];
    if (n.leftChild)  dfs(n.leftChild);
    if (n.rightChild) dfs(n.rightChild);
  }
  dfs(1);
  return order;
}

function renderTree() {
  if (!treeData) return;

  var container = document.getElementById('tree-container');
  var W = container.clientWidth  || 900;
  var H = 580;

  // Remove old SVG (but keep info panel + ecg)
  d3.select('#tree-container svg.main-tree').remove();

  var svg = d3.select('#tree-container')
    .insert('svg', ':first-child')
    .attr('class', 'main-tree')
    .attr('width',  W)
    .attr('height', H);

  var g = svg.append('g');

  // Build node map
  var nodeMap = {};
  treeData.forEach(function(n) { nodeMap[n.id] = n; });

  // Compute positions using d3.tree
  var root = d3.hierarchy(nodeMap[1], function(n) {
    var children = [];
    if (n.leftChild  && nodeMap[n.leftChild])  children.push(nodeMap[n.leftChild]);
    if (n.rightChild && nodeMap[n.rightChild]) children.push(nodeMap[n.rightChild]);
    return children.length ? children : null;
  });

  var treeLayout = d3.tree().size([W - 80, H - 100]);
  treeLayout(root);

  // Shift so root is centred and has top margin
  var descendants = root.descendants();
  var minY = d3.min(descendants, function(d) { return d.y; });
  descendants.forEach(function(d) {
    d.x += 40;
    d.y += 60 - minY;
  });

  // Determine which nodes are visible / highlighted in walkthrough
  var visibleIds = new Set();
  var highlightId = null;
  if (wtStep < 0 || wtStep >= wtSequence.length) {
    // Show all
    treeData.forEach(function(n) { visibleIds.add(n.id); });
  } else {
    // Show nodes up to and including current step
    for (var i = 0; i <= wtStep; i++) visibleIds.add(wtSequence[i]);
    highlightId = wtSequence[wtStep];
  }

  // Draw edges first (under nodes)
  var links = root.links();
  links.forEach(function(link) {
    var srcId = link.source.data.id;
    var tgtId = link.target.data.id;
    var visible = visibleIds.has(srcId) && visibleIds.has(tgtId);
    var isHighlighted = (highlightId !== null) &&
                        (tgtId === highlightId || srcId === highlightId);
    var isDimmed = !isHighlighted && highlightId !== null;

    if (!visible) return;

    var edgeClass = 'edge-line' +
      (isHighlighted ? ' highlighted' : '') +
      (isDimmed      ? ' dimmed'      : '');

    g.append('path')
      .attr('class', edgeClass)
      .attr('d', d3.linkVertical()
        .x(function(d) { return d.x; })
        .y(function(d) { return d.y; })
        ({source: link.source, target: link.target}));

    // Edge label
    var midX = (link.source.x + link.target.x) / 2;
    var midY = (link.source.y + link.target.y) / 2;
    var isLeft = link.target.data.isLeft;
    var thresh = link.source.data.threshold;
    var edgeTxt = thresh !== null
      ? (isLeft ? '\\u2264 ' + thresh : '> ' + thresh)
      : (isLeft ? 'Yes' : 'No');

    g.append('text')
      .attr('class', 'edge-label' + (isDimmed ? ' dimmed' : ''))
      .attr('x', midX + (isLeft ? -14 : 14))
      .attr('y', midY)
      .attr('text-anchor', 'middle')
      .text(edgeTxt);
  });

  // Draw nodes
  descendants.forEach(function(d) {
    var n = d.data;
    if (!visibleIds.has(n.id)) return;

    var isHighlighted = (n.id === highlightId);
    var isDimmed      = (highlightId !== null && !isHighlighted);

    var nodeType = n.isLeaf
      ? (n.majority === 'Disease' ? 'leaf-disease' : 'leaf-healthy')
      : 'split';

    var cls = 'tree-node-' + nodeType +
      (isHighlighted ? ' highlighted' : '') +
      (isDimmed      ? ' dimmed'      : '');

    var grp = g.append('g')
      .attr('class', cls)
      .attr('transform', 'translate(' + d.x + ',' + d.y + ')')
      .style('cursor', 'pointer')
      .on('click', function() { showPanel(n); });

    if (n.isLeaf) {
      grp.append('ellipse').attr('rx', 52).attr('ry', 28);
    } else {
      grp.append('rect')
        .attr('x', -52).attr('y', -28)
        .attr('width', 104).attr('height', 56)
        .attr('rx', 8).attr('ry', 8);
    }

    // Icon
    grp.append('text')
      .attr('class', 'node-icon')
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .text(n.icon || (n.isLeaf ? (n.majority === 'Disease' ? '🫀' : '💚') : '🔍'));

    // Label
    var displayLabel = n.isLeaf ? n.majority : (n.varLabel || n.var);
    // Truncate if long
    if (displayLabel.length > 12) displayLabel = displayLabel.substring(0, 11) + '…';

    grp.append('text')
      .attr('class', 'node-text')
      .attr('y', 6)
      .attr('text-anchor', 'middle')
      .text(displayLabel);

    // n=
    grp.append('text')
      .attr('class', 'node-n')
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .text('n = ' + n.n);
  });

  // Auto-show panel for highlighted node
  if (highlightId !== null && nodeMap[highlightId]) {
    showPanel(nodeMap[highlightId]);
  }
}

function showPanel(n) {
  var panel = document.getElementById('node-info-panel');
  var title = document.getElementById('panel-title');
  var badge = document.getElementById('panel-badge');
  var body  = document.getElementById('panel-body');
  var nEl   = document.getElementById('panel-n');

  var icon = n.icon || (n.isLeaf ? (n.majority === 'Disease' ? '🫀' : '💚') : '🔍');
  title.innerHTML = icon + ' ' + (n.isLeaf ? n.majority : (n.varLabel || n.var));

  if (n.isLeaf) {
    var badgeCls = n.majority === 'Disease' ? 'badge-disease' : 'badge-healthy';
    badge.innerHTML = '<span class=\"panel-badge ' + badgeCls + '\">Final Prediction: ' + n.majority + '</span>';
  } else {
    badge.innerHTML = '<span class=\"panel-badge badge-split\">Decision Node</span>';
  }

  body.innerHTML = n.explanation || '';
  nEl.innerHTML  = '👥 Patients reaching this node: <strong>' + n.n + '</strong>';

  panel.style.display = 'block';
}

// Resize handler
window.addEventListener('resize', function() {
  if (treeData) renderTree();
});
        "))
      ),
    ),
  ),
  
  tabsetPanel(
    id = "tabs2",  # dummy to not clash
    type = "hidden"
  )
)

# Actually rebuild tabs properly — the ui above is a bit tangled; clean version:
ui <- page_sidebar(
  title = tags$span("❤️  Heart Disease — Decision Tree Explorer"),
  
  sidebar = sidebar(
    width = 280,
    sliderInput("split_ratio", "Training Data Split:",
                min = 0.1, max = 0.99, value = 0.5),
    
    conditionalPanel(
      condition = "input.tabs == 'Decision Tree'",
      sliderInput("tree_depth",
                  label = tooltip(trigger = list("Tree Depth:", bsicons::bs_icon("info-circle")),
                                  "rpart may not change at every depth due to pruning."),
                  min = 1, max = 30, value = 5),
      hr(),
      tags$p(style = "font-weight:700; font-size:0.88rem; margin-bottom:6px;",
             "🚶 Guided Walkthrough"),
      div(style = "display:flex; gap:6px; margin-bottom:6px;",
          actionButton("wt_prev", "◀ Prev",
                       style = "flex:1; font-size:0.8rem; padding:5px 2px;"),
          actionButton("wt_next", "Next ▶",
                       style = "flex:1; font-size:0.8rem; padding:5px 2px;")
      ),
      div(style = "display:flex; gap:6px; margin-bottom:4px;",
          actionButton("wt_reset", "↺ Reset",
                       style = "flex:1; font-size:0.8rem; padding:5px 2px;"),
          actionButton("wt_all",   "⬛ All",
                       style = "flex:1; font-size:0.8rem; padding:5px 2px;")
      ),
      uiOutput("wt_counter_ui"),
      hr(),
      tags$h6("Legend", style = "font-weight:700; margin-bottom:6px;"),
      tags$div(style = "font-size:0.82rem; display:flex; flex-direction:column; gap:5px;",
               tags$div(style = "display:flex;align-items:center;gap:7px;",
                        tags$div(style = "width:13px;height:13px;background:#1a3a5c;border:2px solid #57c5e0;border-radius:3px;"),
                        "Split node (decision)"),
               tags$div(style = "display:flex;align-items:center;gap:7px;",
                        tags$div(style = "width:13px;height:13px;background:#0f3d24;border:2px solid #4ade80;border-radius:50%;"),
                        "Leaf: No Disease"),
               tags$div(style = "display:flex;align-items:center;gap:7px;",
                        tags$div(style = "width:13px;height:13px;background:#3d0f0f;border:2px solid #f87171;border-radius:50%;"),
                        "Leaf: Disease"),
               tags$div(style = "display:flex;align-items:center;gap:7px;",
                        tags$div(style = "width:13px;height:13px;background:#2a4a1c;border:2px solid #fbbf24;border-radius:3px;"),
                        "Current node")
      ),
      hr(),
      tags$small(style = "color:#64748b;",
                 tags$b("Click"), " any node for explanation.", tags$br(),
                 tags$b("Scroll"), " to zoom.")
    ),
    
    conditionalPanel(
      condition = "input.tabs == 'Random Forest'",
      sliderInput("rf_trees", label = tooltip(
        trigger = list("Number of Trees:", bsicons::bs_icon("info-circle")),
        "More trees = more stable but slower."), min = 50, max = 500, value = 100),
      sliderInput("rf_mtry", label = tooltip(
        trigger = list("Variables per Split (mtry):", bsicons::bs_icon("info-circle")),
        "Features randomly considered at each split."), min = 1, max = 13, value = 3),
      sliderInput("rf_min_n", label = tooltip(
        trigger = list("Minimum Node Size:", bsicons::bs_icon("info-circle")),
        "Smallest leaf size."), min = 1, max = 20, value = 5)
    )
  ),
  
  tabsetPanel(
    id = "tabs",
    
    # ── Decision Tree ─────────────────────────────────────────────────
    tabPanel("Decision Tree",
             
             card(card_header("📊 Model Performance"),
                  tableOutput("dt_metrics_table")),
             
             card(
               card_header("🌳 Interactive Decision Tree — click any node to learn more"),
               
               # CSS
               tags$head(tags$style(HTML("
          #tree-container {
            width:100%; height:600px;
            background: linear-gradient(160deg,#071424 0%,#0d2137 60%,#071424 100%);
            border-radius:12px; position:relative; overflow:hidden;
          }
          #tree-container::after {
            content:''; position:absolute; inset:0;
            background: radial-gradient(ellipse at 50% 0%,rgba(87,197,224,0.06) 0%,transparent 60%);
            pointer-events:none;
          }
          .edge-line         { stroke:#1e3f5c; stroke-width:2; fill:none; }
          .edge-line.hl      { stroke:#57c5e0; stroke-width:2.5; }
          .edge-line.dim     { opacity:0.15; }
          .edge-label        { fill:#57c5e0; font-size:10px; font-family:'Segoe UI',sans-serif; font-weight:600; }
          .edge-label.dim    { opacity:0.15; }
          .node-split rect   { fill:#0e2840; stroke:#57c5e0; stroke-width:2; cursor:pointer; }
          .node-split.hl rect{ fill:#1e3d10; stroke:#fbbf24; stroke-width:3;
                                filter:drop-shadow(0 0 14px rgba(251,191,36,.55)); }
          .node-split.dim rect{ opacity:.2; }
          .node-healthy ellipse  { fill:#0a2e1a; stroke:#4ade80; stroke-width:2; cursor:pointer; }
          .node-healthy.hl  ellipse { stroke:#fbbf24; stroke-width:3;
                                       filter:drop-shadow(0 0 14px rgba(251,191,36,.55)); }
          .node-healthy.dim ellipse  { opacity:.2; }
          .node-disease ellipse  { fill:#2e0a0a; stroke:#f87171; stroke-width:2; cursor:pointer; }
          .node-disease.hl  ellipse { stroke:#fbbf24; stroke-width:3;
                                       filter:drop-shadow(0 0 14px rgba(251,191,36,.55)); }
          .node-disease.dim ellipse  { opacity:.2; }
          .node-label   { fill:#dff0f8; font-size:11px; font-weight:700;
                          font-family:'Segoe UI',sans-serif; pointer-events:none; }
          .node-sub     { fill:#7ec8de; font-size:9.5px;
                          font-family:'Segoe UI',sans-serif; pointer-events:none; }
          .node-icon-t  { font-size:13px; pointer-events:none; }
          #info-panel {
            position:absolute; top:12px; right:12px; width:255px;
            background:rgba(6,16,30,0.96); border:1px solid #57c5e0;
            border-radius:10px; padding:14px 15px; color:#dff0f8;
            font-size:.81rem; line-height:1.55; display:none;
            backdrop-filter:blur(10px);
            box-shadow:0 8px 32px rgba(0,0,0,.6);
            animation:panelIn .18s ease;
          }
          @keyframes panelIn{ from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
          #info-panel .ptitle{ font-size:.93rem; font-weight:700; color:#57c5e0;
                               margin-bottom:5px; display:flex; align-items:center; gap:5px; }
          #info-panel .pbadge{ display:inline-block; padding:2px 9px; border-radius:12px;
                               font-size:.73rem; font-weight:700; margin-bottom:7px; }
          .bs { background:#0e2840; color:#57c5e0; border:1px solid #57c5e0; }
          .bh { background:#0a2e1a; color:#4ade80; border:1px solid #4ade80; }
          .bd { background:#2e0a0a; color:#f87171; border:1px solid #f87171; }
          #info-panel .pn  { font-size:.73rem; color:#64748b; margin-top:8px;
                             border-top:1px solid #1a3a52; padding-top:6px; }
          #info-panel .pcl { float:right; cursor:pointer; color:#64748b;
                             font-size:1rem; line-height:1; }
          #info-panel .pcl:hover { color:#dff0f8; }
          /* ECG */
          #ecg-line { position:absolute; bottom:10px; left:0; right:0;
                      height:28px; opacity:.12; pointer-events:none; }
        "))),
               
               div(id = "tree-container",
                   htmlOutput("tree_placeholder"),
                   div(id = "info-panel",
                       span(class="pcl",
                            onclick="document.getElementById('info-panel').style.display='none'", "✕"),
                       div(class="ptitle", id="ptitle"),
                       div(id="pbadge"),
                       div(id="pbody"),
                       div(class="pn", id="pn")
                   ),
                   tags$svg(id="ecg-line", viewBox="0 0 1000 28", preserveAspectRatio="none",
                            tags$path(
                              d="M0,14 L80,14 L95,14 L105,1 L115,27 L125,14 L140,14 L220,14 L235,14 L245,1 L255,27 L265,14 L280,14 L360,14 L375,14 L385,1 L395,27 L405,14 L420,14 L500,14 L515,14 L525,1 L535,27 L545,14 L560,14 L640,14 L655,14 L665,1 L675,27 L685,14 L700,14 L780,14 L795,14 L805,1 L815,27 L825,14 L840,14 L920,14 L1000,14",
                              stroke="#57c5e0", `stroke-width`="1.5", fill="none"))
               ),
               
               # Load D3 + tree script
               tags$script(src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"),
               tags$script(HTML("
(function(){
  var treeData=null, wtStep=0, wtSeq=[];

  Shiny.addCustomMessageHandler('treeData', function(payload){
    treeData = payload.nodes;
    wtSeq    = buildSeq(treeData);
    wtStep   = 0;
    Shiny.setInputValue('wt_max', wtSeq.length - 1, {priority:'event'});
    draw();
  });

  Shiny.addCustomMessageHandler('wtGo', function(s){
    wtStep = s;
    draw();
  });

  function buildSeq(nodes){
    var m={};
    nodes.forEach(function(n){ m[n.id]=n; });
    var out=[];
    (function dfs(id){ if(!m[id]) return; out.push(id);
      var n=m[id];
      if(n.leftChild)  dfs(n.leftChild);
      if(n.rightChild) dfs(n.rightChild);
    })(1);
    return out;
  }

  function draw(){
    if(!treeData) return;
    var c=document.getElementById('tree-container');
    var W=c.clientWidth||900, H=580;
    d3.select('#tree-container svg.mt').remove();

    var svg=d3.select('#tree-container')
      .insert('svg',':first-child')
      .attr('class','mt').attr('width',W).attr('height',H);

    // zoom
    var gWrap=svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.3,3])
      .on('zoom',function(e){ gWrap.attr('transform',e.transform); }));
    var g=gWrap.append('g');

    var map={};
    treeData.forEach(function(n){ map[n.id]=n; });

    // d3 hierarchy
    var root=d3.hierarchy(map[1],function(n){
      var ch=[];
      if(n.leftChild  && map[n.leftChild])  ch.push(map[n.leftChild]);
      if(n.rightChild && map[n.rightChild]) ch.push(map[n.rightChild]);
      return ch.length?ch:null;
    });
    d3.tree().size([W-100, H-120])(root);
    root.descendants().forEach(function(d){ d.x+=50; d.y+=60; });

    // visibility
    var visible=new Set(), hlId=null;
    if(wtStep<0||wtStep>=wtSeq.length){
      treeData.forEach(function(n){ visible.add(n.id); });
    } else {
      for(var i=0;i<=wtStep;i++) visible.add(wtSeq[i]);
      hlId=wtSeq[wtStep];
    }

    // edges
    root.links().forEach(function(lk){
      var sid=lk.source.data.id, tid=lk.target.data.id;
      if(!visible.has(sid)||!visible.has(tid)) return;
      var isHL=(tid===hlId||sid===hlId);
      var isDim=(!isHL && hlId!==null);
      g.append('path')
        .attr('class','edge-line'+(isHL?' hl':'')+(isDim?' dim':''))
        .attr('d', d3.linkVertical().x(function(d){return d.x;}).y(function(d){return d.y;})(lk));
      var mx=(lk.source.x+lk.target.x)/2, my=(lk.source.y+lk.target.y)/2;
      var th=lk.source.data.threshold;
      var lbl=th!=null?(lk.target.data.isLeft?'\\u2264 '+th:'> '+th):(lk.target.data.isLeft?'Yes':'No');
      g.append('text').attr('class','edge-label'+(isDim?' dim':'')).attr('x',mx+(lk.target.data.isLeft?-14:14)).attr('y',my).attr('text-anchor','middle').text(lbl);
    });

    // nodes
    root.descendants().forEach(function(d){
      var n=d.data;
      if(!visible.has(n.id)) return;
      var isHL=(n.id===hlId), isDim=(hlId!==null&&!isHL);
      var cls=n.isLeaf?(n.majority==='Disease'?'node-disease':'node-healthy'):'node-split';
      cls+=(isHL?' hl':'')+(isDim?' dim':'');
      var grp=g.append('g').attr('class',cls)
        .attr('transform','translate('+d.x+','+d.y+')')
        .on('click',function(){ openPanel(n); });
      if(n.isLeaf){
        grp.append('ellipse').attr('rx',54).attr('ry',29);
      } else {
        grp.append('rect').attr('x',-54).attr('y',-29).attr('width',108).attr('height',58).attr('rx',8);
      }
      grp.append('text').attr('class','node-icon-t').attr('y',-12).attr('text-anchor','middle').text(n.icon||(n.isLeaf?(n.majority==='Disease'?'🫀':'💚'):'🔬'));
      var lbl=n.isLeaf?n.majority:(n.varLabel||n.var);
      if(lbl.length>13) lbl=lbl.substr(0,12)+'…';
      grp.append('text').attr('class','node-label').attr('y',4).attr('text-anchor','middle').text(lbl);
      grp.append('text').attr('class','node-sub').attr('y',19).attr('text-anchor','middle').text('n = '+n.n);
    });

    if(hlId&&map[hlId]) openPanel(map[hlId]);
  }

  function openPanel(n){
    var p=document.getElementById('info-panel');
    var icon=n.icon||(n.isLeaf?(n.majority==='Disease'?'🫀':'💚'):'🔬');
    document.getElementById('ptitle').innerHTML=icon+' '+(n.isLeaf?n.majority:(n.varLabel||n.var));
    var bc=n.isLeaf?(n.majority==='Disease'?'bd':'bh'):'bs';
    var bt=n.isLeaf?('Final Prediction: '+n.majority):'Decision Node (depth '+n.depth+')';
    document.getElementById('pbadge').innerHTML='<span class=\"pbadge '+bc+'\">'+bt+'</span>';
    document.getElementById('pbody').innerHTML=n.explanation||'';
    document.getElementById('pn').innerHTML='👥 Patients at this node: <strong>'+n.n+'</strong>';
    p.style.display='block';
  }

  window.addEventListener('resize',function(){ if(treeData) draw(); });
})();
        "))
             )
    ),
    
    # ── Random Forest ─────────────────────────────────────────────────
    tabPanel("Random Forest",
             card(card_header("📊 Model Performance"), tableOutput("rf_metrics_table")),
             helpText("Random Forest builds many decision trees and predicts by majority vote."),
             plotOutput("rf_plot")
    )
  )
)


# SERVER ------------------------------------------------------------------

server <- function(input, output, session) {
  
  # ── Data split ────────────────────────────────────────────────────────
  ds <- reactive({
    set.seed(123)
    sp <- initial_split(h, prop = input$split_ratio)
    list(train = training(sp), test = testing(sp))
  })
  
  # ── Decision Tree ─────────────────────────────────────────────────────
  model <- reactive({
    spec <- decision_tree(tree_depth = input$tree_depth) %>%
      set_engine("rpart") %>% set_mode("classification")
    fit  <- spec %>% fit(target ~ ., data = ds()$train)
    preds   <- fit %>% predict(ds()$test) %>% pull(.pred_class)
    results <- ds()$test %>% mutate(predicted = preds)
    TP <- sum(results$target==1 & results$predicted==1)
    TN <- sum(results$target==0 & results$predicted==0)
    FP <- sum(results$target==0 & results$predicted==1)
    FN <- sum(results$target==1 & results$predicted==0)
    list(
      acc  = (TP+TN)/(TP+TN+FP+FN),
      sens = TP/(TP+FN), spec = TN/(TN+FP), prec = TP/(TP+FP),
      rpart_fit = fit$fit
    )
  })
  
  # Send tree to JS whenever model changes
  observeEvent(list(input$tree_depth, input$split_ratio), {
    nodes <- rpart_to_nodelist(model()$rpart_fit)
    session$sendCustomMessage("treeData", list(nodes = nodes))
    wt_step(0)
  }, ignoreNULL = TRUE, ignoreInit = FALSE)
  
  output$dt_metrics_table <- renderTable({
    data.frame(
      Metric = c("Accuracy","Sensitivity","Specificity","Precision"),
      Value  = round(c(model()$acc, model()$sens, model()$spec, model()$prec), 4)
    )
  })
  
  output$tree_placeholder <- renderUI({ NULL })
  
  # ── Walkthrough state ─────────────────────────────────────────────────
  wt_step  <- reactiveVal(0)
  wt_max   <- reactiveVal(0)
  
  observeEvent(input$wt_max, { wt_max(input$wt_max) })
  
  observeEvent(input$wt_next, {
    s <- min(wt_step() + 1, wt_max())
    wt_step(s)
    session$sendCustomMessage("wtGo", s)
  })
  observeEvent(input$wt_prev, {
    s <- max(wt_step() - 1, 0)
    wt_step(s)
    session$sendCustomMessage("wtGo", s)
  })
  observeEvent(input$wt_reset, {
    wt_step(0)
    session$sendCustomMessage("wtGo", 0)
  })
  observeEvent(input$wt_all, {
    wt_step(-1)
    session$sendCustomMessage("wtGo", -1)
  })
  # Also reset on param change
  observeEvent(list(input$tree_depth, input$split_ratio), {
    wt_step(0)
  })
  
  output$wt_counter_ui <- renderUI({
    total <- wt_max() + 1
    step  <- wt_step()
    if (step < 0) {
      tags$p(class="wt-counter-text", style="font-size:.78rem;color:#64748b;margin-top:4px;text-align:center;",
             "Showing full tree")
    } else {
      tags$p(class="wt-counter-text", style="font-size:.78rem;color:#64748b;margin-top:4px;text-align:center;",
             paste0("Step ", step + 1, " of ", total))
    }
  })
  
  # ── Random Forest ─────────────────────────────────────────────────────
  rf <- reactive({
    spec <- rand_forest(trees=input$rf_trees, mtry=input$rf_mtry, min_n=input$rf_min_n) %>%
      set_engine("ranger", importance="impurity") %>% set_mode("classification")
    fit     <- spec %>% fit(target ~ ., data = ds()$train)
    preds   <- fit %>% predict(ds()$test) %>% pull(.pred_class)
    results <- ds()$test %>% mutate(predicted=preds)
    TP <- sum(results$target==1 & results$predicted==1)
    TN <- sum(results$target==0 & results$predicted==0)
    FP <- sum(results$target==0 & results$predicted==1)
    FN <- sum(results$target==1 & results$predicted==0)
    imp <- data.frame(Variable=names(fit$fit$variable.importance),
                      Importance=fit$fit$variable.importance) %>% arrange(desc(Importance))
    list(acc=(TP+TN)/(TP+TN+FP+FN), sens=TP/(TP+FN),
         spec=TN/(TN+FP), prec=TP/(TP+FP), imp=imp)
  })
  
  output$rf_metrics_table <- renderTable({
    data.frame(Metric=c("Accuracy","Sensitivity","Specificity","Precision"),
               Value=round(c(rf()$acc,rf()$sens,rf()$spec,rf()$prec),4))
  })
  
  output$rf_plot <- renderPlot({
    ggplot(rf()$imp, aes(x=reorder(Variable,Importance), y=Importance)) +
      geom_col(fill="#1a5f7a") +
      coord_flip() +
      labs(title="Random Forest — Variable Importance", x=NULL, y="Mean Decrease in Impurity") +
      theme_minimal(base_size=13) +
      theme(plot.background=element_rect(fill="#f8fafc"), panel.grid.major.y=element_blank())
  })
}

shinyApp(ui = ui, server = server)