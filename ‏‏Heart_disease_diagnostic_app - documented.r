# app.R
# ============================================================================ #
# GCW1 — Dark Medical Theme + D3 Interactive Tree (Tabs 3 & 4)
# ============================================================================ #
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
# Data ====
get_data_path <- function() {
  candidates  <- c("processed.cleveland.data.csv", "processed.cleveland.data")
  app_dir     <- tryCatch(normalizePath(dirname(sys.frame(1)$ofile)), error = function(e) NULL)
  search_dirs <- unique(na.omit(c(app_dir, getwd())))
  for (d in search_dirs) for (f in candidates) {
    p <- file.path(d, f); if (file.exists(p)) return(p)
  }
  stop("Dataset file not found.")
}

load_heart_data <- function(path) {
  heart <- read.csv(path, header = FALSE, na.strings = "?")
  colnames(heart) <- c("age","sex","cp","trestbps","chol","fbs","restecg",
                       "thalach","exang","oldpeak","slope","ca","thal","target")
  heart$target <- ifelse(heart$target > 0, 1, 0)
  for (col in c("age","trestbps","chol","thalach","oldpeak"))
    heart[[col]] <- suppressWarnings(as.numeric(as.character(heart[[col]])))
  for (col in c("sex","cp","fbs","restecg","exang","slope","ca","thal","target"))
    heart[[col]] <- as.factor(heart[[col]])
  heart <- na.omit(heart)
  heart$target <- factor(heart$target, levels = c("0","1"))
  heart
}

heart        <- load_heart_data(get_data_path())
heart.recipe <- heart |> recipe(target ~ .) |> step_dummy(all_nominal_predictors())

set.seed(123)
idx0 <- which(heart$target == "0"); idx1 <- which(heart$target == "1")
train_index <- c(sample(idx0, floor(0.70*length(idx0))),
                 sample(idx1, floor(0.70*length(idx1))))
train <- heart[train_index, ]; test <- heart[-train_index, ]

# ============================================================================ #
# Variable lookup for D3 labels ====
var_info <- list(
  age      = list(label="Age",                    icon="🎂", unit="years",      desc="The patient's age in years."),
  sex      = list(label="Sex",                    icon="👤", unit="(0=F,1=M)",  desc="Patient's biological sex."),
  cp       = list(label="Chest Pain Type",        icon="💢", unit="(0-3)",      desc="0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic."),
  trestbps = list(label="Resting Blood Pressure", icon="🩺", unit="mmHg",      desc="Blood pressure at rest."),
  chol     = list(label="Cholesterol",            icon="🧪", unit="mg/dl",     desc="Serum cholesterol level."),
  fbs      = list(label="Fasting Blood Sugar",    icon="🍬", unit="(>120=1)",  desc="Fasting blood sugar > 120 mg/dl."),
  restecg  = list(label="Resting ECG",            icon="📈", unit="(0-2)",     desc="0=normal, 1=ST-T abnormality, 2=LV hypertrophy."),
  thalach  = list(label="Max Heart Rate",         icon="❤️", unit="bpm",       desc="Max heart rate achieved during exercise."),
  exang    = list(label="Exercise Angina",        icon="🏃", unit="(0/1)",     desc="Exercise-induced chest pain (1=yes)."),
  oldpeak  = list(label="ST Depression",          icon="📉", unit="mm",        desc="ST depression induced by exercise relative to rest."),
  slope    = list(label="ST Slope",               icon="📐", unit="(0-2)",     desc="Slope of peak exercise ST segment."),
  ca       = list(label="Major Vessels",          icon="🔬", unit="(0-3)",     desc="Number of major vessels coloured by fluoroscopy."),
  thal     = list(label="Thalassemia",            icon="🧬", unit="(0-3)",     desc="Blood disorder test result.")
)

# ============================================================================ #
# rpart -> flat node list for D3 ====
rpart_to_nodelist <- function(tree) {
  frame    <- tree$frame
  nodes_r  <- as.integer(rownames(frame))
  splits_m <- tree$splits
  non_leaf <- nodes_r[frame[as.character(nodes_r),"var"] != "<leaf>"]
  node_thresh <- vector("list", length(nodes_r))
  names(node_thresh) <- as.character(nodes_r)
  if (!is.null(splits_m) && length(non_leaf) > 0) {
    n_map <- min(length(non_leaf), nrow(splits_m))
    for (i in seq_len(n_map))
      node_thresh[[as.character(non_leaf[i])]] <- as.numeric(splits_m[i,"index"])
  }
  lapply(nodes_r, function(nd) {
    nd_chr  <- as.character(nd)
    var     <- as.character(frame[nd_chr,"var"])
    is_leaf <- (var == "<leaf>")
    yval    <- as.integer(frame[nd_chr,"yval"])
    majority <- ifelse(yval == 2, "Disease", "No Disease")
    n_obs   <- frame[nd_chr,"n"]
    thresh  <- node_thresh[[nd_chr]]
    depth   <- floor(log2(nd))
    parent  <- if (nd == 1L) NA_integer_ else nd %/% 2L
    lc <- nd*2L; rc <- nd*2L+1L
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
             "<br><br>Split: patients with <em>", label, " &le; ", round(thresh,2),
             "</em> go left &nbsp;|&nbsp; <em>", label, " > ", round(thresh,2), "</em> go right.")
    }
    list(id=nd, parent=if(is.na(parent)) NULL else parent,
         isLeft=if(nd==1L) NULL else as.logical(nd%%2L==0L),
         var=var, varLabel=label, icon=icon, unit=unit,
         threshold=if(is.null(thresh)||is.na(thresh)) NULL else as.numeric(round(thresh,2)),
         isLeaf=is_leaf, majority=majority, n=n_obs, depth=depth,
         leftChild=if(lc %in% nodes_r) lc else NULL,
         rightChild=if(rc %in% nodes_r) rc else NULL,
         explanation=explanation)
  })
}

# ============================================================================ #
# Helper functions ====
get_prob1_safe <- function(pm) {
  cn <- colnames(pm)
  if (!is.null(cn) && "1" %in% cn) return(pm[,"1"])
  if (ncol(pm) >= 2) return(pm[,ncol(pm)])
  as.numeric(pm[,1])
}

compute_metrics <- function(actual, pred_class, pred_prob) {
  actual <- factor(actual, levels=c("0","1"))
  pred_class <- factor(pred_class, levels=c("0","1"))
  cm <- table(Predicted=pred_class, Actual=actual)
  acc  <- sum(diag(cm))/sum(cm)
  sens <- if(sum(cm[,"1"])>0) cm["1","1"]/sum(cm[,"1"]) else NA
  spec <- if(sum(cm[,"0"])>0) cm["0","0"]/sum(cm[,"0"]) else NA
  roc_obj <- tryCatch(
    pROC::roc(actual, as.numeric(pred_prob), quiet=TRUE, levels=c("0","1"), direction="<"),
    error=function(e) NULL)
  auc_val <- if(!is.null(roc_obj)) as.numeric(pROC::auc(roc_obj)) else NA
  list(cm=cm, acc=acc, sens=sens, spec=spec, roc=roc_obj, auc=auc_val)
}

plot_cm_heatmap <- function(cm, title_text="") {
  df <- as.data.frame(cm); colnames(df) <- c("Predicted","Actual","Count")
  ggplot(df, aes(x=Actual, y=Predicted, fill=Count)) +
    geom_tile(color="#57c5e0", linewidth=0.6) +
    geom_text(aes(label=Count), size=5, color="#dff0f8", fontface="bold") +
    scale_fill_gradient(low="#0d2137", high="#57c5e0") +
    labs(title=title_text, x="Actual (0=Healthy, 1=Disease)", y="Predicted") +
    theme_minimal(base_size=13) +
    theme(legend.position="none",
          plot.background=element_rect(fill="#071424",color=NA),
          panel.background=element_rect(fill="#071424",color=NA),
          panel.grid=element_blank(),
          axis.text=element_text(color="#dff0f8"),
          axis.title=element_text(color="#57c5e0"),
          plot.title=element_text(color="#57c5e0",face="bold"))
}

dark_ggplot_theme <- function() {
  theme_minimal(base_size=13) +
    theme(plot.background=element_rect(fill="#071424",color=NA),
          panel.background=element_rect(fill="#0a1e35",color=NA),
          panel.grid.major=element_line(color="#1e3f5c"),
          panel.grid.minor=element_line(color="#0d2137"),
          axis.text=element_text(color="#7ec8de"),
          axis.title=element_text(color="#57c5e0",face="bold"),
          plot.title=element_text(color="#57c5e0",face="bold"),
          legend.background=element_rect(fill="#0a1e35"),
          legend.text=element_text(color="#dff0f8"),
          legend.title=element_text(color="#57c5e0"))
}

# ============================================================================ #
# D3 tree CSS ====
d3_tree_css <- "
  .d3tc {
    width:100%; height:580px;
    background:linear-gradient(160deg,#071424 0%,#0d2137 60%,#071424 100%);
    border-radius:12px; position:relative; overflow:hidden;
  }
  .d3tc::after {
    content:''; position:absolute; inset:0;
    background:radial-gradient(ellipse at 50% 0%,rgba(87,197,224,0.06) 0%,transparent 60%);
    pointer-events:none;
  }
  .edge-line      { stroke:#1e3f5c; stroke-width:2; fill:none; }
  .edge-line.hl   { stroke:#57c5e0; stroke-width:2.5; }
  .edge-line.dim  { opacity:0.15; }
  .edge-lbl       { fill:#57c5e0; font-size:10px; font-family:'Segoe UI',sans-serif; font-weight:600; }
  .edge-lbl.dim   { opacity:0.15; }
  .nd-split rect  { fill:#0e2840; stroke:#57c5e0; stroke-width:2; cursor:pointer; }
  .nd-split.hl rect { fill:#1e3d10; stroke:#fbbf24; stroke-width:3; filter:drop-shadow(0 0 14px rgba(251,191,36,.55)); }
  .nd-split.dim rect { opacity:.2; }
  .nd-healthy ellipse { fill:#0a2e1a; stroke:#4ade80; stroke-width:2; cursor:pointer; }
  .nd-healthy.hl ellipse { stroke:#fbbf24; stroke-width:3; filter:drop-shadow(0 0 14px rgba(251,191,36,.55)); }
  .nd-healthy.dim ellipse { opacity:.2; }
  .nd-disease ellipse { fill:#2e0a0a; stroke:#f87171; stroke-width:2; cursor:pointer; }
  .nd-disease.hl ellipse { stroke:#fbbf24; stroke-width:3; filter:drop-shadow(0 0 14px rgba(251,191,36,.55)); }
  .nd-disease.dim ellipse { opacity:.2; }
  .nd-lbl  { fill:#dff0f8; font-size:11px; font-weight:700; font-family:'Segoe UI',sans-serif; pointer-events:none; }
  .nd-sub  { fill:#7ec8de; font-size:9.5px; font-family:'Segoe UI',sans-serif; pointer-events:none; }
  .nd-icon { font-size:13px; pointer-events:none; }
  .ipanel {
    position:absolute; top:12px; right:12px; width:255px;
    background:rgba(6,16,30,0.96); border:1px solid #57c5e0;
    border-radius:10px; padding:14px 15px; color:#dff0f8;
    font-size:.81rem; line-height:1.55; display:none;
    backdrop-filter:blur(10px); box-shadow:0 8px 32px rgba(0,0,0,.6);
    animation:ipIn .18s ease; z-index:20;
  }
  @keyframes ipIn { from{opacity:0;transform:translateY(-8px)} to{opacity:1;transform:translateY(0)} }
  .ipanel .iptitle { font-size:.93rem; font-weight:700; color:#57c5e0; margin-bottom:5px; display:flex; align-items:center; gap:5px; }
  .ipanel .ipbadge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:.73rem; font-weight:700; margin-bottom:7px; }
  .ibs { background:#0e2840; color:#57c5e0; border:1px solid #57c5e0; }
  .ibh { background:#0a2e1a; color:#4ade80; border:1px solid #4ade80; }
  .ibd { background:#2e0a0a; color:#f87171; border:1px solid #f87171; }
  .ipanel .ipn  { font-size:.73rem; color:#64748b; margin-top:8px; border-top:1px solid #1a3a52; padding-top:6px; }
  .ipanel .ipcl { float:right; cursor:pointer; color:#64748b; font-size:1rem; line-height:1; }
  .ipanel .ipcl:hover { color:#dff0f8; }
  .ecg-deco { position:absolute; bottom:10px; left:0; right:0; height:28px; opacity:.12; pointer-events:none; }
  .wt-ctr { font-size:0.78rem; color:#64748b; margin-top:4px; text-align:center; }
  /* Patient path highlighting */
  .edge-line.path-hl { stroke:#fbbf24; stroke-width:3px; filter:drop-shadow(0 0 6px rgba(251,191,36,0.7)); }
  .edge-lbl.path-hl-lbl { fill:#fbbf24; font-size:11px; }
  .nd-split.path-hl rect    { fill:#2a3d10; stroke:#fbbf24; stroke-width:3px; filter:drop-shadow(0 0 12px rgba(251,191,36,0.6)); }
  .nd-healthy.path-hl ellipse { stroke:#fbbf24; stroke-width:3px; filter:drop-shadow(0 0 14px rgba(251,191,36,0.7)); }
  .nd-disease.path-hl ellipse { stroke:#fbbf24; stroke-width:3px; filter:drop-shadow(0 0 14px rgba(251,191,36,0.7)); }
"

# Reusable ECG SVG
ecg_svg <- tags$svg(class="ecg-deco", viewBox="0 0 1000 28", preserveAspectRatio="none",
                    tags$path(
                      d="M0,14 L80,14 L95,14 L105,1 L115,27 L125,14 L140,14 L220,14 L235,14 L245,1 L255,27 L265,14 L280,14 L360,14 L375,14 L385,1 L395,27 L405,14 L420,14 L500,14 L515,14 L525,1 L535,27 L545,14 L560,14 L640,14 L655,14 L665,1 L675,27 L685,14 L700,14 L780,14 L795,14 L805,1 L815,27 L825,14 L840,14 L920,14 L1000,14",
                      stroke="#57c5e0", `stroke-width`="1.5", fill="none"
                    )
)

# Build a self-contained D3 tree renderer scoped to specific IDs
make_d3_script <- function(cid, pid, msg_data, msg_go, msg_path=NULL) {
  path_handler <- if (!is.null(msg_path)) paste0("
  var pathIds = new Set();
  Shiny.addCustomMessageHandler('",msg_path,"', function(msg){
    pathIds = new Set((msg.path||[]).map(function(x){ return +x; }));
    draw();
  });
") else "  var pathIds = new Set();\n"
  
  HTML(paste0("
(function(){
  var TD=null, WS=0, WQ=[];
  var CID='",cid,"', PID='",pid,"';
",path_handler,"
  Shiny.addCustomMessageHandler('",msg_data,"',function(p){
    TD=p.nodes; WQ=bldSeq(TD); WS=0; pathIds=new Set();
    Shiny.setInputValue('",msg_data,"_max',WQ.length-1,{priority:'event'});
    draw();
  });
  Shiny.addCustomMessageHandler('",msg_go,"',function(s){ WS=s; draw(); });

  function bldSeq(nodes){
    var m={};
    nodes.forEach(function(n){ m[n.id]=n; });
    var out=[];
    (function dfs(id){
      if(!m[id]) return; out.push(id); var n=m[id];
      if(n.leftChild) dfs(n.leftChild);
      if(n.rightChild) dfs(n.rightChild);
    })(1);
    return out;
  }

  function draw(){
    if(!TD) return;
    var c=document.getElementById(CID); if(!c) return;
    var W=c.clientWidth||900, H=555;
    d3.select('#'+CID+' svg.dmt').remove();
    var svg=d3.select('#'+CID).insert('svg',':first-child')
      .attr('class','dmt').attr('width',W).attr('height',H);
    var gW=svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.25,4]).on('zoom',function(e){ gW.attr('transform',e.transform); }));
    var g=gW.append('g');
    var map={};
    TD.forEach(function(n){ map[n.id]=n; });
    var root=d3.hierarchy(map[1],function(n){
      var ch=[];
      if(n.leftChild&&map[n.leftChild]) ch.push(map[n.leftChild]);
      if(n.rightChild&&map[n.rightChild]) ch.push(map[n.rightChild]);
      return ch.length?ch:null;
    });
    d3.tree().size([W-100,H-120])(root);
    root.descendants().forEach(function(d){ d.x+=50; d.y+=60; });

    var vis=new Set(), hlId=null;
    var hasPath=(pathIds&&pathIds.size>0);
    if(WS<0||WS>=WQ.length){
      TD.forEach(function(n){ vis.add(n.id); });
    } else {
      for(var i=0;i<=WS;i++) vis.add(WQ[i]);
      hlId=WQ[WS];
    }

    root.links().forEach(function(lk){
      var sid=lk.source.data.id, tid=lk.target.data.id;
      if(!vis.has(sid)||!vis.has(tid)) return;
      var onPath=hasPath&&pathIds.has(sid)&&pathIds.has(tid);
      var isHL=(!hasPath)&&(tid===hlId||sid===hlId);
      var isDim=hasPath?!onPath:(hlId!==null&&!isHL);
      var cls='edge-line'+(onPath?' path-hl':'')+(isHL?' hl':'')+(isDim?' dim':'');
      g.append('path').attr('class',cls)
        .attr('d',d3.linkVertical().x(function(d){return d.x;}).y(function(d){return d.y;})(lk));
      var mx=(lk.source.x+lk.target.x)/2, my=(lk.source.y+lk.target.y)/2;
      var th=lk.source.data.threshold;
      var lb=th!=null?(lk.target.data.isLeft?'\\u2264 '+th:'> '+th):(lk.target.data.isLeft?'Yes':'No');
      var lblCls='edge-lbl'+(onPath?' path-hl-lbl':'')+(isDim?' dim':'');
      g.append('text').attr('class',lblCls)
        .attr('x',mx+(lk.target.data.isLeft?-14:14)).attr('y',my).attr('text-anchor','middle').text(lb);
    });

    root.descendants().forEach(function(d){
      var n=d.data; if(!vis.has(n.id)) return;
      var onPath=hasPath&&pathIds.has(n.id);
      var isHL=(!hasPath)&&(n.id===hlId);
      var isDim=hasPath?!onPath:(hlId!==null&&!isHL);
      var cls=n.isLeaf?(n.majority==='Disease'?'nd-disease':'nd-healthy'):'nd-split';
      cls+=(onPath?' path-hl':'')+(isHL?' hl':'')+(isDim?' dim':'');
      var grp=g.append('g').attr('class',cls)
        .attr('transform','translate('+d.x+','+d.y+')')
        .on('click',function(){ openP(n); });
      if(n.isLeaf){ grp.append('ellipse').attr('rx',54).attr('ry',29); }
      else { grp.append('rect').attr('x',-54).attr('y',-29).attr('width',108).attr('height',58).attr('rx',8); }
      grp.append('text').attr('class','nd-icon').attr('y',-12).attr('text-anchor','middle')
        .text(n.icon||(n.isLeaf?(n.majority==='Disease'?'🫀':'💚'):'🔬'));
      var lb=n.isLeaf?n.majority:(n.varLabel||n.var);
      if(lb.length>13) lb=lb.substr(0,12)+'\\u2026';
      grp.append('text').attr('class','nd-lbl').attr('y',4).attr('text-anchor','middle').text(lb);
      grp.append('text').attr('class','nd-sub').attr('y',19).attr('text-anchor','middle').text('n = '+n.n);
    });

    // Panel only opens on user click — NOT auto-opened here
  }

  function openP(n){
    var p=document.getElementById(PID); if(!p) return;
    var icon=n.icon||(n.isLeaf?(n.majority==='Disease'?'🫀':'💚'):'🔬');
    p.querySelector('.iptitle').innerHTML=icon+' '+(n.isLeaf?n.majority:(n.varLabel||n.var));
    var bc=n.isLeaf?(n.majority==='Disease'?'ibd':'ibh'):'ibs';
    var bt=n.isLeaf?('Final Prediction: '+n.majority):'Decision Node (depth '+n.depth+')';
    p.querySelector('.ipbadge').innerHTML='<span class=\"ipbadge '+bc+'\">'+bt+'</span>';
    p.querySelector('.ipbody').innerHTML=n.explanation||'';
    p.querySelector('.ipn').innerHTML='\\uD83D\\uDC65 Patients at this node: <strong>'+n.n+'</strong>';
    p.style.display='block';
  }

  window.addEventListener('resize',function(){ if(TD) draw(); });
})();
"))
}

make_info_panel <- function(pid) {
  tags$div(class="ipanel", id=pid,
           tags$span(class="ipcl", onclick=paste0("document.getElementById('",pid,"').style.display='none'"), "✕"),
           tags$div(class="iptitle"),
           tags$div(class="ipbadge"),
           tags$div(class="ipbody"),
           tags$div(class="ipn")
  )
}

make_wt_controls <- function(pfx) {
  tagList(
    tags$p("🚶", tags$b(" Guided Walkthrough"), style="color:#57c5e0;font-size:0.88rem;margin-bottom:6px;"),
    tags$div(style="display:flex;gap:6px;margin-bottom:6px;",
             actionButton(paste0(pfx,"_prev"), "◀ Prev", style="flex:1;font-size:0.8rem;padding:5px 2px;"),
             actionButton(paste0(pfx,"_next"), "Next ▶", style="flex:1;font-size:0.8rem;padding:5px 2px;")),
    tags$div(style="display:flex;gap:6px;margin-bottom:4px;",
             actionButton(paste0(pfx,"_reset"), "↺ Reset", style="flex:1;font-size:0.8rem;padding:5px 2px;"),
             actionButton(paste0(pfx,"_all"),   "⬛ All",   style="flex:1;font-size:0.8rem;padding:5px 2px;")),
    uiOutput(paste0(pfx,"_counter"))
  )
}

tree_legend <- tags$div(
  style="font-size:0.82rem;display:flex;flex-direction:column;gap:5px;",
  tags$h6("Legend", style="font-weight:700;margin-bottom:6px;color:#dff0f8;"),
  tags$div(style="display:flex;align-items:center;gap:7px;",
           tags$div(style="width:13px;height:13px;background:#0e2840;border:2px solid #57c5e0;border-radius:3px;"),"Split node"),
  tags$div(style="display:flex;align-items:center;gap:7px;",
           tags$div(style="width:13px;height:13px;background:#0a2e1a;border:2px solid #4ade80;border-radius:50%;"),"Leaf: No Disease"),
  tags$div(style="display:flex;align-items:center;gap:7px;",
           tags$div(style="width:13px;height:13px;background:#2e0a0a;border:2px solid #f87171;border-radius:50%;"),"Leaf: Disease"),
  tags$div(style="display:flex;align-items:center;gap:7px;",
           tags$div(style="width:13px;height:13px;background:#1e3d10;border:2px solid #fbbf24;border-radius:3px;"),"Current node"),
  tags$small(style="color:#64748b;margin-top:4px;display:block;",
             tags$b("Click")," node for info. ",tags$b("Scroll")," to zoom.")
)

# ============================================================================ #
# App CSS ====
dark_medical_css <- "
  body,.bslib-page-sidebar{background:#071424!important;color:#dff0f8!important;font-family:'Segoe UI',system-ui,sans-serif!important;}
  .navbar{background:linear-gradient(90deg,#040d1a 0%,#0a1e35 100%)!important;border-bottom:1px solid #1e3f5c!important;box-shadow:0 2px 20px rgba(87,197,224,0.15)!important;}
  .navbar .navbar-brand,.navbar-nav .nav-link{color:#dff0f8!important;font-weight:600!important;}
  .navbar .nav-link.active,.navbar .nav-link:hover{color:#57c5e0!important;}
  .navbar .nav-link.active{border-bottom:2px solid #57c5e0!important;}
  .sidebar,.bslib-sidebar-layout>.sidebar{background:#040d1a!important;border-right:1px solid #1e3f5c!important;}
  .sidebar .form-label,.sidebar label,.sidebar h5,.sidebar h6,.sidebar .help-block,.sidebar small,.sidebar p{color:#7ec8de!important;}
  .card{background:linear-gradient(135deg,#0a1e35 0%,#0d2641 100%)!important;border:1px solid #1e3f5c!important;border-radius:12px!important;box-shadow:0 4px 20px rgba(0,0,0,0.4)!important;color:#dff0f8!important;}
  .card-header{background:rgba(87,197,224,0.08)!important;border-bottom:1px solid #1e3f5c!important;color:#57c5e0!important;font-weight:700!important;}
  .card h3,.card h4,.card h5,.card p,.card li{color:#dff0f8!important;}
  .card b,.card strong{color:#57c5e0!important;}
  .bslib-value-box{border-radius:12px!important;border:1px solid #1e3f5c!important;}
  .form-control,.form-select{background:#071424!important;border:1px solid #1e3f5c!important;color:#dff0f8!important;border-radius:8px!important;}
  .form-control:focus,.form-select:focus{border-color:#57c5e0!important;box-shadow:0 0 0 3px rgba(87,197,224,0.2)!important;}
  .irs--shiny .irs-bar{background:#57c5e0!important;border-top:1px solid #57c5e0!important;border-bottom:1px solid #57c5e0!important;}
  .irs--shiny .irs-handle{background:#57c5e0!important;border:2px solid #dff0f8!important;}
  .irs--shiny .irs-from,.irs--shiny .irs-to,.irs--shiny .irs-single{background:#57c5e0!important;}
  .irs--shiny .irs-line{background:#1e3f5c!important;}
  .irs--shiny .irs-grid-text,.irs--shiny .irs-min,.irs--shiny .irs-max{color:#7ec8de!important;}
  .btn-primary{background:linear-gradient(135deg,#1a5f7a 0%,#0d3d52 100%)!important;border:1px solid #57c5e0!important;color:#dff0f8!important;border-radius:8px!important;font-weight:600!important;}
  .btn-primary:hover{background:linear-gradient(135deg,#57c5e0 0%,#1a5f7a 100%)!important;}
  .btn-secondary{background:#0a1e35!important;border:1px solid #1e3f5c!important;color:#7ec8de!important;border-radius:8px!important;font-weight:600!important;}
  .btn-secondary:hover{border-color:#57c5e0!important;color:#57c5e0!important;}
  .accordion-item{background:#0a1e35!important;border:1px solid #1e3f5c!important;border-radius:8px!important;margin-bottom:6px!important;}
  .accordion-button{background:#0d2641!important;color:#57c5e0!important;font-weight:600!important;border-radius:8px!important;}
  .accordion-button:not(.collapsed){background:rgba(87,197,224,0.12)!important;color:#57c5e0!important;box-shadow:none!important;}
  .accordion-button::after{filter:invert(1) sepia(1) saturate(2) hue-rotate(170deg)!important;}
  .accordion-body{color:#dff0f8!important;background:#071424!important;}
  .form-check-input:checked{background-color:#57c5e0!important;border-color:#57c5e0!important;}
  body{padding-bottom:10px;}
  .tab-pane{padding-bottom:20px!important;}
  hr{border-color:#1e3f5c!important;}
  ::-webkit-scrollbar{width:6px;height:6px;}
  ::-webkit-scrollbar-track{background:#040d1a;}
  ::-webkit-scrollbar-thumb{background:#1e3f5c;border-radius:3px;}
  ::-webkit-scrollbar-thumb:hover{background:#57c5e0;}
  .bslib-sidebar-layout>.main{background:#071424!important;}
  .nav-underline .nav-link{color:#7ec8de!important;}
  .nav-underline .nav-link.active{color:#57c5e0!important;border-bottom-color:#57c5e0!important;}
  .step-info-box{background:rgba(87,197,224,0.06);border:1px solid #1e3f5c;border-radius:8px;padding:10px 14px;font-size:0.9em;color:#7ec8de;}
  .pred-badge-disease{background:#2e0a0a;color:#f87171;border:1px solid #f87171;padding:8px 16px;border-radius:999px;font-weight:700;font-size:1.05rem;display:inline-block;margin-top:10px;box-shadow:0 0 12px rgba(248,113,113,0.3);}
  .pred-badge-healthy{background:#0a2e1a;color:#4ade80;border:1px solid #4ade80;padding:8px 16px;border-radius:999px;font-weight:700;font-size:1.05rem;display:inline-block;margin-top:10px;box-shadow:0 0 12px rgba(74,222,128,0.3);}
"

# ============================================================================ #
# UI ====
theme_gc <- bs_theme(version=5, bootswatch="darkly", primary="#57c5e0",
                     bg="#071424", fg="#dff0f8")

ui <- page_navbar(
  id="main_nav",
  title="Tree-Based Methods: Predicting Heart Disease",
  theme=theme_gc,
  
  header=tagList(
    tags$head(
      tags$style(HTML(dark_medical_css)),
      tags$style(HTML(d3_tree_css)),
      tags$script(HTML("document.addEventListener('DOMContentLoaded',function(){
        [].slice.call(document.querySelectorAll('[data-bs-toggle=\"tooltip\"]'))
          .map(function(t){return new bootstrap.Tooltip(t);});
      });"))
    ),
    tags$script(src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js")
  ),
  
  # ── Tab 1 ────────────────────────────────────────────────────────────────
  nav_panel("1. The Problem", value="tab1",
            layout_column_wrap(width=1/2,
                               card(
                                 tags$div(style="display:flex;align-items:center;gap:10px;margin-bottom:16px;",
                                          tags$span(style="font-size:2rem;"),
                                          h3("Predicting Heart Disease: A Case Study",
                                             style="margin:0;color:#57c5e0;font-weight:700;")),
                                 p("In clinical settings, identifying patients at high risk of heart disease quickly and accurately is critical."),
                                 p("This presentation demonstrates how ", tags$b("Tree-Based Machine Learning"),
                                   " can be used to solve this problem using historical health data from Cleveland."),
                                 tags$hr(),
                                 h5("What are Tree-Based Methods?",style="color:#4ade80;"),
                                 p("Tree-based methods ask a sequence of simple yes/no questions about the data until a final classification is reached — like an upside-down tree or flowchart."),
                                 tags$hr(),
                                 h5("Why Tree-Based Methods?",style="color:#4ade80;"),
                                 tags$ul(
                                   tags$li("Highly interpretable — you can follow the logic step-by-step."),
                                   tags$li("They mimic human decision-making processes."),
                                   tags$li("Advanced versions (Random Forests, XGBoost) offer state-of-the-art accuracy.")
                                 )
                               ),
                               card(
                                 h5("How a Decision Tree Works",style="color:#57c5e0;font-weight:700;"),
                                 p("A decision tree classifies data by applying a sequence of rules, working step-by-step down to a final 'leaf' prediction."),
                                 tags$img(src="decision_tree_transparent.png", alt="Decision tree illustration",
                                          style="display:block;margin:0 auto;max-width:100%;max-height:380px;height:auto;margin-bottom:12px;"),
                                 tags$p(style="font-size:0.9em;color:#7ec8de;text-align:center;",
                                        tags$i("Built on training data — evaluated on separate testing data.")),
                                 p("We split our data into non-overlapping training and testing partitions. The tree learns from training data, and we measure accuracy on unseen testing data.")
                               )
            )
  ),
  
  # ── Tab 2 ────────────────────────────────────────────────────────────────
  nav_panel("2. The Data", value="tab2",
            layout_column_wrap(width=1/3,
                               value_box("Observations", nrow(heart), theme=value_box_theme(bg="#0a1e35",fg="#57c5e0"),
                                         p("total patients",style="color:#7ec8de;")),
                               value_box("Predictor Variables", ncol(heart)-1, theme=value_box_theme(bg="#0a1e35",fg="#4ade80"),
                                         p("clinical measurements",style="color:#7ec8de;")),
                               value_box("Prevalence (Disease)",
                                         paste0(round(mean(as.numeric(as.character(heart$target)))*100,1),"%"),
                                         theme=value_box_theme(bg="#2e0a0a",fg="#f87171"),
                                         p("patients with heart disease",style="color:#f87171;"))
            ),
            layout_column_wrap(width=1/2,
                               card(
                                 h4("From simple examples to real-world complexity",style="color:#57c5e0;"),
                                 p("We predict a ",tags$b("target variable")," (heart disease yes/no) using 13 different ",tags$b("predictor variables"),"."),
                                 h5("The value of tree-based methods here",style="color:#4ade80;"),
                                 tags$ul(
                                   tags$li(tags$b("Mixed data types: "),"Handles numeric and categorical variables naturally."),
                                   tags$li(tags$b("Built-in feature selection: "),"Irrelevant variables are simply not used."),
                                   tags$li(tags$b("Complex interactions: "),"Captures rules like 'high cholesterol is a risk indicator only if the patient is also over 60.'")
                                 )
                               ),
                               card(
                                 h4("The 13 predictors explained",style="color:#57c5e0;"),
                                 accordion(open=FALSE,
                                           accordion_panel("Demographics & Symptoms",
                                                           tags$ul(tags$li(tags$b("Age:")," years (Numeric)"),
                                                                   tags$li(tags$b("Sex:")," 0=Female, 1=Male"),
                                                                   tags$li(tags$b("cp:")," Chest pain type 0-3"))),
                                           accordion_panel("Clinical Vitals",
                                                           tags$ul(tags$li(tags$b("trestbps:")," Resting blood pressure mmHg"),
                                                                   tags$li(tags$b("chol:")," Serum cholesterol mg/dl"),
                                                                   tags$li(tags$b("fbs:")," Fasting blood sugar >120"))),
                                           accordion_panel("Test Results (ECG & Exercise)",
                                                           tags$ul(tags$li(tags$b("restecg:")," Resting ECG 0-2"),
                                                                   tags$li(tags$b("thalach:")," Max heart rate bpm"),
                                                                   tags$li(tags$b("exang:")," Exercise-induced angina"),
                                                                   tags$li(tags$b("oldpeak:")," ST depression mm"),
                                                                   tags$li(tags$b("slope:")," ST slope 0-2"),
                                                                   tags$li(tags$b("ca:")," Major vessels 0-3"),
                                                                   tags$li(tags$b("thal:")," Thalassemia 0-3")))
                                 )
                               )
            )
  ),
  
  # ── Tab 3 ────────────────────────────────────────────────────────────────
  nav_panel("3. Building a Tree", value="tab3",
            layout_sidebar(
              sidebar=sidebar(
                tags$p("🔬", tags$b(" Growing a tree"), style="color:#57c5e0;font-size:0.95rem;"),
                tags$div(class="step-info-box",
                         p("Adjust the slider to control how deep the tree grows. Click any node in the tree to learn what rule it applies.")
                ),
                tags$br(),
                
                # Pruning mode toggle
                radioButtons("s3_prune_mode", "Tree mode:",
                             choices = c("Auto-pruned (best fit)" = "auto", "Manual depth" = "manual"),
                             selected = "auto", inline = TRUE
                ),
                uiOutput("s3_mode_desc"),
                
                # Depth slider — only shown in manual mode
                conditionalPanel("input.s3_prune_mode == 'manual'",
                                 sliderInput("step_depth", "Tree Depth:", min=1, max=5, value=1, step=1),
                                 
                                 sliderInput("s3_cp", "Complexity (cp):", min=0.001, max=0.1, value=0.01, step=0.001),
                                 
                                 sliderInput("s3_split", "Train/Test Split:", min=0.5, max=0.9, value=0.7, step=0.05),
                                 
                                 # Live split display
                                 uiOutput("s3_split_display")
                )
              ),
              
              uiOutput("step_metrics_ui"),
              
              layout_column_wrap(width=1/2,
                                 card(
                                   card_header("🌳 Interactive Decision Tree — click any node"),
                                   tags$div(
                                     style="background:rgba(251,191,36,0.07);border-left:3px solid #fbbf24;padding:7px 12px;border-radius:0 6px 6px 0;margin:8px 4px 4px;",
                                     tags$p(style="font-size:0.75rem;color:#fbbf24;margin:0;",
                                            "📚 ", tags$b("Teaching simplification: "),
                                            "this tree uses only ", tags$b("Age"), " and ", tags$b("Max Heart Rate"),
                                            " so the decision boundaries can be visualised in the 2D plot alongside. The real model (Tab 4) uses all 13 clinical variables."
                                     )
                                   ),
                                   tags$div(class="d3tc", id="s3-tc",
                                            make_info_panel("s3-ip"),
                                            ecg_svg
                                   ),
                                   tags$script(make_d3_script("s3-tc","s3-ip","s3td","s3go"))
                                 ),
                                 card(
                                   card_header("📊 2D Problem Space (Age vs Heart Rate)"),
                                   plotOutput("plot_step_2d", height=490),
                                   tags$p(style="font-size:0.9em;color:#7ec8de;text-align:center;margin-top:6px;",
                                          tags$i("2D slice only — the full model uses all 13 clinical variables."))
                                 )
              )
            )
  ),
  
  # ── Tab 4 ────────────────────────────────────────────────────────────────
  nav_panel("4. Full Decision Tree", value="tab4",
            layout_sidebar(
              sidebar=sidebar(
                tags$p("🧬", tags$b(" Patient Profile"), style="color:#57c5e0;font-size:0.95rem;"),
                tags$div(class="step-info-box",
                         p("Enter a patient's clinical values below, then click Predict to see which path through the tree they follow.")
                ),
                tags$br(),
                # Demographics
                tags$p(style="font-size:0.75rem;color:#57c5e0;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;","Demographics"),
                sliderInput("p_age","Age (years):", min=20, max=80, value=55, step=1),
                radioButtons("p_sex","Sex:", choices=c("Female"="0","Male"="1"), selected="1", inline=TRUE),
                # Symptoms
                tags$p(style="font-size:0.75rem;color:#57c5e0;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;","Symptoms"),
                selectInput("p_cp","Chest Pain Type:",
                            choices=c("Typical Angina"="0","Atypical Angina"="1","Non-anginal"="2","Asymptomatic"="3"), selected="3"),
                radioButtons("p_exang","Exercise Angina:", choices=c("No"="0","Yes"="1"), selected="0", inline=TRUE),
                # Vitals
                tags$p(style="font-size:0.75rem;color:#57c5e0;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;","Clinical Vitals"),
                sliderInput("p_trestbps","Resting BP (mmHg):", min=90, max=200, value=130, step=1),
                sliderInput("p_chol","Cholesterol (mg/dl):", min=100, max=600, value=240, step=1),
                radioButtons("p_fbs","Fasting Blood Sugar >120:", choices=c("No"="0","Yes"="1"), selected="0", inline=TRUE),
                sliderInput("p_thalach","Max Heart Rate (bpm):", min=60, max=220, value=150, step=1),
                # Test results
                tags$p(style="font-size:0.75rem;color:#57c5e0;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;","Test Results"),
                selectInput("p_restecg","Resting ECG:",
                            choices=c("Normal"="0","ST-T Abnormality"="1","LV Hypertrophy"="2"), selected="0"),
                numericInput("p_oldpeak","ST Depression:", value=1.0, min=0, max=6.2, step=0.1),
                selectInput("p_slope","ST Slope:",
                            choices=c("Upsloping"="0","Flat"="1","Downsloping"="2"), selected="1"),
                selectInput("p_ca","Major Vessels (0-3):", choices=c("0"="0","1"="1","2"="2","3"="3"), selected="0"),
                selectInput("p_thal","Thalassemia:",
                            choices=c("Normal"="1","Fixed Defect"="2","Reversible Defect"="3"), selected="1"),
                actionButton("btn_predict","🔍 Predict Patient Outcome", class="btn-primary",
                             style="width:100%;margin-top:4px;"),
                actionButton("btn_reset_patient","↺ Reset / New Patient", class="btn-secondary",
                             style="width:100%;margin-top:6px;"),
                uiOutput("predicted_condition_ui"),
                tags$hr(),
                tags$p("⚙️", tags$b(" Model Settings"), style="color:#57c5e0;font-size:0.95rem;"),
                radioButtons("s4_prune_mode", "Tree mode:",
                             choices = c("Auto-pruned (best fit)" = "auto", "Manual depth" = "manual"),
                             selected = "auto", inline = TRUE
                ),
                uiOutput("s4_mode_desc"),
                conditionalPanel("input.s4_prune_mode == 'manual'",
                                 sliderInput("s4_depth", "Tree Depth:", min=1, max=10, value=5, step=1),
                                 sliderInput("s4_cp",    "Complexity (cp):", min=0.001, max=0.1, value=0.015, step=0.001),
                                 sliderInput("s4_split", "Train/Test Split:", min=0.5, max=0.9, value=0.7, step=0.05),
                                 uiOutput("s4_split_display")
                )
              ),
              
              uiOutput("tab4_metrics_ui"),
              
              layout_column_wrap(width=1,
                                 card(
                                   card_header("🏥 Full Clinical Decision Tree — click any node to learn more"),
                                   tags$div(class="d3tc", id="s4-tc",
                                            make_info_panel("s4-ip"),
                                            ecg_svg
                                   ),
                                   tags$script(make_d3_script("s4-tc","s4-ip","s4td","s4go","s4path")),
                                   tags$p(style="font-size:0.9em;color:#7ec8de;text-align:center;margin-top:10px;",
                                          tags$i("Built on Training Data — accuracy scores from unseen Testing Data."))
                                 )
              )
            )
  ),
  
  # ── Tab 5 ────────────────────────────────────────────────────────────────
  nav_panel("5. Ensembles (RF & XGBoost)", value="tab5",
            navset_card_underline(
              nav_panel("🌲 Random Forest",
                        layout_column_wrap(width=1/2,
                                           plotOutput("plot_rf_imp"),
                                           plotOutput("plot_rf_cm_heat")
                        )
              ),
              nav_panel("⚡ XGBoost",
                        card(card_header("⚡ XGBoost — Boosted Tree Visualisation"),
                             p("Boosted trees build sequentially, each learning from the previous tree's mistakes."),
                             grVizOutput("plot_xgb_tree"))
              )
            )
  ),
  
  # ── Tab 6 ────────────────────────────────────────────────────────────────
  nav_panel("6. Takeaways", value="tab6",
            layout_column_wrap(width=1/3,
                               card(
                                 tags$div(style="text-align:center;padding:8px 0 12px;",
                                          tags$span("🌳",style="font-size:2.5rem;"),
                                          h4("Decision Trees",style="color:#57c5e0;font-weight:700;margin-top:8px;")),
                                 tags$div(style="background:rgba(87,197,224,0.06);border:1px solid #1e3f5c;border-radius:8px;padding:14px;",
                                          tags$p(tags$b("Strengths:",style="color:#4ade80;")),
                                          tags$ul(tags$li("Excellent for stakeholder communication"),
                                                  tags$li("Fully transparent — follow each path"),
                                                  tags$li("No data transformation needed")),
                                          tags$p(tags$b("Limitations:",style="color:#f87171;")),
                                          tags$ul(tags$li("Prone to overfitting"),tags$li("Lower accuracy than ensembles")))
                               ),
                               card(
                                 tags$div(style="text-align:center;padding:8px 0 12px;",
                                          tags$span("🌲",style="font-size:2.5rem;"),
                                          h4("Random Forests",style="color:#4ade80;font-weight:700;margin-top:8px;")),
                                 tags$div(style="background:rgba(74,222,128,0.06);border:1px solid #1e3f5c;border-radius:8px;padding:14px;",
                                          tags$p(tags$b("Strengths:",style="color:#4ade80;")),
                                          tags$ul(tags$li("Greatly improved accuracy"),
                                                  tags$li("Averaging trees reduces overfitting"),
                                                  tags$li("Built-in feature importance")),
                                          tags$p(tags$b("Limitations:",style="color:#f87171;")),
                                          tags$ul(tags$li("Cannot draw a single path"),tags$li("Less interpretable")))
                               ),
                               card(
                                 tags$div(style="text-align:center;padding:8px 0 12px;",
                                          tags$span("⚡",style="font-size:2.5rem;"),
                                          h4("XGBoost",style="color:#fbbf24;font-weight:700;margin-top:8px;")),
                                 tags$div(style="background:rgba(251,191,36,0.06);border:1px solid #1e3f5c;border-radius:8px;padding:14px;",
                                          tags$p(tags$b("Strengths:",style="color:#4ade80;")),
                                          tags$ul(tags$li("Often best-in-class performance"),
                                                  tags$li("Learns from previous mistakes"),
                                                  tags$li("Handles missing data well")),
                                          tags$p(tags$b("Limitations:",style="color:#f87171;")),
                                          tags$ul(tags$li("Requires careful tuning"),tags$li("Can overfit without regularisation")))
                               )
            )
  ),
  
)


# ============================================================================ #
# Server ====
server <- function(input, output, session) {
  
  # ── Tab 3: Step tree ─────────────────────────────────────────────────────
  
  output$s3_mode_desc <- renderUI({
    is_auto <- !is.null(input$s3_prune_mode) && input$s3_prune_mode == "auto"
    if (is_auto) {
      tags$div(
        style="background:rgba(74,222,128,0.06);border-left:3px solid #4ade80;padding:7px 12px;border-radius:0 6px 6px 0;margin-bottom:10px;",
        tags$p(style="font-size:0.75rem;color:#7ec8de;margin:0;",
               tags$b(style="color:#4ade80;","✓ Auto-pruned: "),
               "rpart automatically finds the optimal depth using cross-validation. It stops growing branches that don't improve accuracy enough to justify the added complexity."
        )
      )
    } else {
      tags$div(
        style="background:rgba(87,197,224,0.06);border-left:3px solid #57c5e0;padding:7px 12px;border-radius:0 6px 6px 0;margin-bottom:10px;",
        tags$p(style="font-size:0.75rem;color:#7ec8de;margin:0;",
               tags$b(style="color:#57c5e0;","✎ Manual: "),
               "you directly control depth, complexity, and data split. Use this to explore how each parameter affects the tree shape and accuracy."
        )
      )
    }
  })
  
  # Reactive train/test split for Tab 3
  s3_data <- reactive({
    is_manual <- !is.null(input$s3_prune_mode) && input$s3_prune_mode == "manual"
    split_ratio <- if (is_manual && !is.null(input$s3_split)) input$s3_split else 0.7
    set.seed(123)
    n      <- nrow(heart)
    idx0   <- which(heart$target == "0"); idx1 <- which(heart$target == "1")
    tr0    <- sample(idx0, floor(split_ratio * length(idx0)))
    tr1    <- sample(idx1, floor(split_ratio * length(idx1)))
    tr_idx <- c(tr0, tr1)
    list(train=heart[tr_idx,], test=heart[-tr_idx,],
         n_train=length(tr_idx), n_test=n - length(tr_idx),
         pct_train=round(split_ratio*100), pct_test=round((1-split_ratio)*100))
  })
  
  # Live split display
  output$s3_split_display <- renderUI({
    d <- s3_data()
    tags$div(
      style="display:flex;gap:4px;margin-top:-8px;margin-bottom:4px;border-radius:6px;overflow:hidden;font-size:0.75rem;font-weight:600;",
      tags$div(style=paste0("flex:",d$pct_train,";background:#1a5f7a;color:#dff0f8;padding:5px 8px;text-align:center;"),
               paste0("Train ", d$pct_train,"% (n=",d$n_train,")")),
      tags$div(style=paste0("flex:",d$pct_test,";background:#2e0a0a;color:#f87171;padding:5px 8px;text-align:center;"),
               paste0("Test ", d$pct_test,"% (n=",d$n_test,")"))
    )
  })
  
  step_model <- reactive({
    if (!is.null(input$s3_prune_mode) && input$s3_prune_mode == "auto") {
      rpart(target ~ age + thalach, data=s3_data()$train, method="class")
    } else {
      req(input$step_depth)
      cp_val <- if(!is.null(input$s3_cp)) input$s3_cp else 0.01
      rpart(target ~ age + thalach, data=s3_data()$train, method="class",
            control=rpart.control(maxdepth=input$step_depth, cp=cp_val, minsplit=2))
    }
  })
  
  observeEvent(step_model(), {
    session$sendCustomMessage("s3td", list(nodes=rpart_to_nodelist(step_model())))
    session$sendCustomMessage("s3go", -1)  # always show full tree
  }, ignoreNULL=TRUE)
  
  output$step_metrics_ui <- renderUI({
    req(step_model())
    tr    <- s3_data()$train
    preds <- predict(step_model(), tr, type="class"); actual <- tr$target
    acc   <- sum(preds==actual)/nrow(tr)
    hi    <- which(preds=="0"); di <- which(preds=="1")
    ch    <- if(length(hi)>0) sum(actual[hi]=="0")/length(hi) else 0
    cd    <- if(length(di)>0) sum(actual[di]=="1")/length(di) else 0
    
    metric_label <- function(txt, tip) {
      tags$div(style="font-size:0.72rem;color:#7ec8de;text-transform:uppercase;letter-spacing:0.5px;display:flex;align-items:center;gap:4px;",
               txt,
               tooltip(trigger=bsicons::bs_icon("info-circle", size="0.85em", style="color:#57c5e0;cursor:pointer;"), tip)
      )
    }
    
    tags$div(
      style="display:flex;gap:0;background:linear-gradient(135deg,#0a1e35,#0d2641);border:1px solid #1e3f5c;border-radius:10px;overflow:hidden;margin-bottom:12px;",
      tags$div(style="flex:1;padding:10px 16px;border-right:1px solid #1e3f5c;",
               metric_label("Accuracy", "The percentage of all patients the tree correctly classified as either healthy or diseased. Higher is better, but always check it alongside sensitivity and specificity."),
               tags$div(style="font-size:1.5rem;font-weight:700;color:#4ade80;", paste0(round(acc*100,1),"%")),
               tags$div(style="font-size:0.72rem;color:#64748b;","on training data")),
      tags$div(style="flex:1;padding:10px 16px;border-right:1px solid #1e3f5c;",
               metric_label("Predicted Healthy (NPV)", "Negative Predictive Value: of all patients the tree predicted were healthy, what % actually were healthy. A low NPV means the tree is incorrectly clearing sick patients."),
               tags$div(style="font-size:1.5rem;font-weight:700;color:#57c5e0;", paste0(round(length(hi)/nrow(train)*100,1),"%")),
               tags$div(style="font-size:0.72rem;color:#64748b;", paste0(round(ch*100,1),"% actually healthy"))),
      tags$div(style="flex:1;padding:10px 16px;",
               metric_label("Predicted Disease (PPV)", "Positive Predictive Value: of all patients the tree flagged as having disease, what % actually did. A low PPV means the tree is raising too many false alarms."),
               tags$div(style="font-size:1.5rem;font-weight:700;color:#f87171;", paste0(round(length(di)/nrow(train)*100,1),"%")),
               tags$div(style="font-size:0.72rem;color:#64748b;", paste0(round(cd*100,1),"% actually sick")))
    )
  })
  
  # 2D scatter — explicit height avoids margins error
  output$plot_step_2d <- renderPlot({
    m  <- step_model()
    tr <- s3_data()$train
    gd <- expand.grid(
      age     = seq(min(tr$age,    na.rm=TRUE), max(tr$age,    na.rm=TRUE), length.out=100),
      thalach = seq(min(tr$thalach,na.rm=TRUE), max(tr$thalach,na.rm=TRUE), length.out=100)
    )
    gd$pred_class <- predict(m, newdata=gd, type="class")
    ggplot() +
      geom_raster(data=gd, aes(x=age,y=thalach,fill=pred_class), alpha=0.35) +
      geom_point(data=tr, aes(x=age,y=thalach,color=target), size=2.5, shape=16) +
      scale_color_manual(values=c("0"="#4ade80","1"="#f87171"),
                         name="Actual Status", labels=c("Healthy","Disease")) +
      scale_fill_manual(values=c("0"="#4ade80","1"="#f87171"), guide="none") +
      dark_ggplot_theme() +
      labs(x="Age (Years)", y="Max Heart Rate (thalach)") +
      theme(legend.position="bottom")
  }, height=460)
  
  outputOptions(output, "plot_step_2d", suspendWhenHidden=FALSE)
  
  # ── Tab 4: Full tree ──────────────────────────────────────────────────────
  
  output$s4_mode_desc <- renderUI({
    is_auto <- is.null(input$s4_prune_mode) || input$s4_prune_mode == "auto"
    if (is_auto) {
      tags$div(
        style="background:rgba(74,222,128,0.06);border-left:3px solid #4ade80;padding:7px 12px;border-radius:0 6px 6px 0;margin-bottom:10px;",
        tags$p(style="font-size:0.75rem;color:#7ec8de;margin:0;",
               tags$b(style="color:#4ade80;","✓ Auto-pruned: "),
               "rpart finds the optimal tree using cross-validation — stops growing branches that don't improve accuracy enough to justify the complexity."
        )
      )
    } else {
      tags$div(
        style="background:rgba(87,197,224,0.06);border-left:3px solid #57c5e0;padding:7px 12px;border-radius:0 6px 6px 0;margin-bottom:10px;",
        tags$p(style="font-size:0.75rem;color:#7ec8de;margin:0;",
               tags$b(style="color:#57c5e0;","✎ Manual: "),
               "control depth, complexity, and data split to explore how each parameter affects real-world performance."
        )
      )
    }
  })
  
  s4_data <- reactive({
    is_manual <- !is.null(input$s4_prune_mode) && input$s4_prune_mode == "manual"
    split_ratio <- if (is_manual && !is.null(input$s4_split)) input$s4_split else 0.7
    set.seed(123)
    idx0   <- which(heart$target == "0"); idx1 <- which(heart$target == "1")
    tr_idx <- c(sample(idx0, floor(split_ratio*length(idx0))),
                sample(idx1, floor(split_ratio*length(idx1))))
    list(train=heart[tr_idx,], test=heart[-tr_idx,],
         n_train=length(tr_idx), n_test=nrow(heart)-length(tr_idx),
         pct_train=round(split_ratio*100), pct_test=round((1-split_ratio)*100))
  })
  
  output$s4_split_display <- renderUI({
    d <- s4_data()
    tags$div(
      style="display:flex;gap:4px;margin-top:-8px;margin-bottom:4px;border-radius:6px;overflow:hidden;font-size:0.75rem;font-weight:600;",
      tags$div(style=paste0("flex:",d$pct_train,";background:#1a5f7a;color:#dff0f8;padding:5px 8px;text-align:center;"),
               paste0("Train ",d$pct_train,"% (n=",d$n_train,")")),
      tags$div(style=paste0("flex:",d$pct_test,";background:#2e0a0a;color:#f87171;padding:5px 8px;text-align:center;"),
               paste0("Test ",d$pct_test,"% (n=",d$n_test,")"))
    )
  })
  
  full_tree_model <- reactive({
    tr <- s4_data()$train
    if (!is.null(input$s4_prune_mode) && input$s4_prune_mode == "manual") {
      req(input$s4_depth, input$s4_cp)
      rpart(target~., data=tr, method="class",
            control=rpart.control(maxdepth=input$s4_depth, cp=input$s4_cp, minsplit=2))
    } else {
      rpart(target~., data=tr, method="class")
    }
  })
  
  observeEvent(full_tree_model(), {
    session$sendCustomMessage("s4td", list(nodes=rpart_to_nodelist(full_tree_model())))
    session$sendCustomMessage("s4go", -1)
  }, ignoreNULL=TRUE)
  
  output$tab4_metrics_ui <- renderUI({
    m     <- full_tree_model()
    te    <- s4_data()$test
    preds <- predict(m, te, type="class"); actual <- te$target
    cm    <- table(Predicted=preds, Actual=actual)
    acc   <- sum(diag(cm))/sum(cm)
    sens  <- if(sum(actual=="1")>0) cm["1","1"]/sum(actual=="1") else 0
    spec  <- if(sum(actual=="0")>0) cm["0","0"]/sum(actual=="0") else 0
    
    metric_label <- function(txt, tip) {
      tags$div(style="font-size:0.72rem;color:#7ec8de;text-transform:uppercase;letter-spacing:0.5px;display:flex;align-items:center;gap:4px;",
               txt,
               tooltip(trigger=bsicons::bs_icon("info-circle", size="0.85em", style="color:#57c5e0;cursor:pointer;"), tip)
      )
    }
    
    tags$div(
      style="display:flex;gap:0;background:linear-gradient(135deg,#0a1e35,#0d2641);border:1px solid #1e3f5c;border-radius:10px;overflow:hidden;margin-bottom:12px;",
      tags$div(style="flex:1;padding:10px 16px;border-right:1px solid #1e3f5c;",
               metric_label("Real-World Accuracy", "The percentage of unseen test patients the tree correctly classified. Unlike training accuracy, this reflects true performance on new data the tree has never seen before."),
               tags$div(style="font-size:1.5rem;font-weight:700;color:#4ade80;", paste0(round(acc*100,1),"%")),
               tags$div(style="font-size:0.72rem;color:#64748b;","on unseen testing data")),
      tags$div(style="flex:1;padding:10px 16px;border-right:1px solid #1e3f5c;",
               metric_label("Sensitivity", "Of all patients who actually had heart disease, what % did the tree correctly catch? This is critical in clinical settings — a low sensitivity means real cases are being missed."),
               tags$div(style="font-size:1.5rem;font-weight:700;color:#f87171;", paste0(round(sens*100,1),"%")),
               tags$div(style="font-size:0.72rem;color:#64748b;","heart disease correctly caught")),
      tags$div(style="flex:1;padding:10px 16px;",
               metric_label("Specificity", "Of all patients who were actually healthy, what % did the tree correctly clear? Low specificity means healthy patients are being unnecessarily flagged as sick."),
               tags$div(style="font-size:1.5rem;font-weight:700;color:#57c5e0;", paste0(round(spec*100,1),"%")),
               tags$div(style="font-size:0.72rem;color:#64748b;","healthy patients correctly cleared"))
    )
  })
  
  patient_pred <- reactiveVal(NULL)
  observeEvent(input$btn_predict, {
    tr  <- s4_data()$train
    one <- tr[1,,drop=FALSE]
    one$age     <- input$p_age
    one$sex     <- factor(input$p_sex,     levels=levels(tr$sex))
    one$cp      <- factor(input$p_cp,      levels=levels(tr$cp))
    one$trestbps<- input$p_trestbps
    one$chol    <- input$p_chol
    one$fbs     <- factor(input$p_fbs,     levels=levels(tr$fbs))
    one$restecg <- factor(input$p_restecg, levels=levels(tr$restecg))
    one$thalach <- input$p_thalach
    one$exang   <- factor(input$p_exang,   levels=levels(tr$exang))
    one$oldpeak <- input$p_oldpeak
    one$slope   <- factor(input$p_slope,   levels=levels(tr$slope))
    one$ca      <- factor(input$p_ca,      levels=levels(tr$ca))
    one$thal    <- factor(input$p_thal,    levels=levels(tr$thal))
    
    m   <- full_tree_model()
    two <- rbind(one, one)
    pc  <- as.character(predict(m, two, type="class")[1])
    patient_pred(pc)
    
    # Use friend's reliable method: match leaf via yval2 count columns
    leaf_id <- tryCatch({
      pred_mat  <- predict(m, two, type="matrix")
      leaf_idx  <- which(
        m$frame$var == "<leaf>" &
          m$frame$yval2[, 2] == pred_mat[1, 2] &
          m$frame$yval2[, 3] == pred_mat[1, 3]
      )
      if (length(leaf_idx) > 0) as.integer(rownames(m$frame)[leaf_idx[1]]) else NULL
    }, error = function(e) NULL)
    
    if (!is.null(leaf_id)) {
      path <- integer(0)
      curr <- leaf_id
      repeat {
        path <- c(path, curr)
        if (curr == 1L) break
        curr <- curr %/% 2L
      }
      session$sendCustomMessage("s4path", list(path=as.list(path)))
    } else {
      session$sendCustomMessage("s4path", list(path=list()))
    }
  })
  
  output$predicted_condition_ui <- renderUI({
    p <- patient_pred(); if(is.null(p)) return(NULL)
    label <- if(p=="1") "🫀 Heart Disease Detected" else "💚 No Heart Disease"
    cls   <- if(p=="1") "pred-badge-disease" else "pred-badge-healthy"
    tags$div(style="margin-top:10px;",
             tags$div(style="font-size:0.85rem;color:#7ec8de;","Predicted condition:"),
             tags$div(class=cls, label))
  })
  
  observeEvent(input$btn_reset_patient, {
    patient_pred(NULL)
    session$sendCustomMessage("s4path", list(path=list()))
    updateSliderInput(session, "p_age",     value=55)
    updateRadioButtons(session, "p_sex",    selected="1")
    updateSelectInput(session, "p_cp",      selected="3")
    updateRadioButtons(session, "p_exang",  selected="0")
    updateSliderInput(session, "p_trestbps",value=130)
    updateSliderInput(session, "p_chol",    value=240)
    updateRadioButtons(session, "p_fbs",    selected="0")
    updateSliderInput(session, "p_thalach", value=150)
    updateSelectInput(session, "p_restecg", selected="0")
    updateNumericInput(session, "p_oldpeak",value=1.0)
    updateSelectInput(session, "p_slope",   selected="1")
    updateSelectInput(session, "p_ca",      selected="0")
    updateSelectInput(session, "p_thal",    selected="1")
  })
  
  observeEvent(full_tree_model(), {
    patient_pred(NULL)
    session$sendCustomMessage("s4path", list(path=list()))
  })
  
  # ── Tab 5: Ensembles ──────────────────────────────────────────────────────
  rf_model <- reactive({
    randomForest(target~., data=train, ntree=100, importance=TRUE)
  })
  
  output$plot_rf_imp <- renderPlot({
    old <- par(no.readonly=TRUE); on.exit(par(old))
    imp <- as.data.frame(importance(rf_model()))
    imp$Variable <- rownames(imp)
    imp <- imp[order(imp$MeanDecreaseAccuracy),]
    imp$Variable <- factor(imp$Variable, levels=imp$Variable)
    ggplot(imp, aes(x=Variable, y=MeanDecreaseAccuracy)) +
      geom_col(fill="#57c5e0", alpha=0.85) +
      coord_flip() + dark_ggplot_theme() +
      labs(title="🌲 Random Forest — Variable Importance", x=NULL, y="Mean Decrease Accuracy")
  })
  
  output$plot_rf_cm_heat <- renderPlot({
    old <- par(no.readonly=TRUE); on.exit(par(old))
    prob1 <- get_prob1_safe(predict(rf_model(), test, type="prob"))
    m     <- compute_metrics(test$target, ifelse(prob1>=0.5,"1","0"), prob1)
    plot_cm_heatmap(m$cm, "🌲 RF Confusion Matrix (Test Data)")
  })
  
  xgb_mod <- boost_tree(mode="classification", engine="xgboost", trees=5) |>
    fit(target~., data=prep(heart.recipe)|>bake(train))
  
  output$plot_xgb_tree <- renderGrViz({
    xgb.plot.tree(extract_fit_engine(xgb_mod), tree_idx=1)
  })
}

shinyApp(ui, server)