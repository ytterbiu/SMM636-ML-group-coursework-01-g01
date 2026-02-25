from shiny import App, render
from shiny import reactive
from shiny.ui import *
from shiny.ui import input_slider

import pandas as pd
import xgboost as xgb
import dtreeviz


def get_data_path():
    return("processed.cleveland.data.csv")

def load_dataset(path):
    df = pd.read_csv(path)
    return(df)


app_ui = page_navbar(
  nav_panel(
    "XGBoost",
    layout_sidebar(
      sidebar(
        h5("XGBoost parameters"),
        input_slider("xgb_num_boost_round", "Number of boosting rounds", min = 1, max = 10, value = 3, step = 1),
        input_slider("xgb_max_depth", "Max depth", min = 1, max = 5, value = 3, step = 1),
        input_slider("xgb_eta", "Learning rate", min = 0, max = 1, value = 1, step = 0.05),
        hr(),
        input_slider("xgb_tree_index", "Tree index to display", min = 1, max = 10, value = 1, step = 1),
        input_slider("xgb_depth_index", "Tree depths to display",  min = 1, max = 5, value = 3, step = 1),
        hr(),
        input_action_button("xgb_display_tree", "OK"),
      ),
      layout_column_wrap(
        card(card_header(tags.strong("XGBoost tree")), output_image("plot_xgb_tree", height = 560),class_ ="shadow-sm"),
        width = 1
      ),
    )
  ),
  title="Heart Disease — Tree-Based Methods",
)

def server(input, output, session):

    df = load_dataset( get_data_path() )
    features = [c for c in df.columns if c != 'target']
    target = "target"
    dtrain = xgb.DMatrix(df[features], df[target])

    
    @render.image
    @reactive.event(input.xgb_display_tree)
    def plot_xgb_tree():
        max_depth = input.xgb_max_depth()
        eta = input.xgb_eta()
        num_boost_round = input.xgb_num_boost_round()
        tree_index = input.xgb_tree_index() - 1
        depth_range = input.xgb_depth_index()
        assert(depth_range <= max_depth)

        params = {
            "max_depth": max_depth,
            "eta": eta,
            "objective":"binary:logistic",
            "subsample":1
            }
        
        xgb_model =  xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round= num_boost_round)

        viz_tr = dtreeviz.model(xgb_model, tree_index=tree_index,
            X_train=df[features], y_train=df[target],
            feature_names=features,
            target_name=target, class_names=["normal", "heart disease"])
        
        tmp_img_path = "./tmp/xgb_tree.svg"
        viz_tr.view(
                show_node_labels=True,
                depth_range_to_display=(0,depth_range)
            ).save(tmp_img_path)
        img = {"src": tmp_img_path}
        
        return img


# This is a shiny.App object. It must be named `app`.
app = App(app_ui, server)