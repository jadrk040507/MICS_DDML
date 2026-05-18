# ==============================================================================
# MICS DoubleML - ML Learners
# ==============================================================================

library(mlr3)
library(mlr3learners)
library(mlr3pipelines)

# ==============================================================================
# LEARNER FACTORIES
# ==============================================================================

create_learners <- function(type = "binary") {
  # Create learners for DoubleML
  # type: "binary" for IRM (binary treatment/outcome), "continuous" for PLR
  
  learners <- list()
  
  # OLS (linear regression / logistic regression)
  if (type == "binary") {
    learners$ols <- list(
      g = lrn("regr.lm"),
      m = lrn("classif.log_reg", predict_type = "prob")
    )
  } else {
    learners$ols <- list(
      g = lrn("regr.lm"),
      m = lrn("regr.lm")
    )
  }
  
  # Lasso (alpha = 1)
  learners$lasso <- list(
    g = lrn("regr.cv_glmnet", alpha = 1),
    m = lrn("classif.cv_glmnet", alpha = 1, predict_type = "prob")
  )
  
  # Ridge (alpha = 0)
  learners$ridge <- list(
    g = lrn("regr.cv_glmnet", alpha = 0),
    m = lrn("classif.cv_glmnet", alpha = 0, predict_type = "prob")
  )
  
  # Elastic Net (alpha = 0.5)
  learners$enet <- list(
    g = lrn("regr.cv_glmnet", alpha = 0.5),
    m = lrn("classif.cv_glmnet", alpha = 0.5, predict_type = "prob")
  )
  
  # Random Forest
  learners$rf <- list(
    g = lrn("regr.ranger", num.trees = 500),
    m = lrn("classif.ranger", num.trees = 500, predict_type = "prob")
  )
  
  # XGBoost
  learners$xgb <- list(
    g = lrn("regr.xgboost", nrounds = 300, max_depth = 4, eta = 0.1, 
            subsample = 0.8, colsample_bytree = 0.8, verbose = 0),
    m = lrn("classif.xgboost", nrounds = 300, max_depth = 4, eta = 0.1,
            subsample = 0.8, colsample_bytree = 0.8, verbose = 0,
            predict_type = "prob")
  )
  
  return(learners)
}

# ==============================================================================
# STACKED ENSEMBLE
# ==============================================================================

create_stacked_ensemble <- function() {
  # Stacked ensemble for regression (g)
  po_enet_g <- PipeOpLearnerCV$new(
    lrn("regr.cv_glmnet", alpha = 0.5), id = "stack_enet_g"
  )
  po_rf_g <- PipeOpLearnerCV$new(
    lrn("regr.ranger", num.trees = 300), id = "stack_rf_g"
  )
  po_xgb_g <- PipeOpLearnerCV$new(
    lrn("regr.xgboost", nrounds = 200, max_depth = 4, eta = 0.1, verbose = 0),
    id = "stack_xgb_g"
  )
  po_union_g <- PipeOpFeatureUnion$new(
    c("stack_enet_g", "stack_rf_g", "stack_xgb_g"), id = "stack_union_g"
  )
  po_meta_g <- PipeOpLearner$new(
    lrn("regr.cv_glmnet", alpha = 0), id = "stack_meta_g"
  )
  po_copy_g <- PipeOpCopy$new(3)
  
  graph_g <- po_copy_g %>>%
    gunion(list(po_enet_g, po_rf_g, po_xgb_g)) %>>%
    po_union_g %>>%
    po_meta_g
  
  stacked_g <- GraphLearner$new(graph_g)
  
  # Stacked ensemble for classification (m)
  po_enet_m <- PipeOpLearnerCV$new(
    lrn("classif.cv_glmnet", alpha = 0.5, predict_type = "prob"), id = "stack_enet_m"
  )
  po_rf_m <- PipeOpLearnerCV$new(
    lrn("classif.ranger", num.trees = 300, predict_type = "prob"), id = "stack_rf_m"
  )
  po_xgb_m <- PipeOpLearnerCV$new(
    lrn("classif.xgboost", nrounds = 200, max_depth = 4, eta = 0.1, verbose = 0,
        predict_type = "prob"), id = "stack_xgb_m"
  )
  po_union_m <- PipeOpFeatureUnion$new(
    c("stack_enet_m", "stack_rf_m", "stack_xgb_m"), id = "stack_union_m"
  )
  po_meta_m <- PipeOpLearner$new(
    lrn("classif.cv_glmnet", alpha = 0, predict_type = "prob"), id = "stack_meta_m"
  )
  po_copy_m <- PipeOpCopy$new(3)
  
  graph_m <- po_copy_m %>>%
    gunion(list(po_enet_m, po_rf_m, po_xgb_m)) %>>%
    po_union_m %>>%
    po_meta_m
  
  stacked_m <- GraphLearner$new(graph_m)
  
  return(list(g = stacked_g, m = stacked_m))
}