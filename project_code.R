# NAME: Danica Pham | NUMBER: 5629

library(readr)
library(dplyr)
library(xgboost)
library(Matrix)
library(Metrics)
library(tidyr)
library(stringr)
library(Ckmeans.1d.dp)

train_df <- read_csv("training.csv.gz")
test_df <- read_csv("testing.csv.gz")

# ==============================================================================
# 1) DATA SETUP AND FEATURE ENGINEERING
# ==============================================================================

# feature engineering for closestdefapproach
engineer_features <- function(df) {
  df %>%
    mutate(closestdefapproach_clean = str_remove_all(closestdefapproach, "\\[|\\]")) %>%
    separate(closestdefapproach_clean, 
             into = c("def_dist_100", "def_dist_075", "def_dist_050", "def_dist_025"), 
             sep = ",", 
             convert = TRUE) %>%
    mutate(
      def_closeout_velocity = as.numeric(def_dist_100) - as.numeric(def_dist_025),
      is_closest_contesting = ifelse(!is.na(contester1_id) & closestdef_id == contester1_id, TRUE, FALSE),
      help_defense_required = ifelse(contested == TRUE & is_closest_contesting == FALSE, TRUE, FALSE),
      contester_distance_gap = ifelse(is.na(distcont1), 0, distcont1 - closestdefdist) 
    )
}

train_clean <- engineer_features(train_df)
test_clean <- engineer_features(test_df)

# factoring in shooter efficiency
shooter_stats <- train_clean %>%
  group_by(shooter_id) %>%
  summarize(
    shots_taken = n(),
    shooter_fg_pct = mean(outcome)
  ) %>%
  mutate(shooter_fg_pct = ifelse(shots_taken < 20, 0.45, shooter_fg_pct)) %>%
  select(shooter_id, shooter_fg_pct)

train_clean <- train_clean %>% 
  left_join(shooter_stats, by = "shooter_id") %>%
  mutate(shooter_fg_pct = replace_na(shooter_fg_pct, 0.45))

test_clean <- test_clean %>% 
  left_join(shooter_stats, by = "shooter_id") %>%
  mutate(shooter_fg_pct = replace_na(shooter_fg_pct, 0.45))

# all features used in model
features <- c("distance", "shotclock", "closestdefdist", "shottype", 
              "num_contesters", "dribblesbefore", "shooterspeed", 
              "locationx", "locationy", "three", "contested",
              "def_dist_100", "def_dist_075", "def_dist_050", "def_dist_025", 
              "def_closeout_velocity", "is_closest_contesting", 
              "help_defense_required", "contester_distance_gap", 
              "shooter_fg_pct")

# split training data 80/20
set.seed(42)
train_indices <- sample(seq_len(nrow(train_clean)), size = 0.8 * nrow(train_clean))

local_train <- train_clean[train_indices, ]
local_val <- train_clean[-train_indices, ]

# ==============================================================================
# XGBOOST MODEL
# ==============================================================================

options(na.action = "na.pass")

# sparse matrices for testing
train_matrix <- sparse.model.matrix(outcome ~ . - 1, data = local_train %>% select(all_of(features), outcome))
val_matrix <- sparse.model.matrix(outcome ~ . - 1, data = local_val %>% select(all_of(features), outcome))

dtrain <- xgb.DMatrix(data = train_matrix, label = local_train$outcome)
dval <- xgb.DMatrix(data = val_matrix, label = local_val$outcome)

xgb_params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 4,   
  eta = 0.02,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 5
)

# train the model (DISCLAIMER: This takes a while. But I promise it's working.)
xgb_model <- xgb.train(
  params = xgb_params, 
  data = dtrain, 
  nrounds = 1500, 
  watchlist = list(train = dtrain, eval = dval),
  early_stopping_rounds = 20, 
  print_every_n = 50
)

# ==============================================================================
# 2.5) EVALUATION
# ==============================================================================

val_probabilities <- predict(xgb_model, dval)

val_logloss <- logLoss(local_val$outcome, val_probabilities)

cat("\n--- LOCAL VALIDATION RESULTS ---\n")
cat("Final Validation Log-Loss:", round(val_logloss, 5), "\n\n")

val_comparison <- data.frame(
  Actual_Outcome = local_val$outcome,
  Predicted_Probability = round(val_probabilities, 4)
)

cat("--- FIRST 10 VALIDATION PREDICTIONS ---\n")
print(head(val_comparison, 10))

# ==============================================================================
# 3) GENERATE PREDICTIONS & EXPORT SUBMISSION
# ==============================================================================

# testing data matrix
test_prep <- test_clean %>% select(all_of(features))
test_prep$outcome <- 0 
test_matrix <- sparse.model.matrix(outcome ~ . - 1, data = test_prep)
dtest <- xgb.DMatrix(data = test_matrix)

final_probabilities <- predict(xgb_model, dtest)

# map shot_id to the predicted probability
submission <- data.frame(
  shot_id = test_clean$shot_id,
  make_prob = final_probabilities
)

write_csv(submission, "submission.csv")

# ==============================================================================
# 4) CHARTS & GRAPHS
# ==============================================================================

# FEATURE IMPORTANCE MATRIX 

importance_matrix <- xgb.importance(model = xgb_model)

xgb.plot.importance(importance_matrix, top_n = 15, measure = "Gain", 
                    main = "XGBoost Feature Importance")

# PROBABILITY DENSITY PLOT 

library(ggplot2)

ggplot(val_comparison, aes(x = Predicted_Probability, fill = as.factor(Actual_Outcome))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("red", "blue"), name = "Actual Outcome (Made)") +
  labs(title = "Distribution of Predicted Probabilities",
       x = "Predicted Probability",
       y = "Density") +
  theme_minimal()

# CALIBRATION CURVE

library(caret)

cal_data <- data.frame(
  Class = factor(ifelse(local_val$outcome == 1, "Made", "Missed")),
  Made = val_probabilities
)

cal_plot <- calibration(Class ~ Made, data = cal_data, class = "Made", cuts = 10)
plot(cal_plot, main = "Model Calibration Curve")

# SPATIAL SHOT

ggplot(train_clean, aes(x = locationy, y = locationx, z = outcome)) +
  stat_summary_hex(fun = mean, bins = 30, alpha = 0.9) +
  scale_fill_viridis_c(name = "Make %") +
  labs(title = "NBA Shot Efficiency by Court Location",
       x = "Baseline (Width)", 
       y = "Sideline (Length)") +
  theme_minimal() +
  coord_fixed()
