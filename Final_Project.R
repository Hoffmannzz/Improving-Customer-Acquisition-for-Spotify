### Spotify User Analysis project 

### Section B Team 24 Yidan Luo (yl599), Zhanwen Zhu (zz343), Ella Wang (yw621), Hoffmann Zhu (hzz2)

# Business Goal: Identify which customer segment of the free users are most likely to convert to premium
# and target that segment with either discounts or extra benefits

## Load libraries and dataset

library(tidyverse)
library(readr)
library(readxl)
library(rpart)
library(randomForest)
library(caret)
library(Metrics)
library(dplyr)
library(tidyr)
library(corrplot)
library(cluster)
library(factoextra)
library(fpc)
library(dplyr)
library(ggplot2)
library(glmnet)
library(xgboost) 
library(class) # For K-Nearest Neighbors
library(pROC) # For Drawing ROC Curve

getwd()
setwd("C:/Users/fireo/OneDrive/Desktop/Data Science for Business")
data <- read_excel("Spotify_data.xlsx")
ori_data <- read_excel("Spotify_data.xlsx")
### Summarize data
head(data)  # To see the first few rows
str(data)   # To see the structure of the data
summary(data)  # To get summary statistics

### Convert data type
data$Age <- as.factor(data$Age)
data$Gender <- as.factor(data$Gender)

### Explore gender
# Calculate the frequency of each gender
gender_counts <- table(data$Gender)
gender_counts
# Define custom colors for each gender
custom_colors <- c("skyblue", "salmon", "lightgreen", "lightcoral")

# Calculate percentages and format labels
percentage_labels <- sprintf("%.1f%%", 100 * gender_counts / sum(gender_counts))

# Create a pie chart for Gender with customizations
pie(gender_counts,
    labels = percentage_labels,  # Add labels to slices
    main = "Gender Distribution of Spotify Users",
    col = custom_colors, 
    border = "white", 
    cex = 0.7,       
    clockwise = TRUE,  
    init.angle = 90   
)

# Calculate percentages and format labels
percentage_labels <- sprintf("%.1f%%", 100 * gender_counts / sum(gender_counts))

# Add a legend with labels
legend("bottomleft", legend = names(gender_counts), title = "Gender", fill = custom_colors)

# Calculate the frequency of each age group
age_counts <- table(data$Age)

# Define custom colors for Age
custom_colors_age <- c("lightblue", "lightgreen", "lightcoral", "lightyellow", "lightpink")

# Calculate percentages and format labels for Age
percentage_labels_age <- sprintf("%.1f%%", 100 * age_counts / sum(age_counts))

# Create a pie chart for Age with customizations
pie(age_counts,
    labels = percentage_labels_age, 
    main = "Age Distribution of Spotify Users",
    col = custom_colors_age,  
    border = "white", 
    cex = 0.7,         
    clockwise = TRUE,  
    init.angle = 90    
)
legend("bottomleft", legend = names(age_counts), title = "Age Group", fill = custom_colors_age)


## Preferred content
# Calculate the frequency of each gender
content_counts <- table(data$preferred_listening_content)

# Define custom colors for each gender
custom_colors_content <- c("skyblue", "salmon")

# Calculate percentages and format labels
percentage_labels <- sprintf("%.1f%%", 100 * content_counts / sum(content_counts))

# Create a pie chart for Gender with customization
pie(content_counts,
    labels = percentage_labels,  # Add labels to slices
    main = "Preferred Listening Content Distribution of Spotify Users",
    col = custom_colors_content, 
    border = "white", 
    cex = 0.7,       
    clockwise = TRUE,  
    init.angle = 90   
)

# Add a legend with labels
legend("bottomleft", legend = names(gender_counts), title = "preferred_content", fill = custom_colors)

### Preferred Music Genres
# Calculate the count of each favorite music genre
genre_counts <- table(data$fav_music_genre)

# Create a data frame for plotting
genre_data <- data.frame(Genre = names(genre_counts), Count = as.numeric(genre_counts))

# Create the vertical bar plot
ggplot(data = genre_data, aes(x = Genre, y = Count)) +
  geom_bar(stat = "identity", fill = "lightblue") +
  labs(x = "Music genres", y = "Count", title = "Preferred Music Genres") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 

### Subscription and Gender
ggplot(data = data, aes(x = spotify_subscription_plan, fill = Gender)) +
  geom_bar(position = "dodge") +
  labs(title = "Count Plot of Spotify Subscription Plans by Gender") +
  theme_minimal()

### Subscription and Age
ggplot(data = data, aes(x = spotify_subscription_plan, fill = Age)) +
  geom_bar(position = "dodge") +
  labs(title = "Count Plot of Spotify Subscription Plans by Age") +
  theme_minimal()

### Corr visualization

# Define columns to encode
columns_to_encode <- c('Age', 'Gender', 'spotify_usage_period', 'spotify_listening_device',
                       'spotify_subscription_plan', 'premium_sub_willingness',
                       'preferred_premium_plan', 'preferred_listening_content',
                       'fav_music_genre', 'music_time_slot', 'music_Influencial_mood',
                       'music_lis_frequency', 'music_expl_method', 'pod_lis_frequency',
                       'fav_pod_genre', 'preferred_pod_format', 'pod_host_preference',
                       'preferred_pod_duration', 'pod_variety_satisfaction')

# Apply label encoding
for (col in columns_to_encode) {
  data[[col]] <- as.integer(factor(data[[col]]))
}

# Calculate the correlation matrix
correlation_matrix <- cor(data)

# Create a heatmap of the correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 0.7)

### Convert original data into factor for easier analysis
ori_data <- ori_data%>%mutate_if(is.character, as.factor)
colnames(ori_data) <- c("Age","Gender","Usage_Period","Listening_Device","Subscription",
                    "Subscription_Willingness","Plan_Type","Content","Favorite_Genre",
                    "Time_Slot","Mood","Frequency","Exploration_Method","Rating","Podcast_Frequency",
                    "Favorite_Podcast_Genre","Pref_Pod_Format","Pred_Host_Type","Pref_Pod_Duration","Variety_Satisfaction")

str(ori_data)

### Unsupervised Learning-- PCA and K-Means 

# Assuming 'ori_data' is your dataset
spotify_data <- ori_data

# Convert categorical data to dummy variables (one-hot encoding)
spotify_data_numeric <- model.matrix(~ . -1, data = spotify_data)

# Standardize the numeric data
scaled_spotify_data <- scale(spotify_data_numeric)

# Determine Optimal Number of Clusters using Elbow Method
elbowplot <- function(data, nc=15, seed=1234) {
  wss <- (nrow(data) - 1) * sum(apply(data, 2, var))
  for (i in 2:nc) {
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)
  }
  plot(1:nc, wss, type="b", xlab="Number of Clusters", ylab="Within Groups Sum of Squares")
}

elbowplot(scaled_spotify_data)

# K-means Clustering with the chosen cluster number ( in this case)
fit.km <- kmeans(scaled_spotify_data, 8, nstart=25)

# PCA for Visualization
pca_result <- prcomp(scaled_spotify_data)
plot_data <- data.frame(pca_result$x[,1:2])
colnames(plot_data) <- c("PC1", "PC2")
plot_data$cluster <- as.factor(fit.km$cluster)

# Visualize Clusters in the space of the first two principal components
ggplot(plot_data, aes(x=PC1, y=PC2, color=cluster)) +
  geom_point(size=2, alpha=0.65) +
  labs(title="Clusters in First Two Principal Components", x="Principal Component 1", y="Principal Component 2") +
  theme_minimal()

# Print the PCA loadings
pca_df <- data.frame(pca_result$rotation[,1:2])
pca_df$Features <- rownames(pca_df)
pca_df <- pca_df[, c("Features", colnames(pca_df)[!colnames(pca_df) %in% "Features"])]
write.csv(pca_df, "pca_summary.csv", row.names = FALSE)

# Summarize and Analyze Clusters
spotify_data$cluster <- fit.km$cluster
cluster_summary <- spotify_data %>%
  group_by(cluster) %>%
  summarise(
    avg_rating = mean(Rating, na.rm = TRUE),
    prop_willing_to_subscribe = mean(Subscription_Willingness == "Yes"),
    prop_currently_subscribed = mean(Subscription == "Premium")
  )

# Display the summary of clusters
print(cluster_summary)

# Analysis of clusters

# Visualizing Cluster Characteristics
# Plotting average rating for each cluster
ggplot(cluster_summary, aes(x=factor(cluster), y=avg_rating)) + 
  geom_bar(stat="identity") +
  labs(title="Average Rating by Cluster", x="Cluster", y="Average Rating")

# Note: Similar plots can be made for other features to understand the nature of each cluster.

# 3. Identifying Distinguishing Features of Clusters
# Calculating the overall mean for features
overall_mean <- colMeans(spotify_data[,-which(names(spotify_data) == "cluster")])

# Comparing cluster mean with overall mean
cluster_difference <- cluster_summary %>%
  mutate_if(is.numeric, list(diff = ~ . - overall_mean[.]))
print(cluster_difference)

# 4. Examine Proportional Differences
# Assuming 'Subscription_Willingness' and 'Subscription' are categorical features
subscription_summary <- spotify_data %>%
  group_by(cluster) %>%
  summarise(
    prop_willing_to_subscribe = mean(Subscription_Willingness == "Yes"),
    prop_currently_subscribed = mean(Subscription == "Premium")
  )

print(subscription_summary)


# Descriptive Statistics for Numeric Features
numeric_features <- c("Age", "Usage_Period", "Frequency", "Rating", "Podcast_Frequency", "Variety_Satisfaction")
numeric_summary <- spotify_data %>%
  group_by(cluster) %>%
  summarise_at(vars(numeric_features), funs(mean=mean, sd=sd, median=median, min=min, max=max), na.rm = TRUE)
print(numeric_summary)

# Categorical Distribution for Categorical Features
categorical_features <- c("Gender", "Listening_Device", "Subscription", "Subscription_Willingness", 
                          "Plan_Type", "Content", "Favorite_Genre", "Time_Slot", "Mood", 
                          "Exploration_Method", "Favorite_Podcast_Genre", "Pref_Pod_Format", 
                          "Pred_Host_Type", "Pref_Pod_Duration")

for (feature in categorical_features) {
  cat_summary <- spotify_data %>%
    group_by(cluster, !!as.symbol(feature)) %>%
    summarise(count = n()) %>%
    mutate(percentage = count / sum(count) * 100)
  
  # Plotting for a visual representation
  ggplot(cat_summary, aes(x=factor(cluster), y=percentage, fill=!!as.symbol(feature))) +
    geom_bar(stat="identity", position="dodge") +
    labs(title=paste("Distribution of", feature, "by Cluster"), x="Cluster", y="Percentage") +
    theme_minimal()
  print(cat_summary)
}


# Modeling

data <- ori_data

# Convert Subscription_Willingness to factor format
data$Subscription_Willingness <- as.factor(data$Subscription_Willingness)

X <- model.matrix(~ . - Subscription_Willingness, data)
y <- data$Subscription_Willingness

# Create 10-fold cross-validation
cv_10fold <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Prepare potential classification models to compare using 10-fold cross-validation
models <- list(
  "Null" = train(X, y, method = "null", trControl = cv_10fold), # Null model as a baseline
  "CART" = train(X, y, method = "rpart", trControl = cv_10fold), # Decision Trees using rpart
  "Random_Forest" = train(X, y, method = "rf", trControl = cv_10fold),
  "XGBoost" = train(X, y, method = "xgbTree", trControl = cv_10fold),
  "KNN" = train(X, y, method = "knn", trControl = cv_10fold) # K-Nearest Neighbors
)

# Save cross-validation results
results <- resamples(models)

# Output summary of accuracy, Kappa, etc.
summary(results)


# Split the data into training and testing sets
set.seed(123) # for reproducibility
splitIndex <- createDataPartition(spotify_data$Subscription_Willingness, p = 0.75, list = FALSE)
train_data <- spotify_data[splitIndex,]
test_data <- spotify_data[-splitIndex,]

# Train the model on the training set
rf_mdl_train_test <- randomForest(Subscription_Willingness~., data = train_data)

# Predict on the test set
predictions <- predict(rf_mdl_train_test, test_data)

# Evaluate the model's performance
confusionMatrix(predictions, test_data$Subscription_Willingness)

prob_pred <- predict(rf_mdl_train_test, test_data, type="prob")
prob_pred <- data.frame(prob_pred)

# Extracting the probability of the 'Yes' class
prob_yes <- prob_pred$Yes

# Define the actual outcomes
actual_outcomes <- test_data$Subscription_Willingness

# Generate the ROC curve
roc_obj <- roc(actual_outcomes, prob_yes, levels = c("No", "Yes"), direction = "<")


# Extracting the coordinates of the ROC curve
roc_df <- data.frame(
  FPR = roc_obj$specificities,
  TPR = roc_obj$sensitivities
)

# Plotting using ggplot2
ggplot(data = roc_df, aes(x = 1 - FPR, y = TPR)) +
  geom_line(color = "blue", size = 1.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  ggtitle("ROC Curve") +
  xlab("False Positive Rate (1 - Specificity)") +
  ylab("True Positive Rate (Sensitivity)") +
  xlim(0, 1) +
  ylim(0, 1)







