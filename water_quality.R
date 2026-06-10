# =============================================================
# Курсовая работа: Классификация качества воды методом LDA
# Мартиросян Р.С., ИНБО-21-24
# =============================================================

# --- Установка пакетов (выполнить один раз) ---
# install.packages("R.matlab")
# install.packages("MASS")
# install.packages("nnet")
# install.packages("writexl")

# --- Загрузка библиотек ---
library(R.matlab)
library(MASS)
library(nnet)
library(writexl)

# =============================================================
# 1. Загрузка и подготовка данных
# =============================================================
setwd("D:/ВУЗ/Дисциплины/4 семестр/ЯПСОД (R+GLARUS) (курсовая)/2. Датасет")

data <- readMat("water_quality.mat")

# X.tr хранится как список из 423 матриц (37 станций x 11 признаков)
mat_list <- lapply(data$X.tr, function(x) x[[1]])
X <- do.call(rbind, mat_list)
df <- as.data.frame(X)

colnames(df) <- c("SC_max", "SC_min", "SC_mean",
                  "pH_min", "pH_max", "pH_mean",
                  "DO_min", "DO_max", "DO_mean",
                  "Temp_min", "Temp_max")

# Y.tr — матрица 37 x 423
df$quality_index <- as.vector(data$Y.tr)

# Проверка
cat("Размер данных:", nrow(df), "x", ncol(df), "\n")
cat("Пропущенные значения:\n")
print(colSums(is.na(df)))
str(df)
head(df)

# =============================================================
# 2. Исследовательский анализ данных (EDA)
# =============================================================
library(ggplot2)

# Гистограммы распределений
df_long <- data.frame(
  value = unlist(df[, 1:11]),
  variable = rep(names(df)[1:11], each = nrow(df))
)
ggplot(df_long, aes(x = value)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Распределения параметров качества воды")

# Boxplot по 3 группам (квантили quality_index)
df$group <- cut(df$quality_index,
                breaks = quantile(df$quality_index, c(0, 1/3, 2/3, 1)),
                labels = c("низкое", "среднее", "высокое"),
                include.lowest = TRUE)

df_long2 <- data.frame(
  value = unlist(df[, 1:11]),
  variable = rep(names(df)[1:11], each = nrow(df)),
  group = rep(df$group, 11)
)
ggplot(df_long2, aes(x = group, y = value, fill = group)) +
  geom_boxplot(alpha = 0.7) +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal() +
  labs(title = "Распределения по группам качества",
       x = "Категория качества", y = "Значение")

# Корреляционная матрица
cor_matrix <- cor(df[, 1:11])
library(reshape2)
cor_melt <- melt(cor_matrix)
ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       limits = c(-1, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Корреляционная матрица")

# =============================================================
# 3. Описательная статистика
# =============================================================
stats_df <- data.frame(
  Параметр = names(df)[1:11],
  Среднее = round(sapply(df[, 1:11], mean), 4),
  Медиана = round(sapply(df[, 1:11], median), 4),
  Стд_откл = round(sapply(df[, 1:11], sd), 4),
  Мин = round(sapply(df[, 1:11], min), 4),
  Макс = round(sapply(df[, 1:11], max), 4)
)
print(stats_df)

# =============================================================
# 4. Проверка гипотез (ANOVA)
# =============================================================
anova_results <- data.frame(
  Параметр = names(df)[1:11],
  F_stat = NA,
  p_value = NA
)
for (i in 1:11) {
  formula <- as.formula(paste(names(df)[i], "~ group"))
  a <- aov(formula, data = df)
  s <- summary(a)
  anova_results$F_stat[i] <- round(s[[1]]$`F value`[1], 2)
  anova_results$p_value[i] <- format(s[[1]]$`Pr(>F)`[1], digits = 4)
}
print(anova_results)

# =============================================================
# 5. Множественная регрессия
# =============================================================
model_lm <- lm(quality_index ~ SC_max + SC_min + SC_mean +
                 pH_min + pH_max + pH_mean +
                 DO_min + DO_max + DO_mean +
                 Temp_min + Temp_max, data = df)
summary(model_lm)
cat("R-squared:", round(summary(model_lm)$r.squared, 4), "\n")

# =============================================================
# 6. Классификация LDA
# =============================================================
df$quality_cat <- cut(df$quality_index,
                      breaks = quantile(df$quality_index, c(0, 1/3, 2/3, 1)),
                      labels = c("низкое", "среднее", "высокое"),
                      include.lowest = TRUE)

set.seed(42)
idx <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))
train <- df[idx, ]
test <- df[-idx, ]

lda_model <- lda(quality_cat ~ SC_max + SC_min + SC_mean +
                   pH_min + pH_max + pH_mean +
                   DO_min + DO_max + DO_mean +
                   Temp_min + Temp_max, data = train)
print(lda_model)

lda_pred <- predict(lda_model, newdata = test)
cm <- table(Predicted = lda_pred$class, Actual = test$quality_cat)
print(cm)
cat("Точность LDA:", round(sum(diag(cm)) / sum(cm), 4), "\n")

# Метрики по классам
precision <- diag(prop.table(cm, 2))
recall <- diag(prop.table(cm, 1))
f1 <- 2 * precision * recall / (precision + recall)
metrics <- data.frame(
  Категория = c("низкое", "среднее", "высокое"),
  Precision = round(precision, 3),
  Recall = round(recall, 3),
  F1 = round(f1, 3)
)
print(metrics)

# =============================================================
# 7. Сравнение с логистической регрессией
# =============================================================
log_model <- multinom(quality_cat ~ SC_max + SC_min + SC_mean +
                        pH_min + pH_max + pH_mean +
                        DO_min + DO_max + DO_mean +
                        Temp_min + Temp_max,
                      data = train, trace = FALSE)

log_pred <- predict(log_model, newdata = test)
cm2 <- table(Predicted = log_pred, Actual = test$quality_cat)
print(cm2)
cat("Точность лог. регрессии:", round(sum(diag(cm2)) / sum(cm2), 4), "\n")

# =============================================================
# 8. Визуализация результатов LDA
# =============================================================
lda_values <- predict(lda_model, newdata = test)$x
plot_df <- data.frame(
  LD1 = lda_values[, 1],
  LD2 = lda_values[, 2],
  quality_cat = test$quality_cat
)

ggplot(plot_df, aes(x = LD1, y = LD2, color = quality_cat)) +
  geom_point(size = 1.5, alpha = 0.6) +
  stat_ellipse(level = 0.95) +
  theme_minimal() +
  labs(title = "Диаграмма рассеяния дискриминант-\nным осям",
       x = paste0("LD1 (", round(lda_model$svd[1]^2 / sum(lda_model$svd^2) * 100, 1), "%)"),
       y = paste0("LD2 (", round(lda_model$svd[2]^2 / sum(lda_model$svd^2) * 100, 1), "%)"),
       color = "Качество воды")

# =============================================================
# 9. Экспорт данных для Glarus BI
# =============================================================
write_xlsx(df, "water_quality.xlsx")
cat("Данные сохранены в water_quality.xlsx\n")
