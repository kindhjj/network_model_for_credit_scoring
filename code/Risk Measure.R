library(psycho)
library(tidyverse)
library(dplyr)

library(gcdnet)
library(glmnet)
library(pROC)
library(kableExtra)
library(ggplot2)

#----------read data and standardization--------------
dat <-
  read.csv("/data/SME_dataset.csv")
input_x <- as.matrix(dat)
input_x <- input_x[,-c(1, 2)]
input_x_std <- input_x %>%
  as_tibble(.) %>%
  mutate_all(., scale) %>% as.matrix(.)

#------------connection (SVD)----------------------
# pca <- prcomp(input_x_std)
# summary(pca)
#
# fu <- svd(input_x_std)
# U <- fu$u[, 1:16]
# D <- diag(fu$d[1:16])
# phi <- qnorm(2 / dim(input_x)[1])
# p_ij <- U %*% D %>%
#   t() %>%
#   cov() %>%
#   `+`(phi) %>%
#   pnorm()
# p_hat <- (p_ij > 0.1)
# sumConnect <- function(x) {
#   z <- map(x, (function(x) {
#     if (x > 0 && x < 30) {
#       1
#     } else {
#       0
#     }
#   })) %>% as.numeric()
#   return(z)
# }
#
# correlatedComp <- p_hat %>%
#   rowSums() %>%
#   sumConnect()

#------------segregate comp---------
comp_str <- c(
  "/result/connect_flag_df_threshold_0.1.csv",
  "/result/connect_flag_df_threshold_0.05.csv"
)
c_str <- c("0.1 gamma C", "0.05 gamma C")
nc_str <- c("0.1 gamma NC", "0.05 gamma NC")
correlatedComp <- comp_str %>%
  map(., ~ read.csv(.)) %>%
  setNames(c_str)
repNum <- function(x) {
  x_cf <- x$connect_flag
  y <- seq(1, length(x_cf))
  z <- map2(.x = x_cf, .y = y, (function(x, y) {
    if (x == 1) {
      y
    } else {
      0
    }
  }))
  z <- z[z != 0]
  return(z)
}
correlatedCompNo <-
  correlatedComp %>% map(., ~ repNum(.) %>% unlist(.))
unCorrelatedCompNo <-
  map2(.x = correlatedCompNo, .y = correlatedComp, ~ seq(1, length(.y$connect_flag))[-.x]) %>% setNames(nc_str)
allSample <- list(seq(1, dim(dat)[1])) %>% setNames("all sample")
allInput <- c(correlatedCompNo, unCorrelatedCompNo, allSample)

#------------LASSO/elasttic-net-------
X_test_num <-
  map2(.x = allInput,
       .y = allInput,
       ~ sample(.x, length(.y) /
                  10 %>% round()))
X_test <- X_test_num %>% map(., ~ input_x_std[c(.),])
X_train_num <-
  map(X_test, ~ setdiff(allSample[[1]], .x) %>% sample())
X_train <- X_train_num %>% map(., ~ input_x_std[c(.),])
Y_train <- map(X_train_num, ~ dat$status[c(.)])
Y_test <- map(X_test_num, ~ dat$status[c(.)])
X_all_num <- map(.x = allInput, ~ sample(.x))
X_all <- X_all_num %>% map(., ~ input_x_std[c(.),])
Y_all <- map(X_all_num, ~ dat$status[c(.)]) %>%
  map(., ~ map(., as.factor)) %>%
  map(., unlist)

#-------------lasso-------------------
lasso_reg <- map2(
  .x = X_all,
  .y = Y_all,
  ~ cv.glmnet(.x, .y, family = "binomial", type.measure = "mse")
)
lasso_coef <- map(lasso_reg, ~ coef(., s = "lambda.min"))
lasso_predict <- map2(.x = lasso_reg,
                      .y = X_test,
                      ~ predict(.x, .y, type = "response", s = "lambda.min"))
cbdList <- function(x) {
  cbdName <- c("0.1 gamma", "0.05 gamma", "all sample")
  z <-
    list(c(x[[1]], x[[3]]), c(x[[2]], x[[4]]), x[[5]]) %>% setNames(cbdName)
  return(z)
}
Y_test_cbd <- cbdList(Y_test)

lasso_predict_cbd <- cbdList(lasso_predict)
lasso_roc <-
  map2(.x = lasso_predict_cbd, .y = Y_test_cbd, ~ roc(.y, .x))
lasso_roc

#-------------elastic-Net-------------------
ela_reg <- map2(
  .x = X_all,
  .y = Y_all,
  ~ cv.glmnet(
    .x,
    .y,
    alpha = 0.5,
    family = "binomial",
    type.measure = "mse"
  )
)
ela_coef <- map(ela_reg, ~ coef(., s = "lambda.min"))
ela_predict <- map2(.x = ela_reg,
                    .y = X_test,
                    ~ predict(.x, .y, type = "response", s = "lambda.min"))
ela_predict_cbd <- cbdList(ela_predict)
ela_roc <-
  map2(.x = ela_predict_cbd, .y = Y_test_cbd, ~ roc(.y, .x))
ela_roc

#-------------adaptive lasso-------------------
ridge_reg <- map2(
  .x = X_all,
  .y = Y_all,
  ~ cv.glmnet(
    .x,
    .y,
    alpha = 0,
    family = "binomial",
    type.measure = "mse"
  )
)
ridge_coef <- map(ridge_reg, ~ coef(., s = "lambda.min")) %>%
  map(., as.numeric) %>%
  map(., abs)
XY_all <-
  map2(.x = X_all, .y = Y_all, ~ list(.x, .y)) %>% map(., ~ setNames(., c("X", "Y")))
ada_lasso_reg <- map2(
  .x = XY_all,
  .y = ridge_coef,
  ~ cv.glmnet(
    .x[[1]],
    .x[[2]],
    penalty.factor = 1 / .y,
    family = "binomial",
    type.measure = "mse"
  )
)
ada_lasso_coef <- map(ada_lasso_reg, ~ coef(., s = "lambda.min"))
ada_lasso_predict <- map2(.x = ada_lasso_reg,
                          .y = X_test,
                          ~ predict(.x, .y, type = "response", s = "lambda.min"))
ada_lasso_predict_cbd <- cbdList(ada_lasso_predict)
ada_lasso_roc <-
  map2(.x = ada_lasso_predict_cbd, .y = Y_test_cbd, ~ roc(.y, .x))
ada_lasso_roc

#-------------adaptive elastic-Net-------------------
ada_ela_reg <- map2(
  .x = XY_all,
  .y = ridge_coef,
  ~ cv.glmnet(
    .x[[1]],
    .x[[2]],
    alpha = 0.5,
    penalty.factor = 1 / .y,
    family = "binomial",
    type.measure = "mse"
  )
)
ada_ela_coef <- map(ada_ela_reg, ~ coef(., s = "lambda.min"))
ada_ela_predict <- map2(.x = ada_ela_reg,
                        .y = X_test,
                        ~ predict(.x, .y, type = "response", s = "lambda.min"))
ada_ela_predict_cbd <- cbdList(ada_ela_predict)
ada_ela_roc <-
  map2(.x = ada_ela_predict_cbd, .y = Y_test_cbd, ~ roc(.y, .x))
ada_ela_roc

#------------combine outcome----------------------
allName <- c(c_str, nc_str, "all sample")
lasso_coef <-
  lasso_coef %>% setNames(., map(allName, ~ paste(., ".lasso")))
ela_coef <-
  ela_coef %>% setNames(., map(allName, ~ paste(., ".ela")))
ada_lasso_coef <-
  ada_lasso_coef %>% setNames(., map(allName, ~ paste(., ".ada_lasso")))
ada_ela_coef <-
  ada_ela_coef %>% setNames(., map(allName, ~ paste(., ".ada_ela")))
all_coef <- c(lasso_coef, ela_coef, ada_lasso_coef, ada_ela_coef)
allName <- names(all_coef)
all_coef_dfr <- all_coef %>%
  map(., as.matrix %>% as.data.frame()) %>%
  (function(x) {
    do.call(cbind, x)
  })
names(all_coef_dfr) <- allName
all_coef_dfr %>%
  kable(.) %>%
  kable_styling(.,
                bootstrap_options = c("striped", "hover", "condensed", "reponsive"))


multiplot <-
  function(...,
           plotlist = NULL,
           file,
           cols = 1,
           layout = NULL) {
    library(grid)
    plots <- c(list(...), plotlist)
    numPlots <- length(plots)
    if (is.null(layout)) {
      layout <- matrix(seq(1, cols * ceiling(numPlots / cols)),
                       ncol = cols,
                       nrow = ceiling(numPlots / cols))
    }
    if (numPlots == 1) {
      print(plots[[1]])
    } else {
      grid.newpage()
      pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
      
      for (i in 1:numPlots) {
        matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
        
        print(plots[[i]],
              vp = viewport(
                layout.pos.row = matchidx$row,
                layout.pos.col = matchidx$col
              ))
      }
    }
  }


g1 <-
  ggroc(lasso_roc) + xlab("FPR") + ylab("TPR") + ggtitle("Lasso") + geom_segment(aes(
    x = 0,
    xend = 1,
    y = 0,
    yend = 1
  ),
  color = "darkgrey",
  linetype = "dashed")
g2 <-
  ggroc(ela_roc) + xlab("FPR") + ylab("TPR") + ggtitle("Elastic-Net") +
  geom_segment(aes(
    x = 0,
    xend = 1,
    y = 0,
    yend = 1
  ),
  color = "darkgrey",
  linetype = "dashed")
g3 <-
  ggroc(ada_lasso_roc) + xlab("FPR") + ylab("TPR") + ggtitle("Adaptive lasso") +
  geom_segment(aes(
    x = 0,
    xend = 1,
    y = 0,
    yend = 1
  ),
  color = "darkgrey",
  linetype = "dashed")
g4 <-
  ggroc(ada_ela_roc) + xlab("FPR") + ylab("TPR") + ggtitle("Adaptive Elastic-Net") +
  geom_segment(aes(
    x = 0,
    xend = 1,
    y = 0,
    yend = 1
  ),
  color = "darkgrey",
  linetype = "dashed")
multiplot(g1, g2, g3, g4, cols = 2)

#-------------roc test------------------
roc.test(lasso_roc[[1]], lasso_roc[[3]])
roc.test(lasso_roc[[2]], lasso_roc[[3]])
roc.test(ada_lasso_roc[[3]], ada_lasso_roc[[1]])
roc.test(ada_lasso_roc[[3]], ada_lasso_roc[[2]])
roc.test(ela_roc[[3]], ela_roc[[1]])
roc.test(ela_roc[[3]], ela_roc[[2]])
roc.test(ada_ela_roc[[3]], ada_ela_roc[[1]])
roc.test(ada_ela_roc[[3]], ada_ela_roc[[2]])
