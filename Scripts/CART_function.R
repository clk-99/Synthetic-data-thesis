library(synthpop)
library(tidyverse)
library(Gmisc)

run_cart <- function(df,model_name,cat_columns){

  sequence <- rep(0,length(colnames(df)))

  for (i in 1:ncol(df)){
    if(colnames(df[i]) %in% cat_columns){
      df[,i]=factor(df[,i])
      nr_factors <- nlevels(df[,i])
      sequence[i] <- nr_factors
      }else{
        df[,i]=as.numeric(df[,i])
        sequence[i] <- 1
      }
  }
  print(sequence)
  df_sequence <- as.matrix(sequence)
  
  cart <- syn(df, visit.sequence = order(df_sequence))
  saveRDS(cart,file=model_name)
  
  # later stadium ook aanpassen van type model voor hyperparameter tuning
  synth <- cart$syn
  
  synth
  
}