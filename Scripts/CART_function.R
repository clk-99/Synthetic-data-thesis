library(synthpop)
library(tidyverse)
library(Gmisc)

run_cart <- function(df,output_path,model_name){

  setwd(output_path)
  features <- colnames(df)
  sequence <- rep(0,length(features))

  for (i in 1:ncol(df)){
    if(is.character(df[,i])){
      df[,i]=factor(df[,i])
      nr_factors <- nlevels(df[,i])
      sequence[i] <- nr_factors
    }
    else if(is.integer(df[,i])){
      min_max_values <- range(df[,i])
      if(min_max_values[2]<=4){
        df[,i]=factor(df[,i])
        nr_factors <- nlevels(df[,i])
        sequence[i] <- nr_factors
      }else{
        df[,i]=as.numeric(df[,i])
        sequence[i] <- 1
      }
    } else {
       sequence[i] <- 1
    }
  }
  df_sequence <- as.matrix(sequence)
  
  cart <- syn(df, visit.sequence = order(df_sequence))
  saveRDS(cart,file=model_name)
  
  # later stadium ook aanpassen van type model voor hyperparameter tuning
  synth <- cart$syn
  
  synth
  
}