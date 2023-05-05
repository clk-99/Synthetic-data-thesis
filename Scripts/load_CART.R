library(synthpop)

load_cart <- function(df,output_path,cat_columns,saved_model){
  setwd(output_path)
  cart <- readRDS(saved_model)
  
  synth <- cart$syn
  
  for(i in colnames(df)){
    if(i %in% cat_columns){
    synth[,i] <- as.character(synth[,i)
    }
  }
  synth
}
