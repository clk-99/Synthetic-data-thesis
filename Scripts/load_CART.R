library(synthpop)

load_cart <- function(df,output_path,saved_model){
  setwd(output_path)
  cart <- readRDS(saved_model)
  
  synth <- cart$syn
  
  syn
}