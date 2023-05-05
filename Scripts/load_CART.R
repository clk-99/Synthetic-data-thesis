library(synthpop)

load_cart <- function(df,output_path,cat_columns,saved_model){
  setwd(output_path)
  cart <- readRDS(saved_model)
  
  synth <- cart$syn
  
  for(col in cat_columns){
    synth$col <- as.character(synth$col)
  }
  
  synth
}
