library(arf)

load_arf <- function(df,output_path,cat_vars,saved_model){
  setwd(output_path)
 
  params <- readRDS(saved_model)
 # params <- forde(arf,df)
  synth <- forge(params, n_synth = length(df))
    
  for(i in colnames(df)){
    if(i %in% cat_vars){
    synth[,i] <- as.character(synth[,i)
    }
  }
  synth
}
