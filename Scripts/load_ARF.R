library(arf)

load_arf <- function(df,cat_vars,saved_model){

  params <- readRDS(saved_model)
  synth <- forge(params, n_synth = nrow(df))

  for(i in colnames(df)){
    if(i %in% cat_vars){
    synth[,i] <- as.character(synth[,i])
    }
  }
  synth
}
