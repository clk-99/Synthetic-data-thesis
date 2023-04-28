library(arf)

load_arf <- function(df,output_path,saved_model){
  setwd(output_path)
  for (i in 1:ncol(df)){
    if(is.character(df[,i])){
      df[,i]=factor(df[,i])
      
    }
    else if(is.integer(df[,i])){
      min_max_values <- range(df[,i])
      if(min_max_values[2]<=4){
        df[,i]=factor(df[,i])
        
      }else{
        df[,i]=as.numeric(df[,i])
        
      }
    }
  }
  params <- readRDS(saved_model)
 # params <- forde(arf,df)
  synth <- forge(params, n_synth = length(df))
  
  synth
}