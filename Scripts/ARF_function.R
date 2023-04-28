library(arf)
library(tidyr)

run_arf <- function(df,nr_trees,output_path,name){
  
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
  arf <- adversarial_rf(df,verbose=TRUE,num_trees=nr_trees)
  params <- forde(arf,df)
  saveRDS(params,file=name)
  syn <- forge(params, n_synth = length(df))

}