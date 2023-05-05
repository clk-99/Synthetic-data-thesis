library(arf)
library(tidyr)

run_arf <- function(df,nr_trees,cat_columns,name){

  for (i in colnames(df)){
    if(i %in% cat_columns){
      df[,i]=factor(df[,i])
      
    }else{
        df[,i]=as.numeric(df[,i])
       
    }
  }
  

  arf <- adversarial_rf(df,verbose=TRUE,num_trees=nr_trees)
  params <- forde(arf,df)
  saveRDS(params,file=name)
  syn <- forge(params, n_synth = length(df))

}