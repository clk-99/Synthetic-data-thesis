import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_heatmaps(df,datatype,real=True):
    fig, axs = plt.subplots(figsize=(11,9))
    features = df.columns.to_list()
    df_num = df.select_dtypes(include='object').columns.to_list()
    df[df_num] = df[df_num].astype('int')    

    cmap = sns.diverging_palette(230,20,as_cmap=True) #choose color

    corr = df.corr()
    mask = np.triu(np.ones_like(corr,dtype=bool))
    sns.heatmap(corr,annot=True,fmt='.2f',mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=.5,cbar_kws={"shrink":.5})
    if real:
        plt.title("Heatmap correlations for real dataset "+str(datatype))
        plt.savefig("heatmap_"+str(datatype)+"_real.pdf",bbox_inches='tight')
    else:
        plt.title("Heatmap correlations for synthetic dataset "+str(datatype))
        plt.savefig("heatmap_"+str(datatype)+"_synthetic.pdf",bbox_inches='tight')
    

def create_boxplots(real_df,syn_df,datatype):    
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')
    concatenated = pd.concat([real_df.assign(dataset='Real'),syn_df.assign(dataset='Synthetic')],ignore_index=True)
    for f in features:      
        fig, axs = plt.subplots(figsize=(11,9))  
        sns.boxplot(x=f,y='dataset',data=concatenated,orient='h')
        plt.title("Boxplots real vs synthetic data for feature "+str(f))
        #plt.show()
        plt.savefig("boxplot_feature_"+str(f)+".pdf",bbox_inches='tight')
        plt.close()
    

def create_kdeplots(real_df,syn_df,datatype):    
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')
    concatenated = pd.concat([real_df.assign(dataset='Real'),syn_df.assign(dataset='Synthetic')],ignore_index=True)
    for f in features:
        fig, axs = plt.subplots(figsize=(11,9))
        sns.kdeplot(x=f,data=concatenated,hue='dataset',common_norm=True,bw_adjust=.2)
        plt.title("KDEplots real vs synthetic data for feature "+str(f))
        plt.savefig("kdeplot_feature_"+str(f)+".pdf",bbox_inches='tight')
        plt.close()
    

def create_violinplots(real_df,syn_df,datatype):
    
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')
    concatenated = pd.concat([real_df.assign(dataset='Real'),syn_df.assign(dataset='Synthetic')],ignore_index=True)

    for f in features:
        fig, axs = plt.subplots(figsize=(11,9))
        sns.violinplot(x='dataset',y=f,data=concatenated)
        plt.title("Violinplots real vs synthetic data for feature "+str(f))
        plt.savefig("violinplots_feature_"+str(f)+".pdf",bbox_inches='tight')
        plt.close()

def create_ecdfplots(real_df,syn_df,datatype):    
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')
    concatenated = pd.concat([real_df.assign(dataset='Real'),syn_df.assign(dataset='Synthetic')],ignore_index=True)

    for f in features:
        fig, axs = plt.subplots(figsize=(11,9))
        sns.ecdfplot(x=f,data=concatenated,hue='dataset',stat='count')
        plt.title("ECDF plots real vs synthetic data for feature "+str(f))
        plt.savefig("ecdfplots_feature_"+str(f)+".pdf")
        plt.close()
    
def create_pairplot(df):
    features = df.columns.to_list()
    target = features[-1]
    features.pop(-1)

    g = sns.PairGrid(df, hue=target, vars=features)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()

def create_model_performance_plot(data_type,df,x,y):
    #os.chdir(output_path)
    fig, ax = plt.subplots(figsize=(11,9))

    df[["Model_type","id"]] = df.Saved_model.str.split("_",expand=True)

    ax = sns.scatterplot(data=df,x=x,y=y,hue='Model_type')
    ax.collections[0].set_sizes([200])
    ax.set_title("Generators performance plot for data "+str(data_type)+" between "+str(x)+" and "+str(y))
    fig.savefig("lineplot_performance_"+str(x)+"_"+str(y)+".pdf")
    plt.close()

def create_arf_performance_plot(data_type,df,x,y):
    #os.chdir(output_path)
    fig, ax = plt.subplots(figsize=(11,9))

    df[["Model_type","id"]] = df.Saved_model.str.split("_",expand=True)
    df = df[df["Model_type"]=='ARF']
    ax = sns.scatterplot(data=df,x=x,y=y,hue='Model_type')
    ax.collections[0].set_sizes([200])
    ax.set_title("ARF performance plot for data "+str(data_type)+" between "+str(x)+" and "+str(y))
    fig.savefig("lineplot_performance_"+str(x)+"_"+str(y)+".pdf")
    plt.close()

def create_dgn_performance_plot(data_type,model,df,x,y):
    #os.chdir(output_path)
    fig, ax = plt.subplots(figsize=(11,9))

    df[["Model_type","id"]] = df.Saved_model.str.split("_",expand=True)
    df = df[df["Model_type"]==model]
    ax = sns.scatterplot(data=df,x=x,y=y,hue='Model_type')
    ax.collections[0].set_sizes([200])
    ax.set_title(str(model)+" performance plot for data "+str(data_type)+" between "+str(x)+" and "+str(y))
    fig.savefig(model+"_plot_performance_"+str(x)+"_"+str(y)+".pdf")
    plt.close()

def create_performance_SDGs_all(df,datasets,datapath):
    x='Train_time(in seconds)'
    y='TabSynDex_score'
    fig, ax = plt.subplots(figsize=(11,9))
    df[["Model_type","id"]] = df.Saved_model.str.split("_",expand=True)    

    ax = sns.scatterplot(data=df,x=x,y=y,style='Dataset',hue='Model_type')
    ax.collections[0].set_sizes([200])
    ax.set_title("All SDGs performance plot for all datasets between "+str(x)+" and "+str(y))
    fig.savefig(datapath+"/all_datasets_plot_performance_"+str(x)+"_"+str(y)+".pdf")
    plt.close()

    

def find_best_clusters(df, max_K):
    clusters_centers = []
    k_values = []

    for k in range(1,max_K):

        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(df)

        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
    
    return clusters_centers, k_values

def generate_elbow_plot(clusters_centers,k_values):

    figure = plt.subplots(figsize=(12,6))
    plt.plot(k_values,clusters_centers,'o-',color='orange')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Cluster Intertia')
    plt.title('Elbow Plot of KMeams')
    plt.show()

def determine_clusters(df):
    #transform data
    scalar = StandardScaler()
    scalar.fit(df)
    scaled_data = scalar.transform(df)
    clusters_centers, k_values = find_best_clusters(scaled_data, 20)

    generate_elbow_plot(clusters_centers, k_values)

    return scaled_data

def create_clusters(df,f1,f2,K):

    scaled_data = determine_clusters(df)
    kmeans_model = KMeans(n_clusters=K)
    kmeans_model.fit(scaled_data)
    df['clusters'] = kmeans_model.labels_

    print(df.head())

    plt.scatter(df[f1], df[f2], c=df['clusters'])
    
    return df

def find_outliers(df,features,pct):
    
    model_IF = IsolationForest(contamination=float(pct),random_state=12)
    model_IF.fit(df[features])
    df['anomaly_scores'] = model_IF.decision_function(df[features])
    df['anomaly'] = model_IF.predict(df[features])
    print(df.head())

    #visualize dataset
    palette = ['#ff7f0e','#1f77b4']
    sns.pairplot(df,vars=df.columns.to_list(),hue='anomaly',palette=palette)
    
    return df

def visualize_data_pca(df):
    #For reproducability of the results
    np.random.seed(42)

    rndperm = np.random.permutation(df.shape[0])
    features = df.columns.to_list()
    target = features[-1] #moet dit wel zo?
    features.pop(-1)

    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[features].values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(8,5))
    sns.scatterplot(x="pca-one",y="pca-two",hue=target,data=df.loc[rndperm,:],legend="full",alpha=0.4)
    
    return df
