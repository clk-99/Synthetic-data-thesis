import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_heatmaps(real_df,syn_df,datatype):
    fig, axs = plt.subplots(ncols=2)
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    print(df_num)
    real_df[df_num] = real_df[df_num].astype('int')
    print(real_df.dtypes)
    syn_df[df_num] = syn_df[df_num].astype('int')
    print(syn_df.dtypes)

    sns.heatmap(real_df,cmap='crest',annot=True,fmt='.1f',ax=axs[0])
    sns.heatmap(syn_df,cmap='crest',annot=True,fmt='.1f',ax=axs[1])
    #maybe also difference in correlations plot
    plt.title("Heatmap real vs synthetic data for dataset "+str(datatype))
    plt.legend()
    plt.savefig("heatmap_"+str(datatype)+".pdf")
    

def create_boxplots(real_df,syn_df,datatype):
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')

    for f in features:
        concatenated = pd.concat([real_df[f].assign(dataset='Real'),syn_df[f].assign(dataset='Synthetic')])
        sns.boxplot(y=f,data=concatenated,style='dataset')
        plt.title("Boxplots real vs synthetic data for feature "+str(f)+"on dataset "+str(datatype))
        plt.savefig("boxplot_feature_"+str(f)+".pdf")
    

def create_kdeplots(real_df,syn_df,datatype):
    features = real_df.columns.to_list()

    for f in features:
        concatenated = pd.concat([real_df[f].assign(dataset='Real'),syn_df[f].assign(dataset='Synthetic')])
        sns.kdeplot(y=f,data=concatenated,style='dataset')
        plt.title("KDEplots real vs synthetic data for feature "+str(f)+"on dataset "+str(datatype))
        plt.savefig("kdeplot_feature_"+str(f)+".pdf")
    

def create_violinplots(real_df,syn_df,datatype):
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(include='object').columns.to_list()
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')

    for f in features:
        concatenated = pd.concat([real_df[f].assign(dataset='Real'),syn_df[f].assign(dataset='Synthetic')])
        sns.violinplot(y=f,data=concatenated,style='dataset')
        plt.title("Violinplots real vs synthetic data for feature "+str(f)+"on dataset "+str(datatype))
        plt.savefig("violinplots_feature_"+str(f)+".pdf")

def create_ecdfplots(real_df,syn_df,datatype):
    features = real_df.columns.to_list()
    df_num = real_df.select_dtypes(exclude=[np.number])
    real_df[df_num] = real_df[df_num].astype('int')
    syn_df[df_num] = syn_df[df_num].astype('int')

    for f in features:
        concatenated = pd.concat([real_df[f].assign(dataset='Real'),syn_df[f].assign(dataset='Synthetic')])
        sns.ecdfplot(y=f,data=concatenated,style='dataset')
        plt.title("ECDF plots real vs synthetic data for feature "+str(f)+"on dataset "+str(datatype))
        plt.savefig("ecdfplots_feature_"+str(f)+".pdf")
    
def create_pairplot(df):
    features = df.columns.to_list()
    target = features[-1]
    features.pop(-1)

    g = sns.PairGrid(df, hue=target, vars=features)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()

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
