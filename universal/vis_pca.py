# -*- coding: utf-8 -*-
# code warrior: Barid
import seaborn as sns
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# tsne_result = load("/Users/barid/Documents/workspace/alpha/tmp/crossInit_n3000_trained.npy")
tsne_result = load("./pca.data.npy")

# tsne_result[:20000,-1] = 0
# tsne_result[40000:,-1] = 2
# import pdb;pdb.set_trace()
# tsne_result = tsne_result[np.random.randint(tsne_result.shape[0], size=20000), :]
np.random.shuffle(tsne_result)

tsne_result_df = pd.DataFrame({'pca_x': tsne_result[:,0], 'pca_y': tsne_result[:,1], 'label':tsne_result[:,-1]})
# bins = [float(0.1*i) for i in range(0,11)]
# bins_label = [str(0.1*i)[:3] for i in range(1,11)]
tsne_result_df["languages"] = pd.cut(tsne_result_df["label"], bins = 15,labels=['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru',  'th', 'tr', 'ur', 'vi', 'zh','sw'])
# tsne_result_df["languages"] = pd.cut(tsne_result_df["label"], bins = 7,labels=["en", "fr", "de", "ru", "zh", "sw", "ur"])
# tsne_result_df["languages"] = pd.cut(tsne_result_df["label"], bins = 3,labels=[ 'de','en','hi'])
ax = sns.scatterplot(x='pca_x', y='pca_y', hue='languages', data=tsne_result_df, markers=True,
    legend="full",s=12, style = "languages",
    alpha=0.2)
# plt.title("XLM+OURS$_{v2}$")
# plt.title("XLM")
plt.legend(loc='upper right',prop={'size': 15})
plt.show()
