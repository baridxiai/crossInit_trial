from sklearn.decomposition import PCA
from numpy import save
import numpy as np
def embSpace_genrator(embeddings):
    for k,em in enumerate(embeddings):
        l,_ = np.shape(em)
        if k == 0:
            language_list = np.ones([l]) *k
        else:
            language_list = np.concatenate([language_list, np.ones([l])*k],0)
    embeddings = np.concatenate(embeddings,0)
    # tsne = TSNE(2,learning_rate='auto',n_iter=10000,verbose=2,random_state=123,init='pca')
    # method="exact")
    pca = PCA(2,random_state=123)
    pca_data = pca.fit_transform(embeddings)
    save("./embSpace_pca.data", np.concatenate([pca_data,np.reshape(language_list,[-1,1])],axis=-1))