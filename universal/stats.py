# -*- coding: utf-8 -*-
# code warrior: Barid
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
# XNLI= "ar bg de el en es fr hi ru sw th tr ur vi zh".split(" ")
XNLI = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es', 'et', 'eu', 'fa',  'fi', 'fr', 'fy', 'ga', 'gd', 'gl',  'gu', 'ha', 'he', 'hi',  'hr',  'hu', 'hy', 'id',  'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la',  'lo', 'lt', 'lv', 'mg', 'mk', 'ml', 'mn', 'mr', 'ms', 'my', 'ne', 'nl', 'no',  'om', 'or', 'pa', 'pl', 'ps', 'pt',  'ro', 'ru', 'sa',  'sd', 'si', 'sk', 'sl', 'so', 'sq', 'sr', 'su', 'sv', 'sw', 'ta',  'te', 'th', 'tl', 'tr', 'ug', 'uk', 'ur',  'uz', 'vi', 'xh', 'yi', 'zh',]
total = []
cross = []
bins = []
labels = []
for i in range(1,11):
    bins.append(np.log(10**i))
for i in range(1,10):
    labels.append("log(1e" + str(i)+")")
bins =   np.array(bins)
def pre_processing(path, name="name",mult=1):
    all_info = []
    sum_info = 0
    with open (path,'r') as data:
        for k,line in enumerate(data.readlines()):
            if k<50000:
                s,  freq = [i for i in line.strip().split(" ")]
                freq = np.array([freq])
                freq =freq.astype(np.float32)[0]
                all_info.append([k,freq,name])
                sum_info += freq
    return [[x[0],math.log((x[1]/sum_info)),x[2]] for x in all_info]

def showline_into_bucket(df_path, bins, labels, style, color):
    df = pre_processing(df_path)
    ax1 = plt.twinx()
    ax_1 = sns.lineplot(data=df,y='freq',color=color,ax=ax1)
    return ax_1
all_info = []
for lang in XNLI:
    tmp = pre_processing("./word_freq/" +lang +"freq.txt",lang)
    all_info  = all_info + tmp
# all = ax_1 + ax_2
df = pd.DataFrame(data=all_info, columns=["word rank", "log(freq)", "language"])
# df["freq"] = (df["freq"]-df["freq"].min())/(df["freq"].max()-df["freq"].min())
# df["freq"] = (df["freq"]-df["freq"].mean())/df["freq"].std()
ax_1 = sns.lineplot(data=df,x='word rank',y='log(freq)',hue="language",linewidth = 2.0,palette='bright')
# ax_1.set_yticks([10**i for i in range(10)])
# plt.title("Zipf curves for English and German Wikipedia dumps.")
plt.show()