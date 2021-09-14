import seaborn as sns
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open("performance.json", "r") as fp:
    stats = json.load(fp)

optims = [
    'adam',
    'sgd',
    'rmsprop',
    ]


for opt in optims:
    loss_df_normal = pd.DataFrame([], columns=['loss', 'epoch'])
    acc_df_normal = pd.DataFrame([], columns=['accuracy', 'epoch'])
    time_df_normal = pd.DataFrame([], columns=['time', 'epoch'])
    loss_df_agc = pd.DataFrame([], columns=['loss', 'epoch'])
    acc_df_agc = pd.DataFrame([], columns=['accuracy', 'epoch'])
    time_df_agc = pd.DataFrame([], columns=['time', 'epoch'])
    for j in range(1,11):
        for k in range(0,10):
            # normal aggregation
            df2 = pd.DataFrame([[stats[opt][str(j)][k][0], k+1]], columns=['loss', 'epoch'])
            loss_df_normal = pd.concat([loss_df_normal, df2])
            df2 = pd.DataFrame([[stats[opt][str(j)][k][1], k+1]], columns=['accuracy', 'epoch'])
            acc_df_normal = pd.concat([acc_df_normal, df2])
            df2 = pd.DataFrame([[stats[opt][str(j)][k][2], k+1]], columns=['time', 'epoch'])
            time_df_normal = pd.concat([time_df_normal, df2])
            # agc aggregation
            df2 = pd.DataFrame([[stats[opt+'_agc'][str(j)][k][0], k+1]], columns=['loss', 'epoch'])
            loss_df_agc = pd.concat([loss_df_agc, df2])
            df2 = pd.DataFrame([[stats[opt+'_agc'][str(j)][k][1], k+1]], columns=['accuracy', 'epoch'])
            acc_df_agc = pd.concat([acc_df_agc, df2])
            df2 = pd.DataFrame([[stats[opt+'_agc'][str(j)][k][2], k+1]], columns=['time', 'epoch'])
            time_df_agc = pd.concat([time_df_agc, df2])

    sns.lineplot(data=loss_df_normal, x='epoch', y='loss', label=opt)
    sns.lineplot(data=loss_df_agc, x='epoch', y='loss', label=opt+'_agc')
    plt.savefig("loss_" + opt + '.png')
    plt.close()
    sns.lineplot(data=acc_df_normal, x='epoch', y='accuracy', label=opt)
    sns.lineplot(data=acc_df_agc, x='epoch', y='accuracy', label=opt+'_agc')
    plt.savefig("acc_" + opt + '.png')
    plt.close()
    sns.lineplot(data=time_df_normal, x='epoch', y='time', label=opt)
    sns.lineplot(data=time_df_agc, x='epoch', y='time', label=opt+'_agc')
    plt.savefig("time_" + opt + '.png')
    plt.close()
