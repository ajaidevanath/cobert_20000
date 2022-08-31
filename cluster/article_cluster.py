def article_cluster(query,df):
    similar = []
    query = model.embed_sentence(query)
    class_pred = knn.predict(query)
    #data = df_articles[df_articles['label']==class_pred].drop(['label'],axis=1)
    for i in range(0,len(df_articles)):
        if df_articles['label'][i] == class_pred:
            y = 1- distance.cosine(model.embed_sentence(preprocess_sentence(df['title'][i])),query)
            similar.append(y)
        else:
            similar.append(0)
    for i in range(len(similar)):
        if similar[i] == 1:
            similar[i] = 0
    return np.argmax(similar)
