def doc_cluster(i,df):
    index = article_cluster(i)
    if df['body'][index] == '':
        return df['title'][index]
    else:
        body = df['body'][index].split('.')
        closest = []
        for items in body:
            d = 1-distance.cosine(model.embed_sentence(items),model.embed_sentence(i))
            closest.append(d)
        for i in range(len(closest)):
            if closest[i] == 1:
                closest[i] = 0
    return body[np.argmax(closest)] + body[np.argmax(closest)+1]