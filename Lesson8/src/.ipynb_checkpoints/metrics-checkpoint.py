import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    hit_rate = int(flags.sum() > 0)  
    
    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list,)   
    hit_rate = int(flags.sum() > 0)
    
    return hit_rate


def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    
    return precision


def miltimodel_precision_at_k(df, actual_column_name, starting_column, top_n):
    for col_name in df.columns[starting_column:]:
        yield col_name, df.apply(lambda row: precision_at_k(row[col_name], row[actual_column_name], k=top_n), axis=1).mean()
        

def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    
    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]
    prices_recommended = prices_recommended[:k]
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = (flags*prices_recommended).sum() / prices_recommended.sum()
     
    return precision


def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list[:k], bought_list)
    recall = flags.sum() / len(bought_list)
    
    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    prices_recommended = np.array(prices_recommended)
    prices_bought = np.array(prices_bought)
    
    flags = np.isin(recommended_list[:k], bought_list)
    recall = (flags * prices_recommended[:k]).sum() / prices_bought.sum()
    
    return recall


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(k):
        
        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            sum_ += p_k
            
    result = sum_ / k
    
    return result


def reciprocal_rank_at_k(recommended_list, bought_list, k=5):
    recommended_list = np.array(recommended_list)
    bought_list = np.array(bought_list)
    
    k = len(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    for u in range(k):
        if flags[u]:
            rr = 1 / (u + 1)
            break
    
    return rr


def ndcg_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    dcg_k = len(recommended_list[:k])
    ideal_k = len(bought_list)
    
    flags = np.isin(recommended_list[:k], bought_list)
    
    if sum(flags) == 0:
        return 0
    
    dcg = 0
    for j in range (dcg_k):
        if flags[j]:
            dcg += flags[j] / np.log2(j + 2)
    
    idcg = 0
    for j in range(ideal_k):
        idcg += 1 / np.log2(j + 2)
        
    ndcg = dcg / idcg
    
    return ndcg

