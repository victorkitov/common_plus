from pylab import *


def plot_hist(x, cutoff=2.5, bins=100):
    '''Plots histogram for x, cutting off outlier values.
    Input:
    x: array-like
    cutoff: how much percent of x distribution to cut from each side (to eliminate outliers)
    bins: count of bins in histogram
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    if type(x)!=np.ndarray:
        x=array(x)
    if type(cutoff)!=list: 
        cutoff = [cutoff, cutoff]
    
    x = x[(x>percentile(x,cutoff[0])) & (x<percentile(x,100-cutoff[1]))]
    hist(x,bins=bins);

    


def get_hist_diff(feature,inds0,inds1,bins_count=50):
    '''feature - pandas series. inds0, inds1 - selections of feature (e.g. for class1 and class2).
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    bins = np.linspace(feature.min(),feature.max(),bins_count)
    p0,_ = np.histogram(feature.loc[inds0].values, bins=bins, density=True)
    p1,_ = np.histogram(feature.loc[inds1].values, bins=bins, density=True)
    hist_diff = 0.5*np.sum(np.diff(bins)*np.abs(p1-p0))
    return hist_diff


def plot_hists(feature,inds0,inds1,feature_name='',label0='0',label1='1',bins_count=50):
    '''feature - pandas series. inds0, inds1 - selections of feature (e.g. for class1 and class2).
    Function plots pairs of histograms - for feature[inds0] and feature[inds0] to see the level of dependence between feature and type of selection (class)
    Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    
    
    plt.figure()
    COLORS = [[0,0.5,1],[1,0,0]]
    bins = np.linspace(feature.min(),feature.max(),bins_count)

    p0,_,_ = plt.hist(feature.loc[inds0].values, color=COLORS[0], bins=bins, alpha=0.5, normed=True,label=label0);
    p1,_,_ = plt.hist(feature.loc[inds1].values, color=COLORS[1], bins=bins, alpha=0.5, normed=True,label=label1);

    hist_diff = 0.5*np.sum(np.diff(bins)*np.abs(p1-p0))
    plt.title('{} - score: {:.3f}'.format(feature_name,hist_diff))
    plt.legend(loc='best')
    return hist_diff


    
def max_hist_diff(feature,y):
    '''Author: Victor Kitov (v.v.kitov@yandex.ru), 03.2016.'''    

    bins = np.linspace(feature.min(),feature.max(),50)
    y_unique = np.unique(y)
    pdf_list=[]
    for cls in y_unique:
        pdf,_ = np.histogram(feature[y==cls].values, bins=bins, density=True)
        pdf_list.append(pdf)

    max_hist_diff=0
    for i in range(len(pdf_list)-1):
        for j in range(i+1,len(pdf_list)):
            hist_diff = 0.5*np.sum(np.diff(bins)*np.abs(pdf_list[i]-pdf_list[j]))
            if hist_diff>max_hist_diff:
                max_hist_diff=hist_diff

    return max_hist_diff



'''
def plot_mult_hists(feature,inds_list,feature_name='',labels_list=None)
    if labels_list is None:
        labels_list=[str(i) for i in range(len(inds_list))]
    assert len(inds_list)==len(labels_list)
    plt.figure()
    COLORS=[[1,0,0],[0.2,1,0],[0,0.5,1],[1,0.5,0],[1,0,1],[0.5,0.5,0.5],[0.5,0,1]]
    bins = np.linspace(feature.min(),feature.max(),50)
    for inds in inds_list:
        p0,_,_=plt.hist(feature[inds].values, color=COLORS[0], bins=bins, alpha=0.5, normed=True,label=label0);
    p1,_,_=plt.hist(feature[inds1].values, color=COLORS[1], bins=bins, alpha=0.5, normed=True,label=label1);
    hist_diff = 0.5*np.sum(np.diff(bins)*np.abs(p1-p0))
    plt.title('{} - score: {:.3f}'.format(feature_name,hist_diff))
    plt.legend(loc='best')
    return hist_diff
'''