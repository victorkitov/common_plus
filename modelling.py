#!/usr/bin/env python
# encoding: utf-8


from pylab import *
import sklearn as skl
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from time import time
import types
import scipy
from common.iteration import piter
import pandas as pd


def out_of_sample_forecasts(clf, Z, train_sels=None, features=None, folds=4, random_state=0):
    '''
    Makes out of sample forecasts for training_sels (using cross-validation) and for testing_sels=~training_sels (using all training data)
    Returns forecast as numpy-array with len=len(Z).
    Z - dataframe, containing columns, specified by features and column "y", giving outputs for prediction.
    train_sels - boolean selector of training set. If omitted or set to None all objects, having not-nan y are treated as training objects.
    features - a list of features to use for prediction with classifier clf
    folds - with how many cross-validation folds to estimate y of the training set (y of testing set=~training_sels) is estimated using all
        objects from the training set.
    
    Output:
    (P, classes) tuple
    where P is NxC matrix of class probabilities (N=len(Z), C=len(unique(y)))
    classes is vector of class labels of length C
    
    '''

    X=Z[features].values
    y = Z.y.values
    
    if train_sels is None:
        train_sels = ~isnan(y)
        
    classes_count = len(unique(y))
    print('Making out-of-sample forecasts...')
    print('For training set... Folds:', end=' ')
    
    X_train = X[train_sels,:]
    y_train = y[train_sels]    
    p_train = np.zeros( (sum(train_sels),classes_count) )

    for fold_num,(cv_train_inds, cv_test_inds) in enumerate( skl.cross_validation.StratifiedKFold(y_train,folds,shuffle=True,random_state=random_state) ):
        print('%d'%fold_num, end=' ')
        clf.fit(X_train[cv_train_inds,:], y_train[cv_train_inds])
        p_train[cv_test_inds,:] = clf.predict_proba(X_train[cv_test_inds])
    print()
            
    p = np.zeros( (len(X), classes_count) )
    p[train_sels] = p_train

    test_sels = ~train_sels
    if sum(test_sels)>0:
        print('For testing set...')
        clf.fit(X[train_sels,:], y[train_sels])
        p[test_sels,:] = clf.predict_proba( X[test_sels,:] )

    print('Done')
    return p, clf.classes_

    
    
    
    


def fit_method(clf,param_grid,XX,yy,feature_lists=None,feature_list_names=None,
                             loss_fun='-acc',
                             folds_count=4, train_sels=None, test_sels=None,
                             top_results_count=5,
                             show_best_results=True, show_all_results=False,show_progress=True,
                             sample_weights=None,
                             random_state=0,repeat_count=1,stratified=False,stratify_values=None,normalize=False):

    assert len(XX)==len(yy)
    
    if isinstance(yy,pd.DataFrame):
        yy = yy.values.astype(np.float32)
        
    if isinstance(XX,pd.DataFrame):
        XX = XX.values.astype(np.float32)
        
    min_loss = np.inf

    if normalize is True:
        XX = (XX-mean(X))/std(XX)

    #assert all(isfinite(XX.ravel()))
    #assert all(isfinite(yy.ravel()))

    t0=time()

    params_list = list(ParameterGrid(param_grid))
    losses4params = []
    
    assert (train_sels is not None and test_sels is not None) or (train_sels is None and test_sels is None)
    if (train_sels is not None and test_sels is not None):
        assert repeat_count==1
        assert folds_count is None
        

    for params in params_list:
        clf.set_params(**params)
        losses=[]
        if show_progress:
            print('*',end='')

        for k in range(repeat_count):
            if folds_count is None:
                folds = [(find(train_sels), find(test_sels))]
            else: #  folds_count != None
                if stratified is True:
                    if stratify_values is None:
                        cc=yy
                    else:
                        cc=stratify_values                    
                    folds = skl.cross_validation.StratifiedKFold(cc,folds_count,shuffle=True,random_state=random_state+k)
                else:
                    folds = skl.cross_validation.KFold(len(yy),folds_count,shuffle=True,random_state=random_state+k)

            for train_inds, test_inds in folds:
                if sample_weights is None:
                    clf.fit(XX[train_inds,:], yy[train_inds])
                else:
                    clf.fit(XX[train_inds,:], yy[train_inds], sample_weight=sample_weights[train_inds] )

                if isinstance(loss_fun,types.FunctionType):
                    #y_hat = predict_by_batches(clf, XX[test_inds,:], clf.__class__.predict, batch_size=10000)
                    y_hat = clf.predict(XX[test_inds,:])
                    loss = loss_fun(yy[test_inds],y_hat,X_train_full.iloc[test_inds])
                else:
                    if loss_fun=='MSE':
                        #y_hat = predict_by_batches(clf, XX[test_inds,:], clf.__class__.predict, batch_size=10000)
                        y_hat = clf.predict(XX[test_inds,:])
                        loss = mean( (yy[test_inds]-y_hat)**2 )
                    elif loss_fun=='RMSE':
                        #y_hat = predict_by_batches(clf, XX[test_inds,:], clf.__class__.predict, batch_size=10000)
                        y_hat = clf.predict(XX[test_inds,:])
                        loss = sqrt( mean( (yy[test_inds]-y_hat)**2 ) )
                    elif loss_fun=='-acc':
                        #y_hat = predict_by_batches(clf, XX[test_inds,:], clf.__class__.predict, batch_size=10000)
                        y_hat = clf.predict(XX[test_inds,:])
                        loss = 1-skl.metrics.accuracy_score( yy[test_inds], y_hat )
                    elif loss_fun=='-AUC': # two classes only
                        #p1_hat = predict_by_batches(clf, XX[test_inds,:], predict_class1prob, batch_size=10000)
                        p_hat = clf.predict_proba(XX[test_inds,:])
                        class1ind = find(clf.classes_==1)[0]
                        p1_hat = p_hat[:,class1ind]
                        loss = 1-skl.metrics.roc_auc_score( yy[test_inds], p1_hat)
                    else:
                        raise ValueError("loss_fun must be '-acc' or '-AUC' or 'MSE'.")

                losses.append(loss)

        losses4params.append( np.mean(losses) )  # mean for all folds and repeat counts (folds_count*repeat_count in total)

    seq = [(params,loss) for (params,loss) in zip(params_list,losses4params)]
    sorted_seq = sorted(seq, key=lambda params_loss:params_loss[1])

    if show_progress:
        print()
    if show_best_results:
        if folds_count is None:
            print('Evaluated on validation set in %.02f min.' % ((time()-t0)/60))
        else: # K-fold cross-validation
            print('Evaluated %d times on %d folds in %.02f min.' % (repeat_count,folds_count,(time()-t0)/60))
        print('Min %i losses in decreasing order:' % top_results_count)
        for params,loss in sorted_seq[:top_results_count]:
            print('  %s %.04f'%(params,loss))
        print()
        
    if show_all_results:
        print()
        print('All losses as they were evaluated:')
        for params,loss in seq:
            print('  %s %.04f'%(params,loss))
        print()
        
    best_params,min_loss = sorted_seq[0]
    clf.set_params(**best_params)
    return clf, min_loss



def fit_method_cv_train_test(clf,param_grid,X_train_full,y_train,feature_lists=None,feature_list_names=None,
                             loss_fun='-acc',
                             folds_count=4, train_sels=None, test_sels=None,
                             top_results_count=5,
                             show_best_results=True, show_all_results=False,show_progress=True,
                             sample_weights=None,
                             random_state=0,repeat_count=1,stratified=False,strat_classes=None,normalize=False,
                             normalize_except=[]):

    assert len(X_train_full)==len(y_train)                             
                             
    if isinstance(y_train,pd.DataFrame):
        yy = y_train.values.astype(np.float)
    else:
        yy = y_train
        
    min_loss = np.inf
    if feature_lists is None:
        feature_lists = [list(X_train_full.columns.values)]
    all_features = set([f for feature_list in feature_lists for f in feature_list])

    if normalize is True:
        X_train_full = X_train_full.copy()
        for f in all_features:
            if f not in normalize_except:
                X_train_full[f] = (X_train_full[f]-X_train_full[f].mean())/X_train_full[f].std()

    for f in all_features:
        assert all(isfinite(X_train_full[f])),'Feature %s contains not numeric values!'%f

    if (feature_list_names is None) or len(feature_list_names)==0:
        feature_list_names = [str(feature_lists[0])]
        for i in range(1,len(feature_lists)):
            plus_features_list = list( set(feature_lists[i])-set(feature_lists[i-1]) )
            minus_features_list = list( set(feature_lists[i-1])-set(feature_lists[i]) )
            feature_list_name=''
            if plus_features_list:
                feature_list_name += ''.join(['+%s '%str(f) for f in plus_features_list])
            if minus_features_list:
                feature_list_name += ''.join(['-%s '%str(f) for f in minus_features_list])
            feature_list_names.append(feature_list_name)

        #feature_list_names = [str(feature_list)[:400] for feature_list in feature_lists]

    assert len(feature_lists)==len(feature_list_names)
    best_feature_list = []
    
    for feature_list,feature_list_name in zip(feature_lists,feature_list_names):
        if show_progress:
            if len(feature_list_name)<=300:
                print('\nFEATURES: %s' % feature_list_name)
            else:
                print('\nlen(FEATURES)=%d'%len(feature_list))
            print('Calculating: ',end='')

        XX = X_train_full[feature_list].values
        ###XX = XX.astype(np.double)
        
        #XX = scipy.sparse.lil_matrix( (len(X_train_full),len(feature_list)) ) 
        #for i,feature in enumerate(piter(feature_list)):
        #    XX[:,i] = X_train_full[feature].values[:,np.newaxis]
        

        if strat_classes is None:
            cc=yy
        else:
            cc=strat_classes.values.astype(np.double)

        t0=time()


        params_list = list(ParameterGrid(param_grid))
        losses4params = []
        
        assert (train_sels is not None and test_sels is not None) or (train_sels is None and test_sels is None)
        if (train_sels is not None and test_sels is not None):
            assert repeat_count==1
            assert folds_count is None
            

        for params in params_list:
            clf.set_params(**params)
            losses=[]
            if show_progress:
                print('*',end='')

            for k in range(repeat_count):
                if folds_count is None:
                    folds = [(find(train_sels), find(test_sels))]
                else: #  folds_count != None
                    if stratified is True:
                        folds = skl.cross_validation.StratifiedKFold(cc,folds_count,shuffle=True,random_state=random_state+k)
                    else:
                        folds = skl.cross_validation.KFold(len(yy),folds_count,shuffle=True,random_state=random_state+k)

                for train_inds, test_inds in folds:
                    if sample_weights is None:
                        clf.fit(XX[train_inds,:], yy[train_inds])
                    else:
                        clf.fit(XX[train_inds,:], yy[train_inds], sample_weight=sample_weights[train_inds] )

                    if isinstance(loss_fun,types.FunctionType):
                        #y_hat = predict_by_batches(clf, XX[test_inds,:], clf.__class__.predict, batch_size=10000)
                        y_hat = clf.predict(XX[test_inds,:])
                        loss = loss_fun(yy[test_inds],y_hat,X_train_full.iloc[test_inds])
                    else:
                        if loss_fun=='-acc':
                            #y_hat = predict_by_batches(clf, XX[test_inds,:], clf.__class__.predict, batch_size=10000)
                            y_hat = clf.predict(XX[test_inds,:])
                            loss = 1-skl.metrics.accuracy_score( yy[test_inds], y_hat )
                        elif loss_fun=='-AUC': # two classes only
                            #p1_hat = predict_by_batches(clf, XX[test_inds,:], predict_class1prob, batch_size=10000)
                            p_hat = clf.predict_proba(XX[test_inds,:])
                            class1ind = find(clf.classes_==1)[0]
                            p1_hat = p_hat[:,class1ind]
                            loss = 1-skl.metrics.roc_auc_score( yy[test_inds], p1_hat)
                        else:
                            raise ValueError("loss_fun must be '-acc' or '-AUC'.")

                    losses.append(loss)

            losses4params.append( np.mean(losses) )  # mean for all folds and repeat counts (folds_count*repeat_count in total)

        seq = [(params,loss) for (params,loss) in zip(params_list,losses4params)]
        sorted_seq = sorted(seq, key=lambda params_loss:params_loss[1])

        if show_progress:
            print()
        if show_best_results:
            if folds_count is None:
                print('Evaluated validation set in %.02f min.' % ((time()-t0)/60))
            else: # K-fold cross-validation
                print('Evaluated %d times on %d folds in %.02f min.' % (repeat_count,folds_count,(time()-t0)/60))
            print('Min %i losses in decreasing order:' % top_results_count)
            for params,loss in sorted_seq[:top_results_count]:
                print('  %s %.04f'%(params,loss))
            print()

        if show_all_results:
            print()
            print('All losses as they were evaluated:')
            for params,loss in seq:
                print('  %s %.04f'%(params,loss))
            print()

        if sorted_seq[0][1] < min_loss:
            best_params, min_loss = sorted_seq[0]
            best_feature_list = feature_list

    if show_all_results:
        if len(feature_lists)>=2 or (show_best_results==False):
            print('\n---------------------------------')
            print('BEST FEATURES: %s\n' % str(best_feature_list))
            print('BEST PARAMS: %s\n' % str(best_params))
            print('BEST SCORE: %s%.4f%s\n' % (RED_COLOR,min_loss,BLACK_COLOR))

    clf.set_params(**best_params)
    return clf, min_loss



def evaluate_feature_collections(clf,param_grid,X_train_full,y_train,experiments,evaluation_params={}):
    '''Run model evaluations for a list of feature sets, stoed in list "experiments"
    Each entry is a Struct with fields:
        experiment.name
        experiment.feature_lists
    '''
    losses = []
    for experiment in experiments:
        print('\n'+'='*50)
        print('FEATURE COLLECTION: %s\n'% experiment.name)
        clf,loss = fit_method_cv_train_test(clf, param_grid, X_train_full, y_train, experiment.feature_lists,
                                            **evaluation_params)
        losses.append(loss)

    if len(experiments)>=2:
        inds = argsort(losses)
        print('\n'+'#'*80)
        print('TOP 5 FEATURE COLLECTIONS AND THEIR LOSSES:')
        for ind in inds[:5]:
            print('%30s: %s%.4f%s' % (experiments[ind].name,RED_COLOR,losses[ind],BLACK_COLOR) )



def predict_class1prob(clf, X):
    p_hat = clf.predict_proba(X)
    class1ind = find(clf.classes_==1)[0]
    return p_hat[:,class1ind]


def show_detailed_accuracy_report(y_hat,X_pd,y_pd,sample_weight=None, best_method_report=True,
                                  show_accuracy_for_disc_features=True, max_disc_values=2):
    '''
    Shows detailed accuracy report for preductions y_hat for true values y_pd (pandas series)
    and corresponding feature matrix X_pd (pandas dataframe).
    best_method_report - show confusion matrix in graphical form
    show_accuracy_for_disc_features - show accuracies given each value of discrete features
    (features are discrete if they take less than max_disc_values unique values)
    '''

    X=X_pd.values
    y=y_pd.values

    if best_method_report is True:  # show best_method_report - classification report, confusion_matrix
        print('\nBest method classification report:')

        print(skl.metrics.classification_report(y, y_hat))

        classes = np.unique(y)
        cm=confusion_matrix(y, y_hat)
        matshow(cm)
        #xticks(classes-1,classes+1)
        #yticks(classes-1,classes+1)
        plt.title('Confusion matrix (absolute)')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm2=confusion_matrix(y, y_hat).astype(float)
        np.fill_diagonal(cm2,np.nan)
        matshow(cm2)
        #xticks(classes-1,classes+1)
        #yticks(classes-1,classes+1)
        plt.title('Confusion matrix (absolute without diagonal)\n')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    if show_accuracy_for_disc_features: # for every discrete feature, taking <= max_disc_values unique values show accuracy of best method
        print('\nBest method classification accuracy given fixed discrete features:')
        binary_features_values = [(feature,X_pd[feature].unique()) for feature in X_pd.columns if len(X_pd[feature].unique())<=max_disc_values]
        for feature, values in binary_features_values:
            for value in sorted(values):
                sels=(X_pd[feature]==value).values
                accuracy = accuracy_score(y[sels], y_hat[sels])
                print('{}={}: accuracy={:.04f} ({} entries)'.format(feature,value,accuracy,sum(sels)))
