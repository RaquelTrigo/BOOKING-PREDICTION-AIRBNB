

def LifgtGBM(data):

    Y = data.country_destination
    
    data.drop(["country_destination"],axis=1,inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.20, random_state=15)
    
    lgb_model = lgb.LGBMClassifier()
    
    lgb_model.get_params()
    
    gridParams = {
          'learning_rate': [0.08, 0.1,0.3],
          'boosting_type' : ['gbdt'],
          'objective' : ['binary'],
          'subsample' : [0.75],
          'max_depth': [6],
          'num_leaves': [70]
        }
    
    grid_solver = GridSearchCV(estimator = lgb_model, 
                       param_grid = gridParams,
                       cv = 5,
                       verbose = 2)
    
    lgb_model = grid_solver.fit(X_train, y_train)
    
    lgb_model.best_params_
    
    y_pred_train= lgb_model.predict(X_train)
    
    y_pred= lgb_model.predict(X_test)

    f1score_train = f1_score(y_train, y_pred_train)
    print('F1_score_train:')
    print(f1score_train) #0.625604117726273 Train
    
    
    f1score_test = f1_score(y_test, y_pred)
    print('F1_score_test:')
    print(f1score_test) #0.6200682512123571
    
    return f1score_train, f1score_test









