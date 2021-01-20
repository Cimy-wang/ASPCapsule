We have relased the code of ASPC nerual network, And the paper has been submitted to IEEE tim. 

 If you have any question, please contact me fell free!
 e-mail: wangjp85@mail2.sysu.edu.cn

******
Validate_set is is an optional parameter. If the value of Validate_set is set, the early_stop operation should be set as follows:
    callback = callbacks.EarlyStopping(monitor='val_acc',
                                       min_delta=0,
                                       patience=args.patience,
                                       verbose=1,
                                       mode='auto',
                                       restore_best_weights=True)

otherwise, it will be judged based on the training accuracy or training loss, and be set as:
    callback = callbacks.EarlyStopping(monitor='acc',
                                       min_delta=0,
                                       patience=args.patience,
                                       verbose=1,
                                       mode='auto',
                                       restore_best_weights=True)
******
If you want to run this code on the other datasets, please directly replace the ********.mat* file in the data folder.
******
