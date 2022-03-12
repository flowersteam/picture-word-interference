def save_csv_four(REF,O,ON,N,pred_category,wa_category,testtCategory):
    import pandas as pd
    df = pd.DataFrame(REF,
                  columns=['unswitch'])
    df['labelswitched'] = O
    df['original_label'] = ON
    df['new_label'] = N
    df.to_csv(path+testCategory+pred_category+wa_category+'.csv')

def save_csv_two(REF,O,pred_category,wa_category,testtCategory):
    import pandas as pd
    df = pd.DataFrame(REF,
                  columns=['unswitch'])
    df['labelswitched'] = O
    df.to_csv(path+testCategory+pred_category+wa_category+'.csv')