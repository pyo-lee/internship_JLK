import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def find_recommend(csv, input_data):
    
    df = pd.read_csv('./{}.csv'.format(csv), index_col=0)

    data_input = pd.DataFrame(0,index=['input'],columns=list(df.keys()))
    input_data = input_data.split(',')
    print('keyword :', input_data)
    
    for key_input in input_data:
        if not key_input in df.keys():
            key_ = False
            break
        else:
            key_=True
    
    result_li = list()
    if key_:
        for c in input_data:
            data_input.iloc[0][c]=1

        sim_max, sim_max_index = 0,0

        result_li = list()
        for i in range(len(df)):

            similarity = cosine_similarity(data_input.values, [df.iloc[i].values])[0][0]
            if similarity>0:
                result_li.append([df.iloc[i].name, similarity])

        result=pd.DataFrame(result_li, columns=['place','similarity']).set_index('place')
        result_top5 = result.sort_values(by=['similarity'],ascending=False)[:5]
        
        print('-'*20)
        for r in range(len(result_top5)):
            print('추천 매장명 : ', result_top5.iloc[r].name)
            print('유사도 : {0:0.4f}'.format(result_top5.iloc[r]['similarity']))
        print('-'*20)
        
    else:
        print('keyword가 옳지 않습니다')
    
    
# python pet_recommend.py --csv 파일명 --input_data keyword
# python pet_recommend.py --csv vector --input_data 양양,글램핑,대전강아지동반
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default='vector')
    parser.add_argument("--input_data", type=str, default='')

    args = parser.parse_args()
    find_recommend(**args.__dict__)
    