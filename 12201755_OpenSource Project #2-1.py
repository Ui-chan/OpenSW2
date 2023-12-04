#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("2019_kbo_for_kaggle_v2.csv")


# In[64]:


years = [2015, 2016, 2017, 2018]
top_10_players = {}
k = 3
for year in years:
    year_data = data[data['year'] == year]
    
    top_10_hits = year_data.nlargest(10, 'H')
    top_10_players[f'{year}년 안타 Top 10'] = top_10_hits
    
    top_10_avg = year_data.nlargest(10, 'avg')
    top_10_players[f'{year}년 타율 Top 10'] = top_10_avg
    
    top_10_hr = year_data.nlargest(10, 'HR')
    top_10_players[f'{year}년 홈런 Top 10'] = top_10_hr
    
    top_10_obp = year_data.nlargest(10, 'OBP')
    top_10_players[f'{year}년 출루율 Top 10'] = top_10_obp

for key, value in top_10_players.items():
    print(f"{key}:")
    print("------------------")
    i = 1
    k+=1
    
    
    for _, player in value.iterrows():
        if k%4==0:
            print(f'{i}위',f"{player['batter_name']} H:{player['H']}")
        elif k%4==1:
            avg = round(player['avg'],3)
            avg2 = "{:.3f}".format(avg)
            print(f'{i}위',f"{player['batter_name']} avg:{avg2}")
        elif k%4==2:
            hr = int(player['HR'])
            print(f'{i}위',f"{player['batter_name']} HR:{hr}")
        elif k%4==3:
            obp = round(player['OBP'], 3)
            obp2 = "{:.3f}".format(obp)
            print(f'{i}위', f"{player['batter_name']} OBP:{obp2}")
            
        i = i + 1
        
    print()
    

# Years 배열에 각각의 2015,2016,2017,2018 저장. 
# Print top 10 players를 하기 위해 top_10_players의 딕셔너리 생성. 

# Years 배열에 있는 값들을 반복문을 통해 하나씩 꺼내 데이터에 접근 
# data[data[‘year’]]  == year일 경우 year_data에 해당 data 저장. 
# nlargest함수를 통해 ‘H’을 기준으로 오름차순 정렬하여 상위 10개 데이터에 접근하여 top_10_hits에 저장. 
# 마찬가지로 나머지 타율, 홈런 수, 출루율도 동일한 과정으로 각각의 변수들 top_10_avg,top_10_hr,top_10_obp에 저장.
# ⭐️ 딕셔너리 형태인 top_10_players의 Key값을 해당 {year}년 -- Top 10 으로 하고 그때의 value값을 위 해당 top_10_ 으로 저장한다. 

# 마지막으로 결과출력을 위해 for문으로 통해  top_10_players.items()을 통해 key, value 에 접근. 
# key값을 출력하고 value값은 많은 data가 들어있기에 다시 for문을 통해 접근. 여기서 index는 필요없어서 일부러 _, 로 설정하고 행 정보를 player에 저장하였다. 그리고 player의 이름을 가져오기 위해 player[‘batter_name’]을 하여 접근하였다. 
# + 순위를 출력하기위해 일부러 i = 1 로 초기화 하였고 for문이 진행될 때 마다 i를 출력하고 i = i + 1로 업데이트를 진행하였다. 
# k를 통해 출력예시를 조정.


# In[34]:


data_2018 = data[data['year'] == 2018]

positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
top_war_players = {}

for position in positions:
    position_data = data_2018[data_2018['cp'] == position]

    top_war_player = position_data.loc[position_data['war'].idxmax()]
    top_war_players[position] = top_war_player


for position, player in top_war_players.items():
    print(f"{position}: {player['batter_name']} (war: {player['war']})")

    
# data_2018에 data[data[‘year’] == 2018 ] 인 데이터들을 저장하였다. 
# Position info를 positions 배열에 저장. 
# 위와 비슷한 맥락으로 top_war_players 딕셔너리를 생성
# position 배열에 있는 값들에 하나 씩 반복하면서 해당 position과 data_2018[data_2018[‘cp’]==position]인 것들을 position_data에 넣는다.
# position_data[‘war’].idxmax()를 통해 war 열에서 가장 큰 값을 가지는 행의 인덱스를 반환한다. 그리고 top_war_player에는 position_data.loc[position_data['war'].idxmax()]는 위에서 찾은 인덱스를 사용하여 해당 행이 저장된다. 
# top_war_players[position]에 해당 정보를 저장한다. 

# 출력에서 top_war_players 딕셔너리를 for문을 사용해서 위 2-1과 동일한 방법으로 진행된다. 
# 2-1에서는 index(key)값이 필요하지 않기에 _를 사용하였지만 여기서는 key(position)을 출력해주기 위해 사용하였다.
# 출력에는 position과 player[‘batter_name’](선수이름) 과 player[‘war’](승리기여도)를 출력해주었다. 



# In[68]:


correlations = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr()

highest_salary_correlation = correlations['salary'].nlargest()

print(f"The highest correlation with salary (연봉) is : {highest_salary_correlation.index[1]}")
print(f"cor: {highest_salary_correlation[1]}")

# 'R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG’와 salary에 대한 correlation를 비교하기 위해 
# correlations에 data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']].corr() 를 저장. 
# 이렇게하면 각각에 대한 상관계수를 알 수 있다. 
# 하지만 우리는 여기서 salary에 대한 어떤 요소가 가장 correlation이 높은지를 알기 위해 highest_salary_correlation에 
# correlations['salary'].nlargest()를 하여 ‘salary’를 기준으로 오름차순 정렬을 하여 가장 높은 상위5개를 저장하였다.
# 그리고 여기서 주의할 점은 salary와 상관계수가 가장 높은건 salary 본인이므로 우린 두번째로 높은 값을 찾아야한다. 
# 그렇기에 출력에는 highest_salary_correlation.index[1]을 하여 salary를 제외한 가장 높은 correlation를 갖는 값을 출력하였다. 


# In[ ]:




