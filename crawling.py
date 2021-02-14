#%%
import pandas as pd 
import datetime

stock_code = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download', header=0)[0] 


# 데이터에서 정렬이 따로 필요하지는 않지만 테스트겸 Pandas sort_values를 이용하여 정렬을 시도해봅니다.
stock_code.sort_values(['상장일'], ascending=True)

# 필요한 것은 "회사명"과 "종목코드" 이므로 필요없는 column들은 제외
stock_code = stock_code[['회사명', '종목코드']] 

# 한글 컬럼명을 영어로 변경 
stock_code = stock_code.rename(columns={'회사명': 'company', '종목코드': 'code'}) 


# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌 
stock_code.code = stock_code.code.map('{:06d}'.format) 

#%%
#company='LG화학' 
for company in stock_code['company']:
    code = stock_code[stock_code.company==company].code.values[0].strip() ## strip() : 공백제거
    df = pd.DataFrame()
    for page in range(1,31):
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)     
        url = '{url}&page={page}'.format(url=url, page=page)
        print(url)
        df = df.append(pd.read_html(url, header=0)[0], ignore_index=True)


    df = df.dropna() 

    # 한글로 된 컬럼명을 영어로 바꿔줌 
    df = df.rename(columns= {'날짜': 'date', '종가': 'close', '전일비': 'diff', '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'}) 
    # 데이터의 타입을 int형으로 바꿔줌 
    df[['close', 'diff', 'open', 'high', 'low', 'volume']] = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int) 

    # 컬럼명 'date'의 타입을 date로 바꿔줌 
    df['date'] = pd.to_datetime(df['date']) 

    # 일자(date)를 기준으로 오름차순 정렬 
    df = df.sort_values(by=['date'], ascending=True) 

    df.to_csv('/home/ha/stock/data/' + str(company))
    
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(df['date'], df['close'])
plt.xlabel('')
plt.ylabel('close')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.savefig(company + ".png")
plt.show()