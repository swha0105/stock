#%%

#%%
import math
import numpy as np 

w = 12
h = 8

ratio = w/h
n = 1
answer_count = 0
x_position = 0
y_position = 0


while True:

    if ratio*n - x_position < 1:
        answer_count = answer_count + 1
        y_position = y_position + 1
        
    elif ratio*n - x_position >1:
        answer_count = answer_count + 2
        x_position = x_position + 1
        y_position = y_position + 1
        
    elif int(ratio*n) - x_position == 1:
        answer_count = answer_count + 1
        x_position = x_position + 1
        y_position = y_position + 1
        
    if y_position >= h or x_position >= w: 
        answer = w*h - answer_count

        break
    else:
        n = n+1
    

print(answer)

#%%

# #%%
# import copy
# import numpy as np

# progresses = np.array([95, 90, 99, 99, 89, 99]	)
# speeds = np.array([1, 1, 1, 1, 1, 1])

# #%%

# def solution(progresses, speeds):
#     Q=[]
#     for p, s in zip(progresses, speeds):
#         tmp = (p-100)//s ## 남은 second 
#         if len(Q)==0 or Q[-1][0]<-(tmp):
#             Q.append([-tmp,1])
#         else:
#             Q[-1][1]+=1
#     return [q[1] for q in Q]

# ans = solution(progresses,speeds)
# #%%
# import copy
# import numpy as np

# progresses = np.array([95, 90, 99, 99, 89, 99]	)
# speeds = np.array([1, 1, 1, 1, 1, 1])
# n = 0

# ans = []

# def decision(input_array,speeds,n,ans):
#     n = n + 1

#     input_array = np.array(input_array)
#     speeds = np.array(speeds)
#     ref = np.sum([input_array,n*speeds],axis=0)
  
#     index_candidate = ref>=100
#     index = []
       
#     i = 0

#     if len(index_candidate) != 0:
                
#         while index_candidate[i] == True :
#             index.append(i)
#             i = i+1
#             if i >= len(index_candidate):
#                 break

#     if len(index) != 0:
#         ans.append(len(index))
        
            

#     input_array = np.delete(input_array,index)
#     speeds = np.delete(speeds,index)

#     if input_array.shape[0] == 0:
#         return ans
#     else:
#         return decision(input_array,speeds,n,ans)

# answer = decision(progresses,speeds,n,ans)


# #%%
# while True:

#     ref = np.sum([progresses,n*speeds],axis=0)
#     ref = np.array(ref)
    
#     finished = np.array([])
    
#     index = np.argwhere(ref>100)

#     if index.shape[0] !=0 :
#         for i in index:
#             if all(ref[:int(i)] > 100) == True:
#                 print("hi",i)
#                 progresses = np.delete(progresses,int(i))
#                 speeds = np.delete(speeds,int(i))
                
    
#                 finished = i
    
#     if finished.shape[0] != 0:
#         answer.append(finished.shape[0])
        
#     n = n + 1

#     if len(progresses) == 0:
#         break
    
# #     for speed in speeds:
# #         progresses
    
    
    

# %%
import numpy as np
w = 8
h = 12

def GCD(a,b):

    r = b%a

    if r == 0:
        return a
    else:
        return GCD(r,a)

GCD_factor = int(GCD(w,h))
w_box = int(w/GCD_factor)
h_box = int(h/GCD_factor)
box_count = 0

if w_box/h_box - float(int(w_box/h_box)) != 0.0: 
    box_count = w_box + h_box -1 
else:
    box_count = w_box


# if w_box/h_box - float(int(w_box/h_box)) != 0.0:
#     box_count = box_count + 1

# if w_box/h_box >= 1:
#     box_count = box_count + int(w_box)
# else:
#     box_count = box_count + int(h_box)

# if w_box/h_box == int(w_box/h_box):
#     box_count = np.max([h_box,w_box])
# else:
#     box_count = np.max([h_box,w_box]) + 1
    
answer = w*h - box_count * GCD_factor 
print(answer)
#%%
import numpy as np
skill = "CBD"
skill_trees = ["BACDE", "CBADF", "AECB", "BDA","CBD"]
#skill_trees = ["AECB"]

answer = 0

#걸러내기


for ref_skill in skill_trees:
    count = 1

    list_in_skill = []

    for skill_name in ref_skill:
        if skill_name in skill:
            list_in_skill.append(skill_name)

    for n,tmp in enumerate(list_in_skill):
        if skill[n] != tmp:
            count = 0
            break

    
    
    if count == 1:
        
        answer = answer + 1
        

    #answer = answer + determine(ref_skill,skill)
    # for skill_name in skill:
    #     if skill_name in no_list:
    #         break
    #     elif skill_name in ref_skill:
    #         index = ref_skill.index(skill_name)
    #     no_list = ref_skill[index+1:]
    
print(answer)
#    answer = answer+1
        
    

#%%

import numpy as np
skill = "CBD"
skill_trees = ["BACDE", "CBADF", "AECB", "BDA","CBD"]
#skill_trees = ["AECB"]

answer = 0

for skills in skill_trees:
    skill_list = list(skill)

    for s in skills:
        if s in skill:
            if s != skill_list.pop(0):
                break
    else:
        answer += 1
#%%
prices = [1, 2, 3, 2, 3]
answer = [ ]


while len(prices) != 1:
    ref_prices = prices.pop(0)

    if ref_prices <= min(prices):
        answer.append(len(prices))
    else:
        for n,com_prices in enumerate(prices):
            if ref_prices > com_prices:
                answer.append(n+1) 
                break
   
answer.append(0)            
    
print(answer)
#%%  
    while len(prices) != 0:
        ref_prices = prices.pop(0)

        for n,com_prices in enumerate(prices):
            if ref_prices > com_prices:
                answer.append(n+1) 
                break
        else:
            answer.append(len(prices))

#%%
# bridge_length  = 2 
# weight = 10	
# truck_weights = [7,4,5,6]

bridge_length = 100
weight = 100
truck_weights = [10,10,10,10,10,10,10,10,10,10]

end = len(truck_weights)
answer = 0
time = 0
truck_passing_weight = []
truck_passing_time = []
truck_passed = []


while len(truck_passed) != end:

    
    if len(truck_passing_time) == 0:
        pass
    
    elif time - truck_passing_time[0] >= bridge_length:
        truck_passed.append(truck_passing_weight.pop(0))
        truck_passing_time.pop(0)
    

    if len(truck_weights) == 0:
        pass
    
    elif sum(truck_passing_weight) + truck_weights[0] <= weight:
        truck = truck_weights.pop(0)
        truck_passing_weight.append(truck)
        truck_passing_time.append(time)

    
    
    time = time + 1




print(time)
        


#%%

import collections

DUMMY_TRUCK = 0


class Bridge(object):

    def __init__(self, length, weight):
        self._max_length = length
        self._max_weight = weight
        self._queue = collections.deque()
        self._current_weight = 0

    def push(self, truck):
        next_weight = self._current_weight + truck
        if next_weight <= self._max_weight and len(self._queue) < self._max_length:
            self._queue.append(truck)
            self._current_weight = next_weight
            return True
        else:
            return False

    def pop(self):
        item = self._queue.popleft()
        self._current_weight -= item
        return item

    def __len__(self):
        return len(self._queue)

    def __repr__(self):
        return 'Bridge({}/{} : [{}])'.format(self._current_weight, self._max_weight, list(self._queue))


def solution(bridge_length, weight, truck_weights):
    bridge = Bridge(bridge_length, weight)
    trucks = collections.deque(w for w in truck_weights)

    for _ in range(bridge_length):
        bridge.push(DUMMY_TRUCK)
    
    
    count = 0
    while trucks:
        bridge.pop()

        if bridge.push(trucks[0]):
            trucks.popleft()
        else:
            bridge.push(DUMMY_TRUCK)

        count += 1

    while bridge:
        bridge.pop()
        count += 1

    return count


def main():
    print(solution(2, 10, [7, 4, 5, 6]), 8)
    print(solution(100, 100, [10]), 101)
    print(solution(100, 100, [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]), 110)


if __name__ == '__main__':
    main()
#%%

def decision(prior,index,count,location):

    ref_value = prior[0]
    ref_index = index[0]
    print(prior,index)
    if ref_value < max(prior):
        prior.append(prior.popleft())
        index.append(index.popleft())
        return decision(prior,index,count,location)
    
    elif ref_index == location:
        count = count + 1
        return count 
    
    else:     
        count = count + 1
        prior.popleft()
        index.popleft()
        return decision(prior,index,count,location)



#%%
import collections

priorities	= [1, 1, 9, 1, 1, 1]
prior = collections.deque(priorities)

index = collections.deque(list(range(len(priorities))))
location = 0
count = 0
count = decision(prior,index,count,location)

print(count)

#%%
priorities	= [1, 1, 9, 1, 1, 1]

queue =  [(i,p) for i,p in enumerate(priorities)]
answer = 0
while True:
    cur = queue.pop(0)
    if any(cur[1] < q[1] for q in queue):
        queue.append(cur)
    else:
        answer += 1
        if cur[0] == location:
            break
print(answer)

#%%



s = "aabbaccc"	



pre_length = len(s)
answer = [pre_length,1]

for i in range(1,int(len(s)/2)+1):
    pre_word = s[:i]
    remain_string = s[i:]
    
    n = 0
    for n in range(i):
        count = 0
        pre_length = len(s)
        remain_length = len(s)
        same_word = 0 
        while len(remain_string) != 0:
            post_word = remain_string[n:i+n]
            
            if pre_word == post_word:
                count = count + 1 
            
            if pre_word == post_word and pre_word == remain_string[i+n:2*i+n]:
                same_word = same_word + 1

            remain_length = len(s) - count *(i-1) - same_word
            

            if remain_length < pre_length:
                answer = [remain_length,i]
                pre_length = remain_length 

            remain_string = remain_string[n+i:]
            pre_word = post_word

print(answer)

#%%


s = "abcabcabcabcdededededede"	
import numpy as np 
#s = "ababcdcdababcdcd"
pre_compress_word = 0
answer = len(s)
for i in range(1,int(len(s)/2)+1):
    count = 0 
    remain_length = len(s)
    
    compress_word = 0
   
    for n in range(0,len(s)-i,i):
        
        pre_word = s[n:n+i]
        next_word = s[n+i:n+2*i]
        
        
        if pre_word == next_word:
            count = count + 1
            continue            
        elif pre_word != next_word and count !=0:
            count = count + 1
            compress_word = compress_word + count*len(pre_word) - ( (int(count/10) + 1)  + len(pre_word))
            count = 0 
        
    if pre_word == next_word: 
        count = count + 1
        compress_word = compress_word + count*len(pre_word) - ( (int(np.log10(count)) + 1)  + len(pre_word))

    
    if compress_word > pre_compress_word:
        
        remain_length = remain_length - compress_word
        pre_compress_word = compress_word
        answer = remain_length
print(answer)

#%%

def compress(text, tok_len):

    words = [text[i:i+tok_len] for i in range(0, len(text), tok_len)]
    #print(words)
    res = []
    cur_word = words[0]
    cur_cnt = 1
    for a, b in zip(words, words[1:] + ['']):
        #print(words, words[1:] + [''])
        if a == b:
            cur_cnt += 1
        else:
            res.append([cur_word, cur_cnt])
            cur_word = b
            cur_cnt = 1
    print(res)
    return sum(len(word) + (len(str(cnt)) if cnt > 1 else 0) for word, cnt in res)

def solution(text):
    return min(compress(text, tok_len) for tok_len in list(range(1, int(len(text)/2) + 1)) + [len(text)])

a = [
    "aabbaccc",
    "ababcdcdababcdcd",
    "abcabcdede",
    "abcabcabcabcdededededede",
    "xababcdcdababcdcd",

    'aaaaaa',
]

for x in a:
    print(solution(x))


#%%

#import collocetion

prices = [1, 2, 3, 2, 3]
answer_tmp = []
concave = 0
answer = []

min_price = prices[0]

while prices:
    cur_price = prices.pop(0)
    if cur_price > min_price:
        min_price = cur_price
        n = 0
 #       answer_tmp = []
    else:
        answer_tmp.append(0)
        answer_tmp = [x+ for x in answer_tmp]
        
    #answer = answer_tmp 
        print(answer_tmp)

#print(answer)

#%%


for price in prices:
    if min_prices > price:
        min_prices = price
        answer.append(answer_tmp)
        break
    else:
        answer_tmp.append(0)
        answer_tmp = [x+1 for x in answer_tmp]
        print(answer_tmp)

else:
    answer.append(len(prices))

# if answer_tmp != [0]:
answer = answer + answer_tmp


#print(answer)



#%%

num_list = []
def make_number(number_list,num_list):

    last_num = number_list.pop(0)

    if len(number_list) == 1:
        num_list = num_list + [last_num]
        return num_list
    
    for add_num in make_number(number_list,num_list):
        num_list = num_list + [add_num + last_num]

    



numbers = "012"
numbers = list(numbers)

a = make_number(numbers,num_list)
print(a)
#%%

num_list = []
n = 2 



last_num = numbers.pop(0)

for add_num in numbers:
    
    num_list = num_list + [add_num + last_num]

last_num = numbers.pop(0)

for n,add_num in enumerate(numbers):
    num_list = [num_list[n]] + [add_num + last_num]    
# %%


a= [1,2]
# %%
scoville = [1, 2, 3, 9, 10, 12]
K = 7
n = 0
scoville_list = []

def func(scoville,K,n):

    n = n+1
        

    min_scoville = scoville.pop(scoville.index(min(scoville)))
    new_scoville = min_scoville + 2 * scoville.pop(scoville.index(min(scoville)))

    scoville.insert(0,new_scoville)
    
    print(scoville,K)
    if min(scoville) >= K :
        
        answer = n 
        return answer
    elif len(scoville) == 1:
        return - 1
    else:
        return func(scoville,K,n)

    
a = func(scoville,K,n)
print(a)

#%%
import numpy as np

scoville = [1, 2, 3, 9, 10, 12]
K = 7
n = 0
def func(scoville,K,n):
    n = n+1
    scoville = np.array(scoville)
    scoville = sorted(scoville[scoville<K]
    
    min_scoville = scoville.pop(0)
    
    new_scoville = min_scoville + 2 * scoville.pop(0)
    scoville.insert(0,new_scoville)
    
    
    if all(K <= x for x in scoville):
        answer = n 
    
        return answer

    else:
        return func(scoville,K,n)

    
a = func(scoville,K,n)
print(a)

#%%

import numpy as np 

A = np.array([[1,1],[2,1],[3,1],[4,1],[5,1]])

A_inverse = np.linalg.pinv(A)
B = np.array([5,11,16,21,27])

mat = np.dot(A_inverse,B)

#%%
import numpy as np 

A = np.array([[1,1,1],[2,1,1],[3,1,1],[4,1,1],[5,1,1]])

A_inverse = np.linalg.pinv(A)


B = np.array([5,11,16,21,27])

mat = np.dot(A_inverse,B)




# %%

for i in range(0,4):
    print(B[i]- A[i,0] * mat[0] - mat[1] - mat[2] ,i)
#%%

-6*mat[0] - mat[1] - mat[2]
#%%

import numpy as np
num = 235386
answer = 0

def num_sorting(num_ref):
    num_str = str(num_ref)
    num_len = len(num_str)

    if num_len % 2 == 0:
        return num_ref
      
    elif num_len % 2 != 0:
        num_ref = 10**num_len
        return num_ref    

    return num_ref
    



while True:
    
    if len(str(num)) % 2 == 0 :
        front = 1
        back = 1
        for i in range(int((np.log10(num)+1)/2)):

            front = front*int(str(num)[i])
            back = back*int(str(num)[-i-1])

        if front == back:
            answer = num
            break
        else:
            num = num + 1
    else:
        num = num_sorting(num)


print(answer)

#%%
for i in range(int(len(str(number))/2)):


#%%

def solution(num):

    import numpy as np
    answer = 0
    
    def num_sorting(num_ref):
        num_str = str(num_ref)
        num_len = len(num_str)

        if num_len % 2 != 0:
            num_ref = 10**num_len

            return num_ref

        if int(num_str[-1]) % 2 == 1:
            num_ref = num_ref + 1
            return num_ref

        return num_ref


    while True:

        num = num_sorting(num)

        front = 1
        back = 1

        for i in range(int((np.log10(num)+1)/2)):

            front = front*int(str(num)[i])
            back = back*int(str(num)[-i-1])

        if front == back:
            answer = num
            break
        else:
            num = num + 2
    
    return answer

#%%

#arr1 = ["()", "(()", ")()", "()"]
#arr2 = [")()", "()", "(()"]
# arr1 = ["()", "(()", "(", "(())"]
# arr2 = [")()","()", "(())", ")()"]

arr1 = ["(","(()(",")" ,"()"]
arr2 = ["))","(",")"]

arr1_sorting = [ ]
arr2_sorting = [ ]

for arr in arr1:
    arr_decision = list(arr)
    count = 0

    for decision in arr_decision:
        
        if decision == "(":
            count = count + 1
        elif decision == ")":
            count = count -1
    
    if count >=0:
        #print(arr,count)
        arr1_sorting.append(count)



for arr in arr2:
    arr_decision = list(arr)
    count = 0
   

    for decision in arr_decision:
        
        if decision == "(":
            count = count + 1
        elif decision == ")":
            count = count -1
    
    if count <=0:
        # print(arr,count)
        arr2_sorting.append(count)



answer = 0
for count_1 in arr1_sorting:
    for count_2 in arr2_sorting:
        print(count_1,count_2, count_1 + count_2 )
        if count_1 + count_2 == 0:
            answer = answer + 1

print(answer)

#%%

bat = [[6,30000],[3,18000],[4,28000],[1,9500]]
N = 20


one_prices = []

for a in bat:
    one_prices.append(a[1]/a[0])


buy_n =0 
money = 0

min_index = one_prices.index(min(one_prices))

while True: 
    if buy_n + bat[min_index][0] > N:
        break
    else:
        money = money + bat[min_index][1]
        buy_n = buy_n + bat[min_index][0]

i = 1
ref_cost = float('inf')
for num,price in bat:
    while True:
        if num*i < N-buy_n:
            i = i + 1
        else:
            break
    
    ref_cost = min([i*price,ref_cost])
    
answer = money + ref_cost
print(answer)

#%%

battery = [[6,30000],[3,18000],[4,28000],[1,9500]]	
n = 20


one_prices = []

for a in battery:
    one_prices.append(a[1]/a[0])


buy_n =0 
money = 0

min_index = one_prices.index(min(one_prices))

while True: 
    if buy_n + battery[min_index][0] >= n:
        break
    else:
        money = money + battery[min_index][1]
        buy_n = buy_n + battery[min_index][0]

#%%

#%%


def iteration(battrey,buy_n,money,min_index):
    #print(buy_n,min_index)
    if buy_n >=n:
        return money

    if buy_n + battery[min_index][0] > n:
        ref_cost =  battery[min_index][1]
        

        for ix,(num,price) in enumerate(battery):
            i = 1
            while True:
                if num*i < n-buy_n:
                    i = i + 1
                else:
                    break
            
            print(battery,min_index,i*price,money)
            
            
            if i*price < ref_cost:
                min_index = ix    
                ref_cost = i*price
                
        
#        print(i*price,ref_cost,min_index)

        money = money + battery[min_index][1]
        buy_n = buy_n + battrey[min_index][0]

        return iteration(battery,buy_n,money,min_index)

    else:
        money = money + battery[min_index][1]
        buy_n = buy_n + battrey[min_index][0]
        
        return iteration(battery,buy_n,money,min_index)



battery = 	[[6,30000],[3,18000],[4,28000],[1,9500]]	
n = 20
prices = [ ]

for _,b in battery:
    prices.append(b)

one_prices = []

for a in battery:
    one_prices.append(a[1]/a[0])


min_index = one_prices.index(min(one_prices))
money = 0
iteration(battery,0,money,min_index)

#%%
for ix,(a,b) in enumerate(battery):
    print(ix,a,b)

#%%
min_index = one_prices.index(min(one_prices))

while buy_n + battery[min_index][0] >= n: 
    
    if buy_n + battery[min_index][0] >= n:
        break
    else:
        money = money + battery[min_index][1]
        buy_n = buy_n + battery[min_index][0]


#%%

if buy_n == n:
    answer = money
else:
    ref_cost = float('inf')
    for num,price in battery:
        i = 1
        while True:
            if num*i < n-buy_n:
                i = i + 1
            else:
                break

        ref_cost = min([i*price,ref_cost])
        print(ref_cost,i)
    answer = money + ref_cost
print(answer)

#%%
A = [ [1,2,3], [4,5,6], [7,8,9] ]
answer = []
for x in zip(*A):
    answer.append(x)
print(answer)
#%%
mylist = ['1','100','33']
a = ''.join(mylist)
#a = list(map(int,mylist))
print(a)
answer = []
# for i in mylist:
#     answer = answer + i
# int(answer)
#%%

n = int(input().strip())
for i in range(n):
    print('*'*(i+1))

#%%

import collections
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 7, 9, 1, 2, 3, 3, 5, 2, 6, 8, 9, 0, 1, 1, 4, 7, 0]
answer = collections.Counter(my_list)
print(answer[1])

# %%
A = [3,2 ,6,7]

A%2

#%%
def test(A):
    return a**2 if b % 2 == 0 for b in A

print(test(A))
 [ i**2 for i in mylist if i %2 == 0]

#%%

#a = int(input())

array = [5,1,2,3,1]
import math
num = 1 
for number in array:
    num = num * number
    print(number,num)

    if math.sqrt(num) - int(math.sqrt(num)) == 0:
        print("found")
        break
    
else:
    print("not found")
#%%
import collections

participant	 = ["mislav", "stanko", "mislav", "ana"]	
completion = ["stanko",  "mislav"]	

p = collections.Counter(participant)
c = collections.Counter(completion)

#p.subtract(c)

# for name in p:
    
    # if p[str(name)] >=1 :
 #       print(name)



list( (p-c).keys() )[0]
# %%
array = ["12","123","1235","567","88"]	
array = sorted(array,key=len)
answer = []
while array != []:
    ref = array.pop(0)
    phone_number_list = [s for s in array if ref in s]


    answer.append(phone_number_list)

#answer = set(sum(answer,[]))
print(answer)

if len(answer) == 0:
    print("True")
else:
    print("False")

#%%

phone_book = ["12","123","1235","567","88"]	
phone_book = ["987987", "87"]	
#phone_book = ["123","234","789"]
phone_book = sorted(phone_book,key=len)

for _ in phone_book:
    ref = phone_book.pop(0)
    
    phone_number_list = [s for s in phone_book if ref in s[:len(ref)]]
    print(phone_number_list,len(phone_number_list))
    if len(phone_number_list) >= 1:
        print("False")
        break
else:
    print("True")
    

#%%
phone_book = ["12","123","1235","567","88"]	
phone_book = sorted(phone_book,key=len)

for p1, p2 in zip(phone_book,phone_book[1:]):
    print(p1,p2)
    if p2.startswith(p1):
        print("False")
        break
else:
    print("True")
#%%
# ist(map(list, zip(*mylist)))
import collections 
import itertools
import functools 
clothes	= [["yellow_hat", "headgear"], ["blue_sunglasses", "eyewear"], ["green_turban", "headgear"],["test", "test"]]
cnt = collections.Counter([kind for name, kind in clothes])

#%%
clothes_dict = collections.defaultdict(list)
number_list = []
for value,key in clothes:
    clothes_dict[str(key)].append(value)

answer =  1
for key in clothes_dict:
    tmp_answer =  len(list(itertools.combinations(clothes_dict[str(key)],1))) + 1
    answer = (answer*tmp_answer)
    
print(answer-1)
#%%
class my_dict(dict):
               
    def __init__(self):
        self = dict()
    
    def add_genre(self,genre):
        self[genre] = []
        self[genre] = []

    def add(self,genre,plays,num):
        try:
            self[genre].append(tuple([plays,num]))
        except KeyError:
            self[genre] = [plays,num]
            

    # def list_to_tuple(self):
    #     for key in dic.keys():
    #         self[str(key)][0] = tuple(self[str(key)][0])
    #         self[str(key)][1] = tuple(self[str(key)][1])



def sum_plays(dic):
    return sum(dic[1][0])


genres = ["classic", "pop", "classic", "classic", "pop"]
plays = [500, 600, 150, 800, 2500]

dic = my_dict()
for genre in list(set(genres)):
    dic.add_genre(genre)

for num,(genre,play) in enumerate(zip(genres,plays)):
    dic.add(genre,play,num)

dic = dict(sorted(dic.items(),key=sum_plays, reverse=True))


tmp = []
for a in list(dic.values()):
    tmp += sorted(a,key=lambda x:x[0],reverse=True)[:2]

answer = [ans[1] for ans in tmp]
#print(answer)
#%%
dic.list_to_tuple()

#%%
class my_dict(dict):
               
    def __init__(self):
        self = dict()
    
    def add_genre(self,genre):
        self[genre] = [[],[]]
        self[genre] = [[],[]]

    def add(self,genre,plays,num):
        try:
            self[genre][0].append(plays)
            self[genre][1].append(num)
        except KeyError:
            self[genre][0] = [plays]
            self[genre][1] = [num]

    def list_to_tuple(self):
        for key in dic.keys():
            self[str(key)][0] = tuple(self[str(key)][0])
            self[str(key)][1] = tuple(self[str(key)][1])



def sum_plays(dic):
    return sum(dic[1][0])


genres = ["classic", "pop", "classic", "classic", "pop"]
plays = [500, 600, 150, 800, 2500]

dic = my_dict()
for genre in list(set(genres)):
    dic.add_genre(genre)

for num,(genre,play) in enumerate(zip(genres,plays)):
    dic.add(genre,play,num)

dic.list_to_tuple()
dic = dict(sorted(dic.items(),key=sum_plays, reverse=True))

#%%
answer = []

for a in list(dic.values()):

    print(a)
    print(sorted(a,key=lambda x:x[0],reverse=True))


    
#print(answer)
#%%

sorted(dic.values(),)
#list(map(sum,*list(dic.values()) ))

#%%
genres = ["classic", "pop", "classic", "classic", "pop"]
plays = [500, 600, 150, 800, 2500]
dic = collections.defaultdict(list)
dic_order = collections.defaultdict(list)


def sum_plays(dic):
    return sum(dic[1])


for genre,play in zip(genres,plays):
    dic[str(genre)].append(play)

#%%
print(dic)
dic = dict(sorted(dic.items(),key=sum_plays, reverse=True))
print(dic)
#%%

for (g,p) in dic.items():
    
    dic_order[str(g)] = sorted(p,reverse=True)

print(dic_order)
#%%

for play,order in zip(plays,order_list):
    dic_order[str(play)] = order



#map(list)

#%%
class Heap:
    def __init__(self,data):
        self.heap_array = list()
        self.heap_array.append(None) #index 1번부터
        self.heap_array.append(data)

    def move_up(self, inserted_idx):
        if inserted_idx <= 1:
            return False
        
        parent_idx = inserted_idx // 2
        
        if self.heap_array[inserted_idx] > self.heap_array[parent_idx]:
            return True
        else:
            return False

    def insert(self,data):
        if len(self.heap_array) == 1:
            self.heap_array.append(data)
            return True

        self.heap_array.append(data)
        inserted_idx = len(self.heap_array) -1 

        while self.move_up(inserted_idx):
            parent_idx = inserted_idx // 2
            self.heap_array[inserted_idx], self.heap_array[parent_idx] =  ...
            self.heap_array[parent_idx],self.heap_array[inserted_idx]
            inserted_idx = parent_idx
        return True

heap = Heap(15)
heap.insert(10)
# heap.insert(8)
# heap.insert(5)
# heap.insert(4)
# heap.insert(20)
heap.heap_array

#%%
import heapq

def my_heap_example(L, T):
  """ 주어진 비커의 리스트를 힙 구조로 변환 """
  heapq.heapify(L) 

  result = 0

  while len(L) >= 2 : #IndexError 방지
      """ 힙에서 최솟값을 가져옴 """
      min_ = heapq.heappop(L) 
      
      if min_ >= T: # 액체의 최솟값이 T보다 크다는 조건 만족(종료)
        print("-"*40, "\nresult:", result)
        return result 
        
      else: # 두 번째로 작은 값 가져와서 합친 값을 힙에 삽입
        min_2 = heapq.heappop(L) 
        heapq.heappush(L, min_ + 2*min_2)
        result += 1
        print("step{}: [{},{}] 합칩".format(result, min_ , min_2))
        print("       →", L)

my_heap_example([1, 2, 3, 9, 10, 12]	,7)     
#%%
import heapq 
A = [1, 2, 3, 9, 10, 12]
#A = [13,13]
K = 7
answer = -1
heapq.heapify(A)

while len(A) >= 2:
    
    min_1 = heapq.heappop(A)

    if min_1 >= K:
        #print()
        break 
    else:
        min_2 = heapq.heappop(A)
        heapq.heappush(A,min_1 + min_2*2)
        answer += 1
        


answer = answer  + 1
print(answer)    

#%%
import numpy as np
A = [[0, 3], [1, 9], [2, 6]]

A = sorted(A,key=lambda x:x[1])
answer = 0


for n,a in enumerate(A):
    answer = answer + (a[1])*(len(A)-n) - a[0]

    print(answer)
    
# %%
#%%
class Heap:
    def __init__(self,data):
        self.heap_array = list()
        self.heap_array.append(None) #index 1번부터
        self.heap_array.append(data)

    def move_up(self, inserted_idx):
        if inserted_idx <= 1:
            return False
        
        parent_idx = inserted_idx // 2
        
        if self.heap_array[inserted_idx] > self.heap_array[parent_idx]:
            return True
        else:
            return False

    def insert(self,data):
        if len(self.heap_array) == 1:
            self.heap_array.append(data)
            return True

        self.heap_array.append(data)
        inserted_idx = len(self.heap_array) -1 

        while self.move_up(inserted_idx):
            parent_idx = inserted_idx // 2
            self.heap_array[inserted_idx], self.heap_array[parent_idx] =  ...
            self.heap_array[parent_idx],self.heap_array[inserted_idx]
            inserted_idx = parent_idx
        return True
#%%
import heapq 

jobs = [[0, 3], [1, 9], [2, 6]]	
heapq.heapify(jobs)

jobs_num = len(jobs)
waiting = []

T = jobs.pop(0)
T = [T[1],T[0]]

ans = T[0]
time_stack = T[0]
cur_time = 1 

while True:

    if len(jobs) != 0:
        heapq.heappush(waiting,[jobs.pop(0)[1],cur_time] )


    if cur_time == time_stack:
        T = heapq.heappop(waiting)
        
        ans += cur_time - T[1] + T[0]
        print(T,waiting,cur_time - T[1] + T[0])
        time_stack = time_stack + T[0]


    if len(jobs) == 0 and len(waiting) == 0:
        break

    cur_time = cur_time + 1

answer = ans / jobs_num
print(answer)

#%%
import heapq 

jobs = [[0, 3], [1, 9], [2, 6]]	

heapq.heapify(jobs)

jobs_num = len(jobs)
waiting = []
time_stack = jobs[0][1]
cur_time = 0
duration = cur_time - jobs[0][0] + jobs[0][1]
ans = duration
jobs.pop(0)
def duration_sum(lists,cur):
    return_list = []
    
    for a in lists:
        return_list.append([ cur-a[0] + a[1],a[1],a[0] ])
    return return_list

while True:

    cur_time = cur_time + 1

    if len(jobs) != 0:
        waiting.append(jobs.pop(0))

    if cur_time == time_stack:
        duration_list = duration_sum(waiting,cur_time)

        heapq.heapify(duration_list)
        
        dur,stk,ref = heapq.heappop(duration_list)
        waiting = list(map(lambda a:[a[2],a[1]],duration_list) )
  
        ans += dur
        time_stack += stk

            
    if len(jobs) == 0 and len(waiting) == 0:
        break
    


answer = ans / jobs_num

print(ans,answer)

#%%
import heapq 

jobs = [[0, 3], [1, 9], [2, 6]]	

waiting = []
heapq.heapify(jobs)

T = jobs[0][1]

time = 0 
answer = 0


while True:

    if len(jobs) != 0:
        heapq.heappush(waiting,(jobs.pop(0)[1]) )

    waiting = [sum(i) for i in zip(waiting,list(range(len(waiting),0,-1)))]
    print(waiting,time,T)
    
    if time == T:
        T = heapq.heappop(waiting)
        answer = answer + T
        time = 0
        
        
    #print(time,T)
    if len(jobs) == 0 and len(waiting) == 0:
        break

    time = time + 1
    



#%%
import time
start = time.time()  
class Heap:
    def __init__(self):
        
        self.max_list = []
        self.max_list.append(None)
        self.min_list = []
        self.min_list.append(None)

    def move_up(self,in_idx):
        if in_idx <= 1:
            return False
        
        pa_idx = in_idx // 2

        if self.max_list[pa_idx] < self.max_list[in_idx]:
            return True
        else:
            return False

    def move_down(self,in_idx):
        if in_idx <= 1:
            return False
        
        pa_idx = in_idx // 2

        if self.min_list[pa_idx] > self.min_list[in_idx]:
            return True
        else:
            return False
    
    def insert(self,num):

        self.max_list.append(num)

        in_idx = len(self.max_list) - 1
        
        while self.move_up(in_idx):
            pa_idx = in_idx // 2
            self.max_list[pa_idx],self.max_list[in_idx] = self.max_list[in_idx],self.max_list[pa_idx]
            in_idx = pa_idx

        self.min_list.append(num)
        in_idx = len(self.min_list) - 1

        while self.move_down(in_idx):
            pa_idx = in_idx // 2
            self.min_list[pa_idx],self.min_list[in_idx] = self.min_list[in_idx],self.min_list[pa_idx]
            in_idx = pa_idx
        return True

    def delete(self,num):
        if  len(self.min_list) == 1 or len(self.max_list) == 1:
            return 
            
        if num == -1:
            self.min_list.pop(1)

        if num == 1:
            self.max_list.pop(1)
        return 

test = Heap()
#operations = ["I 4", "I 3", "I 2", "I 1", "D 1", "D 1", "D -1", "D -1", "I 5", "I 6"] 
operations = ["I -45", "I 653", "D 1", "I -642", "I 45", "I 97", "D 1", "D -1", "I 333"]


answer = []
ans = []
while operations:
    string = operations.pop(0).split()
    
    if string[0] == "I":
        test.insert(int(string[-1]))
    if string[0] == "D":
        test.delete(int(string[-1]))
    #print(string,test.max_list,test.min_list)
if len(test.min_list) <= 2 or len(test.max_list) <= 2:
    answer = [0,0]
else:
    answer = [a for a in test.max_list if a in test.min_list]
    
    #answer = [max(ans),min(ans)]

answer = [ max(answer[1:]),min(answer[1:]) ]
print(answer)
#%%
