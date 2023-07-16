#!/usr/bin/env python
# coding: utf-8

# # question=1

# In[2]:


import regex as re


# In[3]:


data1="This Is A Good Girl"
x=re.findall("[A-Z]",data1)
print(x)


# In[4]:


y=re.findall("[a-z]",data1)
print(y)


# In[5]:


data1="This Is A Good Girl,Only 12th class pass"
z=re.findall("[0-9]",data1)
print(z)


# In[ ]:





# # question=6

# In[37]:


import regex as re


# In[5]:


text="ImportanceOfRegularExpressionInPython"
result=re.findall('[A-Z][^A-Z]*',text)
print(result)


# # question=8

# In[11]:


text1="ImportanceOfRegularExpressionInPython"
pattern= '^[a-z]+_[a-z]+$'
result1= re.search(pattern,text1)
print(result1)


# # question=7

# In[8]:


data1="ab, abb, abc, baa, caa bcc"
pattern= 'ab{2,3}'
result2=re.search(pattern,data1)
print(result2)


# # question=9

# In[18]:


set1="accd, baaacd, bbbbd, acccjjb, dfghb, hggfabgd, accddbbjjjb"
pattern= ( 'a.*?b$')
result2=re.search(pattern,set1)
print(result2)


# # question=2

# In[17]:


set2="a, baaa, bb, ac, dhb, d, ab, aa, abb, acc, abc"
pattern= ('ab*?')
string=re.search(pattern,set2)
print(string)


# # question3

# In[23]:


set3= "Hello World Data Science"
pattern1=(".+")
string2=re.search(pattern1,set3)
print(string2)


# In[28]:


import re


# # question=4

# In[30]:


file="ab, abc, abbc, aabbc"
petterns=('ab?')
set=re.search(petterns, file)
print(set)


# # question=5

# In[31]:


file2="abb, abbbc, abbc, aabbbbbc"
petterns=('ab{3}?')
set2=re.search(petterns, file2)
print(set2)


# In[32]:


import re


# # question=10

# In[34]:


def text_match(text):
    patterns= '^\w+'
    if re.search(patterns, text):
        return 'Found a match!'
    else:
        return('Not matched!')
print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match(" The quick brown fox jumps over the lazy dog."))


# # question=11

# In[1]:


import re


# In[5]:


search="Python_Exercises_1"
pattern='^[a-zA-Z0-9_]*$'
doller= re.search(pattern, search)
print(doller)


# # question=12

# In[1]:


import re


# In[5]:


def starts_with_number(string, number):
    return string.startswith(str(number))

string1= "123abc"
number1= 123
print(starts_with_number(string1, number1))


# # question=13

# In[19]:


ip= "216.08.094.196"
string = re.sub('\.[0]*', '.', ip)
print(string)


# In[29]:


pattern = ['fox', 'dog', 'horse']
text= 'The quick brown fox jumps over the lazy dog'
for pattern in patterns:
    print('Searching for "%s" in "%s" ->' % (pattern, text),)
    if re.search(pattern, text):
        print('Matched!')
        else:
            print('Not Matched!')
        


# # question=16

# In[27]:


pattern = 'fox'
text= 'The quick brown fox jumps over the lazy dog'
match = re.search(pattern, text)
s= match.start()
e = match.end()
print('Found "%s" in "%s" from %d to %d ' %(match.re.pattern, match.string, s, e))


# # question=17

# In[30]:


text = 'Python exercises, PHP exercises, C# exercises'
pattern = 'exercises'
for match in re.findall(pattern, text):
    print('Found "%s"' % match)


# # question=19

# In[36]:


def change_date_format(dt):
    return re.sub(r'(\d{4})-(\d{1,2})-(\d{1,2})','\\3-\\2-\\1',dt)
dt1="2026-01-02"
print("original date in YYYY-M-DD Formate:",dt1)
print("New date in DD-MM-YYYY Formate:",change_date_format(dt1))
    


# # QUESTION=20

# In[37]:


text = "The following example creates an ArrayList with a capacity of 50 elements. Four elements are then added to the ArrayList and the ArrayList is trimmed accordingly."
#find all the words starting with 'a' or 'e'
list = re.findall("[ae]\w+", text)
# Print result.
print(list)


# # QUESTION=21

# In[38]:


text = "The following example creates an ArrayList with a capacity of 50 elements. Four elements are then added to the ArrayList and the ArrayList is trimmed accordingly."

for m in re.finditer("\d+", text):
    print(m.group(0))
    print("Index position:", m.start())
	


# # QUESTION=23

# In[39]:


street = '21 Ramkrishna Road'
print(re.sub('Road$', 'Rd.', street))


# # QUESTION=24

# In[40]:


def text_match(text):
        patterns = '^[a-zA-Z0-9_]*$'
        if re.search(patterns,  text):
                return 'Found a match!'
        else:
                return('Not matched!')

print(text_match("The quick brown fox jumps over the lazy dog."))
print(text_match("Python_Exercises_1"))


# In[1]:


import re


# # question=25

# In[10]:


string_1= "This is is a good good girl"
regex = r'\b(\w+)(?:\W+\1\b)+'
y= re.sub(regex, r'\1', string_1, flags = re.IGNORECASE)
print(y)


# In[1]:


import re


# # question=26

# In[3]:


text1= '**//Python Exercises// - 12.'
pattern = re.compile('[\W_]+')
print(pattern.sub('', text1))


# # question=27

# In[6]:


line=""""RT @kapil_kausik: #Doltiwal I mean #xyzabc is "hurt" by #Demonetization as the same has rendered USELESS <ed><U+00A0><U+00BD><ed><U+00B1><U+0089> "acquired funds" No wo""""
        hashtags1 = re.findall(r'#\W+', line)
        print(hashtags1)


# In[4]:


with open ('Email.txt', 'r') as file:
    content = file.read()
        urls = re.findall(r'#\W+', content)
        for url in urls:
            print(url)


# # question=29

# In[5]:


import re


# In[7]:


import re


with open('sample_text.txt', 'r') as file:
    
    content = file.read()

    
    dates = re.findall(r'\d{2}-\d{2}-\d{4}', content)

    
    for date in dates:
        print(date)


# # question=30

# In[8]:


import re
text = 'Python Exercises, PHP exercises.'
print(re.sub("[ ,.]", ":", text))


# In[ ]:





# In[ ]:




