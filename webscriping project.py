#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install selenium')


# In[2]:


get_ipython().system('pip install webdriver-manager')


# In[3]:



import selenium
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By
import warnings
warnings.filterwarnings("ignore")
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


# In[83]:


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


# # question=1

# In[5]:


driver.get("https://www.naukri.com/")


# In[18]:


designation = driver.find_element(By.CLASS_NAME,"suggestor-input")
designation.send_keys('Data Analyst')


# In[20]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div[6]/div/div/div[5]/div/div/div/div[1]/div/input")
location.send_keys('Bangalore')


# In[23]:


search = driver.find_element(By.CLASS_NAME,"qsbSubmit")
search.click()


# In[24]:


job_location = []


# In[29]:


location_tags = driver.find_elements(By.XPATH,'//span[@class="ellipsis fleft locWdth"]')
for i in location_tags[0:10]:
    location = i.text
    job_location.append(location)


# In[ ]:





# In[30]:


print(len(job_location))


# In[31]:


import pandas as pd
df= pd.DataFrame({'Location':job_location})
df


# # question=2

# In[32]:


driver.get("https://www.naukri.com/")


# In[16]:


designation = driver.find_element(By.CLASS_NAME,"suggestor-input")
designation.send_keys('Data Scientist')


# In[17]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div[7]/div/div/div[5]/div/div/div/div[1]/div/input")
location.send_keys('Bangalore')


# In[36]:


search = driver.find_element(By.CLASS_NAME,"qsbSubmit")
search.click()


# In[38]:


job_location = []


# In[39]:


location_tags = driver.find_elements(By.XPATH,'//span[@class="ellipsis fleft locWdth"]')
for i in location_tags[0:10]:
    location = i.text
    job_location.append(location)


# In[40]:


print(len(job_location))


# In[41]:


import pandas as pd
df= pd.DataFrame({'Location':job_location})
df


# # question=3

# In[25]:


driver.get("https://www.naukri.com/")


# In[26]:


designation = driver.find_element(By.CLASS_NAME,"suggestor-input")
designation.send_keys('Data Scientist')


# In[27]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div[6]/div/div/div[5]/div/div/div/div[1]/div/input")
location.send_keys('New Delhi')


# In[28]:


search = driver.find_element(By.CLASS_NAME,"qsbSubmit")
search.click()


# In[29]:


job_location = []
job_salary = []


# In[30]:


location_tags = driver.find_elements(By.XPATH,'//span[@class="ellipsis fleft locWdth"]')
for i in location_tags[0:10]:
    location = i.text
    job_location.append(location)


# In[32]:


salary_tags = driver.find_elements(By.XPATH,'//span[@class="ellipsis fleft "]')
for i in salary_tags[0:10]:
    salary = i.text
    job_salary.append(salary)


# In[33]:


print(len(job_location), len(job_salary))


# In[34]:


import pandas as pd
df= pd.DataFrame({'Location':job_location,'Salary':job_salary})
df


# # question=4

# In[47]:


driver.get("https://www.flipkart.com/")


# In[48]:


product=  driver.find_element(By.CLASS_NAME,"_3704LK")
product.send_keys('sunglasses')


# In[50]:


location = driver.find_element(By.XPATH,"/html/body/div/div/div[1]/div[1]/div[2]/div[2]/form/div/div/input")
location.send_keys('flipkart')


# In[52]:


search = driver.find_element(By.CLASS_NAME,"_3704LK")
search.click()


# In[54]:


sunglasses_brand= []

sunglasses_price = []


# In[55]:


brand_tags = driver.find_elements(By.XPATH,'//div[@class="_2WkVRV"]')
for i in brand_tags[0:100]:
    brand= i.text
    sunglasses_brand.append(brand)


# In[56]:


price_tags = driver.find_elements(By.XPATH,'//div[@class="_30jeq3"]')
for i in price_tags[0:100]:
    price= i.text
    sunglasses_price.append(brand)


# In[57]:


print(len(sunglasses_brand), len(sunglasses_price))


# In[58]:


import pandas as pd
df= pd.DataFrame({'Brand':sunglasses_brand,'Price':sunglasses_price})
df


# # question=6

# In[6]:


driver.get("https://www.flipkart.com/")


# In[7]:


product=  driver.find_element(By.CLASS_NAME,"Pke_EE")
product.send_keys('sneakers')


# In[8]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div/div[1]/div/div/div/div/div[1]/div/div[1]/div/div[1]/header/div[1]/div[2]/form/div/div/input")
location.send_keys('flipkart')


# In[9]:


search = driver.find_element(By.CLASS_NAME,"Pke_EE")
search.click()


# In[14]:


sneakers_brand= []

sneakers_price = []


# In[15]:


brand_tags = driver.find_elements(By.XPATH,'//div[@class="_2WkVRV"]')
for i in brand_tags[0:100]:
    brand= i.text
    sneakers_brand.append(brand)


# In[16]:


price_tags = driver.find_elements(By.XPATH,'//div[@class="_30jeq3"]')
for i in price_tags[0:100]:
    price= i.text
    sneakers_price.append(brand)


# In[17]:


print(len(sneakers_brand), len(sneakers_price))


# In[18]:


import pandas as pd
df= pd.DataFrame({'Brand':sneakers_brand,'Price':sneakers_price})
df


# # question=5

# In[19]:


driver.get("https://www.flipkart.com/")


# In[20]:


product=  driver.find_element(By.CLASS_NAME,"Pke_EE")
product.send_keys('iphone11')


# In[21]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div/div[1]/div/div/div/div/div[1]/div/div[1]/div/div[1]/header/div[1]/div[2]/form/div/div/input")
location.send_keys('iphone11')


# In[22]:


search = driver.find_element(By.CLASS_NAME,"Pke_EE")
search.click()


# In[23]:


iphone11_rating= []

iphone11_review = []


# In[43]:


rating_tags = driver.find_elements(By.XPATH,'//div[@class="_3LWZlK"]')
for i in rating_tags[0:50]:
    rating= i.text
    iphone11_rating.append(rating)


# In[46]:


review_tags = driver.find_elements(By.XPATH,'//p[@class="_2-N8zT"]')
for i in review_tags[0:13]:
    review= i.text
    iphone11_review.append(review)


# In[47]:


print(len(iphone11_rating), len(iphone11_review))


# In[48]:


import pandas as pd
df= pd.DataFrame({'Rating':iphone11_rating,'Review':iphone11_review})
df


# # question=7

# In[49]:


driver.get("https://www.amazon.in/")


# In[50]:


product=  driver.find_element(By.CLASS_NAME,"nav-input")
product.send_keys('Laptop')


# In[51]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/header/div/div[1]/div[2]/div/form/div[2]/div[1]/input")
location.send_keys('Intel Core i7')


# In[52]:


search = driver.find_element(By.CLASS_NAME,"nav-input")
search.click()


# In[53]:


laptop_title= []

laptop_rating = []
laptop_price = []


# In[76]:


title_tags = driver.find_elements(By.XPATH,'//span[@class="a-size-medium a-color-base a-text-normal"]')
for i in title_tags[0:50]:
    title= i.text
    laptop_title.append(title)


# In[77]:


rating_tags = driver.find_elements(By.XPATH,'//i[@class="a-icon a-icon-star-small a-star-small-4-5 aok-align-bottom"]')
for i in rating_tags[0:50]:
    rating= i.text
    laptop_rating.append(rating)


# In[78]:


price_tags = driver.find_elements(By.XPATH,'//span[@class="a-price-whole"]')
for i in price_tags[0:50]:
    price= i.text
    laptop_price.append(price)


# In[79]:


print(len(laptop_title), len(laptop_rating),len(laptop_price))


# In[80]:


import pandas as pd
df= pd.DataFrame({'Title':laptop_title,'Rating':laptop_rating,'Price':laptop_price})
df


# In[ ]:




