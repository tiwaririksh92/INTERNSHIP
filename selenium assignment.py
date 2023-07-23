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


# In[5]:


driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))


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


# In[17]:


driver.get("https://www.flipkart.com/")


# In[18]:


product=  driver.find_element(By.CLASS_NAME,"Pke_EE")
product.send_keys('sneakers')


# In[19]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div/div[1]/div/div/div/div/div[1]/div/div[1]/div/div[1]/header/div[1]/div[2]/form/div/div/input")
location.send_keys('flipkart')


# In[20]:


search = driver.find_element(By.CLASS_NAME,"Pke_EE")
search.click()


# In[21]:


sneakers_brand= []

sneakers_price = []


# In[43]:


brand_tags = driver.find_elements(By.XPATH,'//div[@class="_2WkVRV"]')
for i in brand_tags[0:100]:
    brand= i.text
    sneakers_brand.append(brand)


# In[44]:


price_tags = driver.find_elements(By.XPATH,'//div[@class="_30jeq3"]')
for i in price_tags[0:100]:
    price= i.text
    sneakers_price.append(brand)


# In[45]:


print(len(sneakers_brand), len(sneakers_price))


# In[46]:


import pandas as pd
df= pd.DataFrame({'Brand':sneakers_brand,'Price':sneakers_price})
df


# In[55]:


driver.get("https://www.flipkart.com/")


# In[56]:


product=  driver.find_element(By.CLASS_NAME,"Pke_EE")
product.send_keys('iphone11')


# In[57]:


location = driver.find_element(By.XPATH,"/html/body/div[1]/div/div[1]/div/div/div/div/div[1]/div/div[1]/div/div[1]/header/div[1]/div[2]/form/div/div/input")
location.send_keys('iphone11')


# In[58]:


search = driver.find_element(By.CLASS_NAME,"Pke_EE")
search.click()


# In[59]:


iphone11_rating= []

iphone11_review = []


# In[60]:


rating_tags = driver.find_elements(By.XPATH,'//div[@class="_3LWZlK"]')
for i in rating_tags[0:100]:
    rating= i.text
    iphone11_rating.append(rating)


# In[61]:


review_tags = driver.find_elements(By.XPATH,'//div[@class="_3_L3jD"]')
for i in review_tags[0:100]:
    review= i.text
    iphone11_review.append(review)


# In[62]:


print(len(iphone11_rating), len(iphone11_review))


# In[63]:


import pandas as pd
df= pd.DataFrame({'Rating':iphone11_rating,'Review':iphone11_review})
df


# In[ ]:




