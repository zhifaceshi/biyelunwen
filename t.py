"""
@Time    :2020/2/17 21:41
@Author  : 梁家熙
@Email:  :11849322@mail.sustech.edu.cn
"""

from selenium.webdriver import Chrome, Firefox, PhantomJS
import time
import requests
from pathlib import Path
import re
from selenium.webdriver.common.action_chains import ActionChains
# 图片下载
# @:param url_info ('http://img.xixik.net/custom/section/country-flag/xixik-cdaca66ba3839767.png','北马里亚纳群岛)
def download_img(url_info):
    if url_info[1]:
        print("-----------正在下载图片 %s"%(url_info[0]))
        # 这是一个图片的url

        url = url_info[0]
        headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
                   'Accept - Encoding':'gzip, deflate',
                   'Accept-Language':'zh-Hans-CN, zh-Hans; q=0.5',
                   'Connection':'Keep-Alive',
                   'Cache-Control': 'max-age=0',
                   'Host':'zxgk.court.gov.cn',
                   'Proxy-Connection': 'keep-alive',
                   'Upgrade-Insecure-Requests': '1',
                   'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
                   }
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"返回状态码不正常{response.status_code}")
        # 获取的文本实际上是图片的二进制文本
        img = response.content
        # 将他拷贝到本地文件 w 写  b 二进制  wb代表写入二进制文本
        path=Path('./pic')
        path.mkdir(exist_ok=True)
        path = path  / url_info[1]
        with open(path, 'wb') as f:
            f.write(img)

driver = PhantomJS()     # 创建Chrome对象.
# 操作这个对象.
driver.get('http://zxgk.court.gov.cn/shixin/')     # get方式访问百度.
i = 1
j = 10000

while j >=0:
    a = driver.find_element_by_id("captchaImg")
    url = (a.get_attribute('src'))
    pic_name = f"{i}.png"
    try:
        download_img([url, pic_name])
    except Exception as e:
        print(e)
        continue
    print(f"{pic_name}已经下载成功，共成功下载{i}张验证码")
    i += 1
    j -= 1
    ActionChains(driver).move_to_element(a).click().perform()
    time.sleep(2) # 防止过快被封ip

driver.quit()   # 使用完, 记得关闭浏览器, 不然chromedriver.exe进程为一直在内存中.
