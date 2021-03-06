import os
import urllib.request  # 导入用于打开URL的扩展库模块
import urllib.parse
import re  # 导入正则表达式模块
from config import DATA_DIR


def open_url(url):
    req = urllib.request.Request(url)  # 将Request类实例化并传入url为初始值，然后赋值给req
    # 添加header，伪装成浏览器
    req.add_header('User-Agent',
                   'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0')
    # 访问url，并将页面的二进制数据赋值给page
    page = urllib.request.urlopen(req)
    # 将page中的内容转换为utf-8编码
    html = page.read().decode('utf-8')
    print(html)
    return html


def get_img(html, path=os.path.join(os.getcwd(), 'webData')):
    # [^"]+\.jpg 匹配除"以外的所有字符多次,后面跟上转义的.和png
    p = r'<img src="([^"]+\.(?:jpg|png|gif|webp|jpeg))"'
    # 返回正则表达式在字符串中所有匹配结果的列表
    imglist = re.findall(p, html)
    # 为什么这里的的<img src ="没了
    print(len(imglist))
    print(imglist)
    # 循环遍历列表的每一个值
    cnt = 0
    for each in imglist:
        # 以/为分隔符，-1返 回最后一个值
        filename = each.split("/")[-1]
        print(filename)
        print(path)
        # 访问each，并将页面的二进制数据赋值给photo
        photo = urllib.request.urlopen(each)
        w = photo.read()
        # 打开指定文件，并允许写入二进制数据
        cnt = cnt + 1
        f = open(os.path.join(path, filename), 'wb')

        # 写入获取的数据
        f.write(w)
        # 关闭文件
        f.close()


# 该模块既可以导入到别的模块中使用，另外该模块也可自我执行
if __name__ == '__main__':
    # test
    # get_img(open_url('https://findicons.com/pack/2787/beautiful_flat_icons'))
    # 搜索关键词: '病名 辣椒'
    url_yybd = "https://www.google.com/search?q=%E7%94%A8%E8%8D%AF%E4%B8%8D%E5%BD%93+%E8%BE%A3%E6%A4%92&hl=zh-CN&source=lnms&tbm=isch&sa=X&ved=2ahUKEwio5aX1yYryAhUHOisKHfbnC98Q_AUoAXoECAEQAw&biw=2048&bih=1047"
    url_tanyi = "https://www.google.com/search?q=%E7%82%AD%E7%96%BD%E7%97%85+%E8%BE%A3%E6%A4%92&hl=zh-CN&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjYl_KguYryAhVTdCsKHeZwDVIQ_AUoAXoECAEQAw&biw=1389&bih=1047"
    url_yi = "https://www.google.com/search?q=%E7%96%AB%E7%97%85+%E8%BE%A3%E6%A4%92&hl=zh-CN&source=lnms&tbm=isch&sa=X&ved=2ahUKEwipxb_wuIryAhWMbysKHT2aDMoQ_AUoAXoECAEQAw&biw=2048&bih=1047"

    url_tanyi_baidu = 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E7%82%AD%E7%96%BD%E7%97%85+%E8%BE%A3%E6%A4%92'
    url_yybd_baidu = 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E7%94%A8%E8%8D%AF%E4%B8%8D%E5%BD%93+%E8%BE%A3%E6%A4%92'
    url_yi_baidu = 'https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E7%96%AB%E7%97%85+%E8%BE%A3%E6%A4%92'

    # 将url作为open_url()的参数，然后将open_url()的返回值作为参数赋给get_img()
    # get_img(open_url(url_yybd), os.path.join(DATA_DIR, 'googlePic', 'yybd'))
    # get_img(open_url(url_tanyi), os.path.join(DATA_DIR, 'googlePic', 'tanyi'))
    # get_img(open_url(url_yi), os.path.join(DATA_DIR, 'googlePic', 'yi'))
    get_img(open_url(url_yybd_baidu), os.path.join(DATA_DIR, 'baiduPic', 'yybd'))
    # get_img(open_url(url_tanyi_baidu), os.path.join(DATA_DIR, 'baiduPic', 'tanyi'))
    # get_img(open_url(url_yi_baidu), os.path.join(DATA_DIR, 'baiduPic', 'yi'))

    # test
    # p = r'<img[.*]src="([^"]+\.(?:jpg|png|gif|webp))"'
    # # 返回正则表达式在字符串中所有匹配结果的列表
    # imglist = re.findall(p, '<img data-ils="4" jsaction="rcuQ6b:trigger.M8vzZb;" class="rg_i Q4LuWd" jsname="Q4LuWd" width="228" height="171" alt="突发性、流行快、毁灭性的辣椒炭疽病该如何防治？你早该这样做了_发病" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQoD5i9sT20kOlVJsnN49d7EetEbtpugqXSyw&amp;usqp=CAU">')
    # # 为什么这里的的<img src ="没了
    # print(len(imglist))
    # print(imglist)