import yagmail


#yagmail.register('hao.bruce@gmail.com', 'wrkbojclogbsvsav')
yag = yagmail.SMTP('hao.bruce@gmail.com')

to = 'hao.bruce@gmail.com'
subject = 'SysTrade position targets'
body = 'Position targets below:'
html = df[['SettleRaw', 'Symbol', 'SystemPosition']].to_html()
html = html.replace('border="1"', 'border="0"')

yag.send(to=to, subject=subject, contents=[body, html])
