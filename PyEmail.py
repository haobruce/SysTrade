import smtplib


smtpObj = smtplib.SMTP('smtp.gmail.com', 587)
smtpObj.ehlo()
smtpObj.starttls()
smtpObj.login('hao.bruce@gmail.com', 'wrkbojclogbsvsav')  # application-specific password setup on google
# smtpObj.sendmail('hao.bruce@gmail.com', 'hao.bruce@gmail.com', 'Subject: Email content\nMore content.')
