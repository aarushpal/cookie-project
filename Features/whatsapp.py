import pywhatkit
mobile = input("Enter reciever's mobile number with country code: ")
message = input("Enter message you want to send: ")
hours = int(input("Enter the hours when you want to send: "))
minutes = int(input("Enter the minutes when you want to send: "))

pywhatkit.sendwhatmsg(mobile, message, hours, minutes)
