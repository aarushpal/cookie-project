from googlesearch import search
for j in search('How common is saatvik name', tld='com', num=10, stop=10, pause=2):
  print(j)