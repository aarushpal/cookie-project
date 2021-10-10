from GoogleNews import GoogleNews

googlenews = GoogleNews()
googlenews.search('How rich is Ambani')
result = googlenews.result()
print(len(result))

for n in range(len(result)):
    print(n)
    for index in result[n]:
        print(index, '\n', result[n][index])

    exit()