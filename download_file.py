import requests

url = 'https://onlinelibrary.wiley.com/doi/epdf/10.1002/prot.25590'
r = requests.get(url, stream=True, allow_redirects=True)

with open('/tmp/s.pdf', 'wb') as fd:
    for chunk in r.iter_content(1024):
        print(chunk)
        fd.write(chunk)