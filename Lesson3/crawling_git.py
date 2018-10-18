#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import requests
import unittest
import json
from bs4 import BeautifulSoup
import pandas as pd

page_url = 'https://gist.github.com/paulmillr/2657075'
token = 'a286d5bb7d3ffcc7f8cd298bc067d967f7fc7da7'

take = 256

#thejameskyle

def _handle_request_result_and_build_soup(request_result):
  if request_result.status_code == 200:
    html_doc =  request_result.text
    soup = BeautifulSoup(html_doc,"html.parser")
    return soup

def _get_soup(page_url):
  res = requests.get(page_url)
  soup = _handle_request_result_and_build_soup(res)
  return soup

def _extract_name_from_string(text):
    m = re.search("\((.+)\)", text)
    return m.group(1)

def _extract_nickname_from_string(text):
    return text.split(" (")[0]

def _get_average(col):
    return round(sum(col)/len(col), 1)

def get_contributors(page_url):
    soup = _get_soup(page_url)
    table = soup.find("table")
    trs = table.find("tbody").find_all("tr")[:take]
    nicknames = list(map(lambda tr: _extract_nickname_from_string(tr.find_all("td")[0].text), trs))
    return nicknames

def get_git_stars_score(nickname):
    headers = {'Authorization': 'token ' + token}
    r = requests.get('https://api.github.com/users/' + nickname + '/repos?per_page=1000', headers=headers)
  
    if(r.ok):
        json_content = json.loads(r.text or r.content)
        dftemp = pd.DataFrame(json_content)
        if "stargazers_count" in dftemp.columns:
            return _get_average(dftemp["stargazers_count"])
        else: 
            return 0

df = pd.DataFrame({ 'login': get_contributors(page_url)})
scores = []

for name in contributors[:take]:
    score = get_git_stars_score(name)
    scores.append(score)
    print(name + ' : score = ' + str(score))

df['score'] = scores
df = df.sort_values('score', axis=0, ascending=False)

print()
print(df.head())


class Lesson3Tests(unittest.TestCase):
    
    def testExtractName(self):
        self.assertEqual(_extract_nickname_from_string("login (user name)") , "login")
        self.assertEqual(_extract_nickname_from_string("unicodeveloper (Prosper Otemuyiwa)") , "unicodeveloper")
        self.assertEqual(_extract_nickname_from_string("atian25 (TZ | 天猪)") , "atian25")

def main():
    unittest.main()
    
if __name__ == '__main__':
    main()
