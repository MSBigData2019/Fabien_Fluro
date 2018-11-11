#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import requests
import unittest
import json
from bs4 import BeautifulSoup
import pandas as pd

#https://fr.distance24.org/route.json?stops=paris|marseille
page_url = 'http://www.linternaute.com/ville/classement/villes/population'


def _handle_request_result_and_build_soup(request_result):
  if request_result.status_code == 200:
    html_doc =  request_result.text
    soup = BeautifulSoup(html_doc,"html.parser")
    return soup


def _get_soup(page_url):
  res = requests.get(page_url)
  soup = _handle_request_result_and_build_soup(res)
  return soup

def _extract_city_from_string(text):
    return text.split(" (")[0]
    
def _get_distance(city1, city2):
    q = 'https://fr.distance24.org/route.json?stops=' + city1 + '|' + city2
    r = requests.get(q)
    if(r.ok):
        json_content = json.loads(r.text or r.content)
        return json_content['distances'][0]

def get_cities(page_url):
    soup = _get_soup(page_url)
    table = soup.find("table")
    trs = table.find("tbody").find_all("tr")[:50]
    cities = list(map(lambda tr: _extract_city_from_string(tr.find_all("td")[1].find('a').text), trs))
    return cities




_get_distance('paris', 'marseille')

cities = get_cities(page_url)


for city1 in cities:
    for city2 in cities:
       print(city1 + '->' + city2 + ' : ' + str(_get_distance(city1, city2)) + 'Km')
        

