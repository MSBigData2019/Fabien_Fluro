# coding: utf-8
import requests
import unittest
from bs4 import BeautifulSoup

#website_prefix = "https://www.reuters.com/finance/stocks/"

page_url = 'https://www.reuters.com/finance/stocks/financial-highlights/LVMH.PA'


def _handle_request_result_and_build_soup(request_result):
  if request_result.status_code == 200:
    html_doc =  request_result.text
    soup = BeautifulSoup(html_doc,"html.parser")
    return soup

def _get_soup(page_url):
  res = requests.get(page_url)
  soup = _handle_request_result_and_build_soup(res)
  return soup
    

def get_sales(company, soup):
    module = soup.find(class_= "column1").find_all("div", class_ = "module")[1]
    sales = module.find_all("tr")[2].find_all(class_ = "data")
    sales_mean_text = sales[1].text
    sales_high_text = sales[2].text
    sales_low_text = sales[3].text
    
    sales_mean = _convert_string_to_float(sales_mean_text)
    sales_high = _convert_string_to_float(sales_high_text)
    sales_low = _convert_string_to_float(sales_low_text)
    
    company.sales_mean_text = sales_mean_text
    company.sales_high_text = sales_high_text
    company.sales_low_text = sales_low_text
    
    company.sales_mean = sales_mean
    company.sales_high = sales_high
    company.sales_low = sales_low
    return sales
  
def get_shares_price(company, soup):
  module = soup.find(id= "sectionHeader").find("div", class_ = "module")
  company.shares_price_text = module.find(class_ = "nasdaqChange").find_all("span")[1].text.strip()
  company.shares_price = float(company.shares_price_text)
  return company.shares_price
    
    
def get_shares_percentage_change(company, soup):
    module = soup.find(id= "sectionHeader").find("div", class_ = "module") 
    shares_percentage_change_text = module.find(class_ = "priceChange").find_all("span")[1].find('span').text.strip()
    shares_percentage_change = float(shares_percentage_change_text[1:])
    company.shares_percentage_change_text = shares_percentage_change_text
    company.shares_percentage_change = shares_percentage_change  
    return shares_percentage_change  
    

def get_shares_owned_percentage(company, soup):
    module = soup.find(class_ = "column2").find_all("div", class_ = "module")[3]
    percentage_text = module.find_all("tr")[0].find(class_ = "data").text;
    percentage = _convert_percentage_to_float(percentage_text)
    company.shares_owned_percentage_text = percentage_text
    company.shares_owned_percentage = percentage
    return percentage
    
def get_dividend_yield(company, soup):
    module = soup.find(class_ = "column1").find_all("div", class_ = "module")[3]
    data = module.find_all("tr")[1].find_all(class_ = "data")
    dividends_text = list(map(lambda x : x.text, data))
    dividends = list(map(lambda x : float(x), dividends_text))
    company.dividend_company_text = dividends_text[0]
    company.divident_industry_text = dividends_text[1]
    company.divident_sector_text = dividends_text[2]
    
    company.dividend_company = dividends[0]
    company.divident_industry = dividends[1]
    company.divident_sector = dividends[2]
    return dividends
  

def _convert_string_to_float(string):
    return float(string.replace(',','')) * 1000000


def _convert_percentage_to_float(string):
    return float(string[:-1])


def get_financial_information(company, page_url):
    soup = _get_soup(page_url)
    
    company = Company(company)
    get_sales(company, soup)
    get_shares_percentage_change(company, soup)
    get_shares_price(company, soup)
    get_shares_owned_percentage(company, soup)
    get_dividend_yield(company, soup)
    company.show_info()
    return

class Company:
    
    def __init__(self, name):
        self.name = name
    
    def show_info(self):
        print(self.name)
        print('===========================')
        print('Sales mean (in millions) : ' + self.sales_mean_text)
        print('Sales high (in millions) : ' + self.sales_high_text)
        print('Sales low (in millions) : ' + self.sales_low_text)
        print('Shares price : ' + self.shares_price_text)
        print('Shares change (%) : ' + self.shares_percentage_change_text)
        print('Percentage shares : ' + self.shares_owned_percentage_text)
        print('Dividend (company) : ' + self.dividend_company_text)
        print('Dividend (industry) : ' + self.divident_industry_text)
        print('Dividend (sector) : ' + self.divident_sector_text)
        print()
        print()

      
class Lesson2Tests(unittest.TestCase):
    
    def testConvertStringFloat(self):
        self.assertEqual(_convert_string_to_float("13,667.70") , 13667700000)
        self.assertEqual(_convert_percentage_to_float("20.57%") , 20.57)

get_financial_information('LVMH', 'https://www.reuters.com/finance/stocks/financial-highlights/LVMH.PA')
get_financial_information('Airbus', 'https://www.reuters.com/finance/stocks/financial-highlights/AIR.PA')
get_financial_information('Danone', 'https://www.reuters.com/finance/stocks/financial-highlights/DANO.PA')


def main():
    unittest.main()
    
if __name__ == '__main__':
    main()
