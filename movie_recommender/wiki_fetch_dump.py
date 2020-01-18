"""
Module to Ffetch xml dump of wiki pages
"""

import requests
from bs4 import BeautifulSoup
import os

WIKI_DUMP_ROOT_URL = 'https://dumps.wikimedia.org/'
WIKI_DUMP_URL = WIKI_DUMP_ROOT_URL + 'enwiki/'

def get_wiki_dump_xml_url():
    index = requests.get(WIKI_DUMP_URL).text
    soup_index = BeautifulSoup(index, "html.parser")
    dumps = [a['href'] for a in soup_index.find_all('a') if a.has_attr('href') and a.text[:-1].isdigit()]
    print(dumps)
    for dump_url in sorted(dumps, reverse=True):
        dump_html = requests.get(WIKI_DUMP_URL + dump_url).text
        soup_dump = BeautifulSoup(dump_html, 'html.parser')
        pages_xml = [a['href'] for a in soup_dump.find_all('a') if a.has_attr('href') and a['href'].endswith('-pages-articles.xml.bz2')]
        if pages_xml:
            print(pages_xml)
            break
    return WIKI_DUMP_ROOT_URL + pages_xml[0]



if __name__ == "__main__":
    xml_url = get_wiki_dump_xml_url()
    print(xml_url)
