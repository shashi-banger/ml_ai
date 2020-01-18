import xml.sax

import subprocess
import mwparserfromhell
import json
import re


def process_article(title, text):
    rotten = [(re.findall('\d\d?\d?%', p), re.findall('\d\.\d\/\d+|$', p),
               p.lower().find('rotten tomatoes')) for p in text.split('\n\n')]

    rating = next(((perc[0], rating[0]) for perc, rating, idx in rotten if
                   len(perc) == 1 and idx > -1), (None, None))

    wikicode = mwparserfromhell.parse(text)
    film = next((template for template in wikicode.filter_templates() 
                 if template.name.strip().lower() == 'infobox film'), None)
    if film:
        properties = {param.name.strip_code().strip(): param.value.strip_code().strip() 
                      for param in film.params
                      if param.value.strip_code().strip()
                     }
        links = [x.title.strip_code().strip() for x in wikicode.filter_wikilinks()]
        return (title, properties, links) + rating

class WikiXmlHandler(xml.sax.handler.ContentHandler):
    def __init__(self, out_file):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._movies = []
        self._curent_tag = None
        self.fd_out = open(out_file, "w")

    def characters(self, content):
        if self._curent_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        if name in ('title', 'text'):
            self._curent_tag = name
            self._buffer = []

    def endElement(self, name):
        if name == self._curent_tag:
            self._values[name] = ' '.join(self._buffer)

        if name == 'page':
            #print(self._values)
            movie = process_article(**self._values)
            if movie:
                self.fd_out.write(json.dumps(movie) + '\n')
                #self._movies.append(movie)

class ParserWiki:
    def __init__(self, out_file):
        self.parser = xml.sax.make_parser()
        self.handler = WikiXmlHandler(out_file)
        self.parser.setContentHandler(self.handler)

    def parse_wiki_dump(self, dump_bz_file):
        for line in subprocess.Popen(['bzcat'], stdin=open(dump_bz_file), stdout=subprocess.PIPE).stdout:
            #print(line)
            self.parser.feed(line)
            #break

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python wii_extraxt_movie_data.py <input_wiki_dump> <output.ndjson>")
        sys.exit(1)
    p_wiki = ParserWiki(sys.argv[2])
    p_wiki.parse_wiki_dump(sys.argv[1])
