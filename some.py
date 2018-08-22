import xml.dom.minidom
from bs4 import BeautifulSoup
import re

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def prettify(input_path, output_path):
    xm = xml.dom.minidom.parse(input_path)
    pretty_xml_as_string = xm.toprettyxml()

    with open(output_path, 'w') as file:
        file.write(pretty_xml_as_string)

class DetailsParser(object):

    def parse(self, details, borrowed):
        return None

    def description(self):
        return ""


class EnglishAndHebrewParser(DetailsParser):

    def __init__(self, english_key, hebrew_key):
        self.english_key = english_key
        self.hebrew_key = hebrew_key

    def parse(self, details, borrowed):

        if borrowed:
            return -1

        if len(details) < 2 or len(details) > 3:
            print('WARNING: Invalid length: %d' %len(details))
            return -2

        dic = {}
        for text in details[:-1]:
            sibling_text = text.lower()
            english_count = 0
            hebrew_count = 0
            for c in sibling_text:
                if 'a' <= c <= 'z' :
                    english_count += 1
                if 1488 <= ord(c) <= 1514:
                    hebrew_count += 1
            if english_count > hebrew_count:
                dic[self.english_key] = sibling_text
            else:
                dic[self.hebrew_key] = sibling_text
        return dic


class SimpleParser(DetailsParser):

    def __init__(self, key):
        self.key = key

    def parse(self, details, borrowed):

        if borrowed:
            return -1

        if len(details) != 2:
            print('WARNING: Invalid length: %d' %len(details))
            return -2

        return {self.key : details[0]}


class ForeignApplicationParser(DetailsParser):

    def parse(self, details, borrowed):

        if borrowed and len(details) == 3:
            return {'Foreign ID': details[2], 'Foreign Date': details[1], 'Foreign Country': details[0]}

        if len(details) != 6 or details[5] != '[31]' or details[3] != '[32]' or details[1] != '[33]':
            print('WARNING: wrong format')
            return None

        return {'Foreign ID': details[4], 'Foreign Date': details[2], 'Foreign Country': details[0]}



switcher = {'[54]' : EnglishAndHebrewParser('English Name', 'Hebrew Name'),
            '[22]' : SimpleParser('Application Date'),
            '[31]' : ForeignApplicationParser(),
            '[51]' : SimpleParser('Int C'),
            '[71]' : EnglishAndHebrewParser('English Applicant', 'Hebrew Applicant'),
            '[72]' : EnglishAndHebrewParser('English Inventor', 'Hebrew Inventor'),
            '[87]' : SimpleParser('International Publication Number'),
            '[74]' : EnglishAndHebrewParser('English Address For Service', 'Hebrew Address For Service'),
            '[62]' : SimpleParser('Divisional Application')}


def parse_details(details, last_parser):
    last_component = details[-1]

    borrowed = True
    parser = last_parser
    if re.match('\[[\d][\d]\]', last_component):
        borrowed = False
        parser = switcher.get(last_component)
        if parser is None:
            print("Could not find appropriate parser for %s" % last_component)
            return None, -3

    return parser.parse(details, borrowed), parser


def main():
    handler = open("/tmp/ap/doc.xml").read()
    soup = BeautifulSoup(handler, 'xml')
    count = 0

    for patent in soup.find_all(text="[21][11]"):
        details_dict = {}
        tr = patent.find_parent(name="w:tr")
        if tr is None:
            continue

        last_parser = None
        received_borrowing_error = False
        for sibling in tr.find_next_siblings(name="w:tr"):

            sibling_texts = []
            for inner_texts_container in sibling.find_all(name="w:tc"):
                sibling_texts.append("".join([t.text for t in inner_texts_container.find_all(name="w:t")]))
            sibling_texts = list(filter(('').__ne__, sibling_texts))
            if len(sibling_texts) <= 0:
                continue
            parsed_details, last_parser = parse_details(sibling_texts, last_parser)
            if parsed_details == -1:
                received_borrowing_error = True
            else:
                if received_borrowing_error:
                    print('WARNING: Borrowing error in the middle of parse')
                    continue

                if parsed_details is not None:
                    if isinstance(parsed_details, dict):
                        for k,v in parsed_details.items():
                            if details_dict.get(k) is None:
                                details_dict[k] = v
                            else:
                                if isinstance(details_dict[k], list):
                                    details_dict[k] = [v] + details_dict[k]
                                else:
                                    details_dict[k] = [v, details_dict[k]]
                    else:
                        print("WARNING: GOT ERROR: %d" % parsed_details)
                else:
                    print(sibling_texts)

        for k,v in details_dict.items():
            print("%s : %s" % (k,v))
        print('\n')


    print(count)

if __name__ == "__main__":
    print()
    #prettify("/tmp/PAT2/word/document.xml", "/tmp/ap/doc.xml")
    main()