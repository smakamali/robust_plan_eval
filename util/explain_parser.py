import re
from util.query_parser import parse_query

# Helper function to return 'n' sized chunks from 'l'
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Helper function to return string between 'first' and 'last' in 's'
def find_between(s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def remove_selectivity(query):
    # This regex matches " SELECTIVITY <number>" where <number> can be an optional plus sign and a decimal number.
    pattern = r'\s+SELECTIVITY\s+\+?\d+(\.\d+)?'
    cleaned_query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    return cleaned_query

class ExplainParser:
    def __init__(self, raw_text):
        self.raw_text = raw_text

    #returns a hash with parsed values from a long explain as retrieved from db2
    def parse(self):
        # initialize JSON
        self.json = {}
        
        # Load optimized statment
        cut_text = self.raw_text.rsplit('Optimized Statement:\n-------------------',-1)[1]
        op_statment = cut_text.rsplit('Access Plan:',-1)[0]
        op_statment = remove_selectivity(op_statment.strip())
        
        # Create a dictionary of table aliases
        tab_alias_dic,_,_,_ = parse_query(op_statment)
        self.json['tab_alias_dic'] = tab_alias_dic

        return self.json

