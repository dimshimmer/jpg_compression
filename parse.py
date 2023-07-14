import os
from copy import deepcopy
from typing import Any
from Simp_com import Image_Com

# file_name = input(f'Input the name of the file')
class parser(object):
    def __init__(self) -> None:
        
        tag_map = {
            "ffd8": "SOI",
            "ffc4": "DHT",
            'ffc8': "JPG",
            'ffcc': "DAC",
            "ffda": "SOS",
            "ffdb": "DQT",
            "ffdc": "DNL",
            "ffdd": "DRI",
            "ffde": "DHP",
            "ffdf": "EXP",
            "fffe": "COM",
            "ff01": "TEM",
            "ffd9": "EOI"
        }

        for i in range(16):
            tag = 'ffe%x' % i
            tag_map[tag] = 'APP'+str(i)

        for i in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:
            tag = 'ffc%x' % i
            tag_map[tag] = 'SOF'+str(i)

        for i in range(8):
            tag = 'ffd%x' % i
            tag_map[tag] = 'RST'+str(i)

        for i in range(14):
            tag = 'fff%x' % i
            tag_map[tag] = 'JPG'+str(i)

        map_tag = { v: k for k, v in tag_map.items() }
        self.tag_map = tag_map

       

    def __call__(self, file_name) -> Any:
        file_path = os.path.join(os.getcwd(),file_name)
        with open(file_path,'rb') as f:
            file_content = f.read()
            file_content = [hex(v).replace('0x','').zfill(2) for v in file_content]
        
        # print(file_content[89:102])
        # print(file_content[102:135])
        # print(file_content[135:318])
        
        # print(file_content[-20:])
        # print(file_content[0])
        self.sections = {}
        last_sec = 0
        # print(self.tag_map['ffd9'])
        for k in self.tag_map:
            self.sections[self.tag_map[k]] = []

        for ind,c in enumerate(file_content):
            if c == 'ff' and ind > 0 and (c + file_content[ind + 1]) in self.tag_map: 
                # print(f'{last_sec} - {ind}')
                self.sections[self.tag_map['ff'+file_content[last_sec + 1]]].append(file_content[last_sec:ind])
                last_sec = ind
                print(ind)
        keys = [k[1] for k in self.tag_map.items()]
        # print(keys)
        for k in keys:
            if len( self.sections[k]) == 0: del self.sections[k]
        print(f'The len of the file is:{len(file_content)}')
        # print(len(self.sections))
        print(self.sections['SOI'])
        for k in self.sections:
            for l in self.sections[k]:
                print(f'{k} : {len(l)} \tbytes')
    def steg(self, msg):
        com = Image_Com()
par = parser()
par('./gray_data.jpg')

'''
TODO:
1. Parse the sections
    1. Get the huffman table
    2. Get the RLC
    3. Get the RGB matrix
    4. Apply the steg function
'''
            
        
        