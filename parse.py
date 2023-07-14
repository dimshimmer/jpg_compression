import os

# file_name = input(f'Input the name of the file')
class parser(object):
    def __init__(self,file_name) -> None:
        self.file = os.path.join(os.getcwd(),file_name)
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

        with open(self.file,'rb') as f:
            file_content = f.read()
        # print(file_content)
        self.sections = {}
        last_sec = 0
        for ind,c in enumerate(file_content):
            
            if c == b"\xff" and last_sec > 0: 
                self.sections[self.tag_map[file_content[last_sec + 1]]] = file_content[last_sec,ind]
                last_sec = ind
        print(f'The len of the file is:{len(file_content)}')
        print(len(self.sections))
        for k in self.sections:
            print(f'{k} : {len(self.sections[k])} \tbytes')
        
par = parser('./more_data_19.jpg')
            
        
        