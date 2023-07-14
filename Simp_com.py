from PIL import Image
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import random
import glob
import time

def create_tab(line):
    '''
    Auxiliary Function
    To create a format output for the line
    '''
    line_cnt = line.count('\n')
    v_s = line_cnt + 2
    
    lines = line.split('\n')
    maxl = 0
    for line in lines:
        if len(line) > maxl:
            maxl = len(line)
    h_line = '*' + '-'*(maxl + 2) + '*'
    
    print(h_line)
    print('| ' + ' ' * maxl + ' |')
    
    for line in lines:
        print('| ' + line + ' '*(maxl - len(line)) + ' |')
    print('| ' + ' ' * maxl + ' |')
    print(h_line)

class haff_tree(object):
    ''''
    The definition of the Huffman Tree node
    
    key: The original data
    weight: The frequency of the key appeared in the original file
    lchild: left child
    rchild: right child
    '''
    def __init__(self,key,weight ) -> None:
        self.key = key
        self.weight = weight
        self.lchild = None
        self.rchild = None

class Image_Com(object):
    '''
    The main class to compress, decompress and test the efficience
    '''
    def __init__(self) -> None:
        '''
        The initialization of the class:
        Generate the dct table 
        Store the quantilizeation tables
        '''
        create_tab("[Initial...]")
        self.dct_table = np.zeros(shape=(8,8))
        # Initialization the work path
        self.path = os.getcwd() + "\\"
        self.tab = [" "] * 8
        self.spider = spider()

        # Generate the DCT transform table, for the latter use
        # The DCT transformation is: F = dct · A · dct.T
        for i in range(8):
            if i == 0:
                c = np.sqrt(1/8)
            else:
                c = np.sqrt(2/8)
            for j in range(8):
                self.dct_table[i][j] = c * np.cos((2 * j + 1) * i * np.pi / 16)
        # The two quantitation matrixes
        self.cqt = np.array(
            [16, 11, 10, 16, 24, 40, 51, 61,
            12, 12, 14, 19, 26, 58, 60, 55,
            14, 13, 16, 24, 40, 57, 69, 56,
            14, 17, 22, 29, 51, 87, 80, 62,
            18, 22, 37, 56, 68, 109, 103, 77,
            24, 35, 55, 64, 81, 104, 113, 92,
            49, 64, 78, 87, 103, 121, 120, 101,
            72, 92, 95, 98, 112, 100, 103, 99,]).reshape((8,8))

        self.lqt = np.array(
            [
            17, 18, 24, 47, 99, 99, 99, 99,
            18, 21, 26, 66, 99, 99, 99, 99,
            24, 26, 56, 99, 99, 99, 99, 99,
            47, 66, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
            99, 99, 99, 99, 99, 99, 99, 99,
        ]
        ).reshape((8,8))
        
        # Use the numpy specificatoin to speed up
        self.qt_num = np.dstack((self.lqt,self.cqt,self.cqt))

        # The Zig-Zag transformation, to let the matrix to be Zig-Zag
        # transformed
        self.zigt = np.array([
            0, 1, 8, 16, 9, 2, 3, 10,
            17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
            ],dtype=np.uint8)
        self.zagt = np.array([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 41, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
            ], dtype= np.uint8)
        
        # Two matrixes, to transform from rgb to yuv, and yuv to rgb
        self.rgb2yuv = np.array([[0.299,0.587,0.114],
                                 [-0.14713,-0.28886,0.436],
                                 [0.651,-0.51499,-0.10001]]).reshape((3,3))
        
        self.yuv2rgb = np.array([[1.0,0,1.13983],
                                [1.0,-0.39465,-0.58059],
                                [1.0,2.03211,0]]).reshape((3,3))
        
        create_tab(">>>[Initial done!]")
    '''
    For the following process
    The yuv blocks is arranged as:
    y: yuv_blocks[:,:,0]
    u: yuv_blocks[:,:,1]
    v: yuv_blocks[:,:,2]
    '''

    def __YUV444_422__FIN__(self,yuv_block):
        '''
        The function to transform YUV444 to YUV422
        Merge the u and v matrixes to one matrix
        '''

        res = np.zeros(shape=(8,8,yuv_block.shape[2] // 3 * 2))
        res[:,:,::2] = yuv_block[:,:,::3]
        res[:,::2,1::2] = yuv_block[:,::2,1::3]
        res[:,1::2,1::2] = yuv_block[:,::2,2::3]
        yuv_block[:,1::2,1::3] = 0
        yuv_block[:,1::2,2::2] = 0
        
        return res
    def __YUV422_444__FIN__(self, y_uv):
        '''
        The function to transform YUV422 to YUV444
        Reconstruct the u and v matrixes, from the merged matrix
        '''
        
        res = np.zeros(shape=(8,8,y_uv.shape[2] // 2 * 3),dtype=  y_uv.dtype)
        res[:,:,::3] = y_uv[:,:,::2]
        res[:,::2,1::3] = y_uv[:,::2,1::2]
        res[:,::2,2::3] = y_uv[:,1::2,1::2]
        
        return res
    
    def __Zig__FIN__(self, mat):
        '''
        Transform three matrixes using the Zig transformation
        Use the numpy, to vectorize the process
        Transform three matrixes the same time
        '''
        res = np.zeros(shape= (64,mat.shape[2])).astype(np.uint8)
        matrix = mat.reshape((64,mat.shape[2])).astype(np.uint8)
        for i, ind in enumerate(self.zigt):
            res[i,:] = matrix[ind,:]
        
        return res
    
    
    def __Haff_dfs__(self, tab, root,tmp):
        '''
        Transpose the huffman tree, to construct the 
        huffman table
        '''
        if root.lchild != None or root.rchild != None:
            self.__Haff_dfs__(tab, root.lchild, tmp + "0")
            self.__Haff_dfs__(tab, root.rchild, tmp + "1")
        else:
            tab[root.key] = tmp
    def __RLC__FIN__(self, mat):
        '''
        Transform the matrix using run length coding
        Every time process 64 pixels, i.e. one block
        '''
        inds = np.array(range(64))
        ind = (mat != 0)
        res = []
        
        for i in range(mat.shape[1]):
            ind_tmp = inds[ind[:,i]]

            # Use the 0 padding
            if len(ind_tmp) == 0 or ind_tmp[-1] != 63:
                ind_tmp = np.append(ind_tmp,63)
            ind_cnt_tmp = np.append(ind_tmp,64) - np.insert(ind_tmp,0,-1) - 1
            tmp = np.zeros(ind_tmp.size * 2,dtype=mat.dtype)
            tmp[1::2] = mat[:,i][ind_tmp]
            tmp[::2] = ind_cnt_tmp[:-1]
            res.append(tmp)
        res = np.hstack(tuple(res)).astype(np.uint8)
        return res
                
    def __Haff__(self, c_cnt):
        '''
        According to the Frenquency table, construct the haff tree

        '''
        res = {}
        nodes = []
        for i, weight in enumerate(c_cnt):
            if weight > 0: nodes.append(haff_tree(i,weight))
        # Sort all the nodes, and construct the tree
        nodes.sort(key=lambda x:x.weight)
        while len(nodes) > 1:
            no_0 = nodes[0]
            no_1 = nodes[1]

            new_no = haff_tree(-1, no_0.weight + no_1.weight)
            new_no.lchild = no_0
            new_no.rchild = no_1
            nodes.pop(0)
            nodes.pop(0)
            i = 0
            for i in range(len(nodes)):
                if nodes[i].weight > new_no.weight:
                    break
            # Insert the least node to the first location
            nodes.insert(max(i,0),new_no)
        self.__Haff_dfs__(res,nodes[0],"")
        self.haf_root = nodes[0]

        return res
    def __DCT__FIN__(self, mat):
        '''
        DCT transformation:
        Use the broadcast feature of the numpy, to transform the whole matrix
        Transformed the matrix twice, to broadcast the dot calculation

        Parameters
        ----------
        mat : The yuv matrix

        Return
        ------
        The transformed matrix 
        '''
        mat_tmp = mat.swapaxes(2,0)
        tmp_0 = np.dot(self.dct_table,mat_tmp)
        tmp_0 = np.dot(tmp_0.swapaxes(1,0), np.transpose(self.dct_table))
        
        res = tmp_0.swapaxes(0,2)

        return res

    
    def __QUAN__FIN__(self,mat, method):
        '''Do the Quantization
        If method == 1:
           use the self.lqt to do the quantization
        If method == 2:
           use the self.cqt to do the quantization
        Parameters
        ----------
        mat : the input matrix to be do quantization
        method : depends the mateix to be used

        Return
        ------
        The Quantized matrix
        '''
        res = np.zeros(shape=mat.shape, dtype=mat.dtype)
        if method == 1:
            
            res[:,:,::3] = np.round(mat[:,:,::3] / self.lqt.reshape((8,8,1)))
            res[:,:,1::3] = np.round(mat[:,:,1::3] / self.cqt.reshape((8,8,1)))
            res[:,:,2::3] = np.round(mat[:,:,2::3] / self.cqt.reshape((8,8,1)))
            
        elif method == 2:
            res = np.round(mat[:,:,::2] / self.qt_num[:,:,:2])
        else:
            create_tab("Wrong option")
            exit()
        return res.astype(np.uint8)

    
    def __Fill_Div__NUM__(self,mat):
        '''
        Fill the block, to the multiple of 8
        And arrange them, to the third dimension, instead a list
        '''
        pad_h = 0
        pad_w = 0
        h, w = (mat.shape[0],mat.shape[1])

        if h % 8 != 0:
            pad_h = 8 - h % 8
        if w % 8 != 0:
            pad_w = 8 - w % 8
        
        tmp = np.pad(mat, ((0,pad_h), (0,pad_w),(0,0)),'constant')

        tmp_res = []

        tmp_h = np.split(tmp,tmp.shape[0]//8,0)  
        for arr in tmp_h:
            block = np.split(arr,tmp.shape[1]//8,1)
            tmp_res.extend(block)

        res = np.dstack(tuple(tmp_res))

        return res
 
    
    def comby_444_FIN(self,yuv_blocks,show_blocks):
        '''
        Use the YUV444 to compress a picture

        Parameters:
        -----------
        yuv_blocks: cutted blocks, stack all matrixes on the third dimension
        show_blocks: show the lossy processed picture, see the effection
        '''
        dct_blocks = self.__DCT__FIN__(yuv_blocks)
        quan_blocks = self.__QUAN__FIN__(dct_blocks,1)
        # append the precessed matrix
        show_blocks.append(quan_blocks)
        zig_blocks = self.__Zig__FIN__(quan_blocks)
        res = self.__RLC__FIN__(zig_blocks)

        return res
     
    def comby_422_FIN(self,yuv_blocks,show_blocks):
        '''
        Use the YUV422 to compress a picture

        Parameters:
        -----------
        yuv_blocks: cutted blocks, stack all matrixes on the third dimension
        show_blocks: show the lossy processed picture, see the effection
        '''
        dct_blocks = self.__DCT__FIN__(yuv_blocks)
        quan_blocks = self.__QUAN__FIN__(dct_blocks,1)
        show_blocks.append(quan_blocks)
        y_uv_q = self.__YUV444_422__FIN__(quan_blocks)
        zig_blocks = self.__Zig__FIN__(y_uv_q)
        res = self.__RLC__FIN__(zig_blocks)

        return res

    
    def com_one_pic(self,img, img_name,target_dir,show = 0, mode = 1):
        '''
        Compress one pictures

        Parameters:
        -----------
        img : the meta data of the input picture
        img_name : the name of the img file
        target_dir : saved directory
        show : if 0: don't show the picture; if 1: show the compressed picture
        mode : if 1: use YUV444;if 2: use YUV422
        '''
        RGB_ = np.array(img)
        height = RGB_.shape[0]
        width = RGB_.shape[1]
        print(f'\n*****----- Process pic {img_name} -----*****')
        yuv = self.__RGB2YUV__NUM__(RGB_)

        # Fill the blocks to the multiple of 8
        # Divid the whole block, to the blocks of 8x8
        yuv_blocks = self.__Fill_Div__NUM__(yuv)
        show_blocks = []

        T1 = time.clock()
        print("[Compression start]")
        if mode == 2:
            # Use the YUV422 to compress
            
            all_blocks_num = self.comby_422_FIN(yuv_blocks,show_blocks)
            print("[Use YUV422 to compress]")

        else:
            # Use the YUV444 to compress
            
            all_blocks_num = self.comby_444_FIN(yuv_blocks,show_blocks)
            print("[Use YUV444 to compress]")
        T2 =time.clock()
        print('[Time of compressing :%s s]' % ((T2 - T1)))

        haff_cnt = np.zeros(256, dtype=np.int32)
        
        # construct the word frequency table
        for c in all_blocks_num:
            haff_cnt[c] += 1
        
        # construct the huffman coding table
        haff_table = self.__Haff__(haff_cnt.tolist())
        bit_unit = ""
        
        cnt = 1024
        f_name = target_dir + "\\" + os.path.splitext(img_name )[0] + "_{}.d1m".format(mode)

        # save the output
        with open(f_name, "wb") as outf:
            # save meta data: height, width and the frequency table
            outf.write(height.to_bytes(2,byteorder = 'big'))
            outf.write(width.to_bytes(2,byteorder = 'big'))
            for cod in haff_cnt.astype(np.int32):
                outf.write(int(cod).to_bytes(4,byteorder = 'big'))
                            
            outf.flush()
            # Save the encoded content
            for term in all_blocks_num:
                cod = haff_table[term]
                bit_unit = cod + bit_unit
                while len(bit_unit) >= 8:
                    byte2_write = int(bit_unit[len(bit_unit) - 8:],2)
                    bit_unit = bit_unit[:len(bit_unit) - 8]  
                    cnt += 1
                    outf.write(byte2_write.to_bytes(1,byteorder='big'))
                    outf.flush()
            # Padding the content and save the mode and the padding length
            if len(bit_unit) > 0:
                length = (8 - len(bit_unit))
                bit_unit = "0" * length + bit_unit
                byte2_write = int(bit_unit,2)
                outf.write(byte2_write.to_bytes(1,byteorder='big'))
                outf.flush()
                outf.write(length.to_bytes(1,byteorder='big'))
                outf.flush()
                outf.write(mode.to_bytes(1,byteorder='big'))
                outf.flush()
                cnt += 3
            else:
                a = 0
                outf.write(a.to_bytes(1,byteorder='big'))
                outf.flush()
                outf.write(mode.to_bytes(1,byteorder='big'))
                outf.flush()
                cnt += 2
         
        # Prepare for showing the picture
        iquan = self.__IQUAN__FIN__(show_blocks[0],1)
        yuv_blocks = self.__IDCT__FIN__(iquan)
        yuv = self.__Cons_Dfill__FIN__(yuv_blocks,height,width)
        RGB_ = self.__YUV2RGB__NUM__(yuv) 

        img = Image.fromarray(np.uint8(RGB_)).convert('RGB')
        if show == 1:
            img.show()

        print("*****-----! Compressing Done !-----*****")
        img.save(target_dir+"\\" + os.path.splitext(img_name)[0] + "_d1m_sh1mm32_{}.bmp".format(mode))
        img.save(target_dir+"\\" + os.path.splitext(img_name)[0] + "_d1m_sh1mm32_{}.jpg".format(mode))
        return os.path.getsize(f_name)
    
    def __RGB2YUV__NUM__(self, img):
        '''
        Transform from RGB to YUV
        '''
        res = img @ self.rgb2yuv.T
        return res

    def com_pics(self, src, target):
        '''
        Comoress all the pictures in the src directory, and save the results in the target directory
        '''

        path = os.getcwd()
        dir_list = os.listdir(self.path  + src)
        self.mkdir(path + target)
        res = []
        create_tab("[Please choose the compression level]:\n[1]. bigger file, more accuracy\n[2]. smaller file, less accuracy")
        
        choice = (int(input("[ 1 / 2 ?]: ")))
        # Create the target directory
        self.mkdir(self.path + target)
        
        for img_name in dir_list:
            # Identify the valid file name
            if img_name.endswith((".bmp",".jpg",".gif",".jpeg",".tiff",".png")):
                img_path = path + "\\"+src + "\\"+img_name
                img = Image.open(img_path).convert("RGB")
                self.com_one_pic(img, img_name, self.path + target,mode = choice)
    
    def __RLC2_blocks__FIN__(self, rlc_arr,choice):
        '''
        Transform the rlc code to blocks
        Reconstruct the blocks from rlc array
        '''
        cnt = 0
        tmp_li = []
        all_blocks = []
        rlc_li = rlc_arr.tolist()
        for i in range(0,len(rlc_li),2):
            tmp_cnt = rlc_li[i] 
            tmp_data = rlc_li[i + 1]
            cnt = cnt + tmp_cnt + 1
            tmp_li.extend([0] * tmp_cnt)
            tmp_li.append(tmp_data)
            # every 64 element a block
            if cnt >= 64:
                all_blocks.append(np.array(tmp_li[:64]).astype(np.uint8))
                tmp_li = tmp_li[64:]
                cnt = cnt - 64
        
        
        res = np.dstack(tuple(all_blocks))
        return res
    
    
    def __IZig__FIN__(self, tmp_li):
        '''
        anti Zig transform
        '''
        res = np.zeros(shape=(64,tmp_li.shape[2])).astype(np.uint8)
        seq = tmp_li.reshape(64,tmp_li.shape[2]).astype(np.uint8)
        for i,ind in enumerate(self.zagt):
            res[i,:] = seq[ind,:]
        return res.reshape((8,8,tmp_li.shape[2])).astype(np.int8)
    
    
    
    def __IQUAN__FIN__(self,mat,method):
        '''
        Use the Quantilizaztion matrixes, to reconstruct the DCT results
        '''
        mat = mat.astype(np.int8)
        res = np.zeros(shape=mat.shape, dtype=np.float32)
        # YUV444 version
        if method == 1:
            res[:,:,::3] = np.round(mat[:,:,::3] * self.lqt.reshape((8,8,1)))
            res[:,:,1::3] = np.round(mat[:,:,1::3] * self.cqt.reshape((8,8,1)))
            res[:,:,2::3] = np.round(mat[:,:,2::3] * self.cqt.reshape((8,8,1)))
        # YUV422 version
        elif method == 2:
            res = mat * self.cqt
        else:
            create_tab("Wrong option")
            exit()

        return res.astype(np.float32)
    
    def __IDCT__FIN__(self, mat):
        '''
        Anti DCT transformation
        Transform the DCT processed matrix to the original matrix
        '''
        res = np.dot(self.dct_table.T, mat.swapaxes(2,0))
        res = np.dot(res.swapaxes(1,0), self.dct_table)

        return res.swapaxes(0,2)
    

    def __Cons_Dfill__FIN__(self, blocks,height, width):
        '''
        From the filled and divided blocks, construct the whole matrix 
        '''
        h_fill = height
        w_fill = width
        if h_fill % 8 != 0:
            h_fill = height + 8 - height % 8
        if w_fill % 8 != 0:
            w_fill = width + 8 - width % 8
        tmp_rlist = [blocks[:,:,i:i+3] for i in range(0, blocks.shape[2],3)]
        rlist = []

        for hi in range(h_fill // 8):
            start = hi * w_fill // 8
            rlist.append(np.hstack(tuple(tmp_rlist[start: start + (w_fill // 8)])))
        img = np.vstack(tuple(rlist))

        img = img[:height, :width,:]
        return img
    def __YUV2RGB__NUM__(self,yuv):
        '''
        Transform the YUV matrixes to RGB matrixes
        With the numpy broadcast specification
        '''
        res = yuv @ self.yuv2rgb.T
        res[res > 255] = 255
        res[res < 0] = 0
        return res

    def decom_show_pic(self, img_name):
        ''''
        decompress one picture and show the decompressed picture
        Parameters:
        -----------
        img_name : the name of the file, typically end with .d1m
        '''
        with open (self.path + img_name, "rb") as inf:
            inf.seek(0, 2)
            eof = inf.tell()
            inf.seek(0, 0)
            # read the height and width, and the frenquency table
            height = int.from_bytes( inf.read(2),byteorder= 'big')
            width = int.from_bytes(inf.read(2),byteorder='big')
            
            cnt_table = []
            for i in range(256):
                cnt_table.append(int.from_bytes(inf.read(4), byteorder='big'))

            haff_table = self.__Haff__(cnt_table)

            byte2_read = ""
            tmp_li = []

            # get all the huffman encoded data
            total_len = eof - inf.tell()
            byte2_read = ''.join(["{0:08b}".format(byte) for byte in reversed(inf.read())])
        create_tab("[Read done !]")
        com_choice = int(byte2_read[:8],2)

        byte2_read = byte2_read[8:]
        z_num = int(byte2_read[:8],2)

        byte2_read = byte2_read[z_num + 8:]
        haf = {}

        for i in range (256):
            if i in haff_table:haf[haff_table[i]] = i

        total_len = len(byte2_read)
        index = 0
        la = -1
        T1 = time.clock()
        buf = ''
        # transpose the data, and decode the huffman code
        for char in byte2_read:
            buf += char
            index += 1
            if buf in haf:
                tmp_li.append(haf[buf])
                buf = ''
                per = ((index * 100)// total_len)
                if per > la:
                    print("[Decoding Huffman]: ",">" * per + "-" * (100 - per),per,"%")
                    la = per
        tmp_li = tmp_li[::-1]
        T2 =time.clock()

        create_tab('[Time of decompressing :%s ms]' % ((T2 - T1)*1000))

        tmp = self.__RLC2_blocks__FIN__(np.array(tmp_li),com_choice)
        # judge the version: YUV444 or YUV422
        if com_choice == 2:
            print("[Decomby 422]")
            
            yuv = self.decomby_422_FIN(tmp,height,width)
        elif com_choice == 1:
            print("[Decomby 444]")
            yuv = self.decomby_444_FIN(tmp,height,width)
        else :
            create_tab("error")
            exit()
        RGB_ = self.__YUV2RGB__NUM__(yuv)
        img = Image.fromarray(np.uint8(RGB_)).convert('RGB')
        img.show()
    
    def decomby_444_FIN(self, all_blocks,height,width):
        '''
        Decompress the file by YUV444
        '''

        izig_blocks = self.__IZig__FIN__(all_blocks)
        iquan_blocks = self.__IQUAN__FIN__(izig_blocks,1)
        yuv_blocks = self.__IDCT__FIN__(iquan_blocks)
        yuv = self.__Cons_Dfill__FIN__(yuv_blocks,height,width)
        yuv = self.__Cons_Dfill__FIN__(yuv_blocks,height,width) 
        return yuv
    
    def decomby_422_FIN(self, all_blocks,height,width):
        '''
        Decompress the file by YUV422
        '''
        
        izig_blocks = self.__IZig__FIN__(all_blocks)
        izig_blocks = self.__YUV422_444__FIN__(izig_blocks)
        iquan_blocks = self.__IQUAN__FIN__(izig_blocks,1)
        yuv_blocks = self.__IDCT__FIN__(iquan_blocks)
        yuv = self.__Cons_Dfill__FIN__(yuv_blocks,height,width)

        return yuv
    
    
    def shell(self):
        '''
        The UI of the program
        '''
        cmd = ""
        self.help_ = "\
        [***----- Pleause choose the function -----***]\n\
        [1]. Compress one picture\n\
        [2]. Compress pictures in one directory\n\
        [3]. Decompress the target picture\n\
        [4]. Collect certain pics on Internet\n\
        [5]. Test and generate the report\n\
        [6]. ** Exit **\n\
        "
        while True:
            create_tab(self.help_)
            choice = int(input("[**Pleause input the function number**]:"))
            if choice == 1:
                pic_name = input("[Please input the name of the picture(under current directory)]:")
                create_tab("[Please choose the compression level]:\n[1]. bigger file, more accuracy\n[2]. smaller file, less accuracy")
                choice = (int(input("[ 1 / 2 ?]: ")))
                self.com_one_pic(Image.open(self.path + pic_name).convert('RGB'),pic_name,self.path,1,choice)

                
            elif choice == 2:
                src_name = input("[Please input the name of the directory(under current directory)]:")
                tar_name = input("[Please input the name of the target saved directory(under current directory)]:")
                
                self.com_pics(src_name,tar_name)
            elif choice == 3:
                src_name = input("[Please input the name of the compressed file(under current directory)]:")
            
                self.decom_show_pic(src_name)

            elif choice == 4:
                pics_type = input("[Please input the type of the pics you want to get]:")
                create_tab("*****[The pictures will be saved in the directory ./{}]*****".format(pics_type))
                self.spider.get_pics(pics_type,pics_type)
            
            elif choice == 5: 
                self.test()
                
            elif choice == 6: 
                break
        
        # clear all the decompressed data, generated in the using history
        # remember to save the target data
        clear = input("[Clear all the generated file ? ]:(yes/no)")
        if clear == 'yes':
            list1 = glob.glob('**/*d1m_sh1mm32*',recursive= True)
            list2 = glob.glob('**/*.d1m',recursive= True)
            list3 = glob.glob('test/*',recursive=True)
            for file in list1:
                try:
                    os.remove(file)
                except OSError as e:
                    create_tab("Error: %s : %s" % (file, e.strerror))
            for file in list2:
                try:
                    os.remove(file)
                except OSError as e:
                    create_tab("Error: %s : %s" % (file, e.strerror))
            for file in list3:
                try:
                    os.remove(file)
                except OSError as e:
                    create_tab("Error: %s : %s" % (file, e.strerror))
            
        create_tab("[ Good day !! ]")

    def mkdir(self,path):
        '''
        Function to make a directory
        '''
        jud = os.path.exists(path)
        if not jud:
            os.makedirs(path)
            create_tab("create successfully")
            return True
        else:
            create_tab("created")
            return False
    
    def test(self):
        '''
        The function to test the efficient of compressing, and generate a simple report: report_tmp.md
        '''
        path = self.path + "test"
        self.mkdir(path)
        path += "\\"
        dir_list = os.listdir(path)
        key_word = ["mountain", "sky", "star", "lion","monkey","bicycle","chocolate"]
        cnt = 50
        # collect 50 pictures randomly
        self.spider.get_pics("test",random.choice(key_word),cnt)
        
        ori_res = []
        comed_1_res = []
        comed_2_res = []
        comed_1_jpg = []
        comed_2_jpg = []
        write_rw = []
        write_rw.append(f'| Ori /Kb | jpg size/Kb | ratial | YUV444/Kb | ratial | YUV422/Kb | ratial |\n')
        write_rw.append(f'| ------- | ----------- | ------ | --------- | ------ | --------- | ------ |\n')
        dir_list = os.listdir(path)
        T1 = time.clock()
        for img_name in dir_list:
            img = Image.open(path + img_name).convert("RGB")
            ori_size = img.size[0] * img.size[1] * 3
            comed_1 = self.com_one_pic(img,img_name, path,mode = 1)
            comed_2 = self.com_one_pic(img,img_name, path,mode = 2)
            jpg_size = os.path.getsize(path + img_name)

            ori_res.append(ori_size)
            comed_1_res.append(comed_1)
            comed_2_res.append(comed_2)
            size_1 = os.path.getsize(path + os.path.splitext(img_name)[0] + "_d1m_sh1mm32_1.jpg")
            size_2 = os.path.getsize(path + os.path.splitext(img_name)[0] + "_d1m_sh1mm32_2.jpg")
            # write a table
            comed_1_jpg.append(size_1)
            comed_2_jpg.append(size_2)
            basic_f = '| {} | {} | {} | {} | {} | {} | {} |\n'
            ori_s = f'{ori_size // 1024}'.center(7,' ')
            jpg_s = f'{jpg_size // 1024}'.center(11,' ')
            s_1 = f'{size_1 // 1024}'.center(9,' ')
            s_2 = f'{size_2 // 1024}'.center(9,' ')
            
            r_0 = f'{round(jpg_size / ori_size * 100,2)}%'.center(6,' ')
            r_1 = f'{round(size_1 / ori_size * 100,2)}%'.center(6,' ')
            r_2 = f'{round(size_2 / ori_size * 100,2)}%'.center(6,' ')
            
            write_rw.append(basic_f.format(ori_s,jpg_s,r_0,s_1,r_1,s_2,r_2))

            
        T2 = time.clock()
        print("[The Time of the test is {} s]".format((T2 - T1) ))

        ori = np.array(ori_res)
        com1 = np.array(comed_1_res)
        com2 = np.array(comed_2_res)
        ratial1 = com1 / ori * 100
        ratial2 = com2 / ori * 100
        # generate a density picture of he distribution of the compression rate
        df = pd.DataFrame({"YUV444":ratial1, "YUV422": ratial2})
        df.plot.density()
        plt.savefig('test_data.jpg')

        with open("report_tmp.md","w",newline="") as f:
            for line in write_rw:
                f.write(line)
            f.write("\n")
            f.write("![]({})".format(self.path + "test_data.jpg"))

test = Image_Com()
test.shell()