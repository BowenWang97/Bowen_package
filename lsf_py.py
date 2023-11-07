import linecache
import os

filename = 'SOI_parallel_grating_coupler_2d'

path = os.path.dirname(__file__)

lsf_file = open(path+'\\'+filename+'.lsf','r')
lsf_file_lines = len(lsf_file.readlines())
lsf_file.close()

py_file = open(path+'\\'+filename+'.py','w')

for line_num in range(lsf_file_lines+1):    
    text = linecache.getline(path+'\\'+filename+'.lsf',line_num)
    text = text.replace('\n','')
    if (text == ''):
        continue
    py_file.write("    lsf_file.write(\'%s\\n\')\n"%text)

py_file.close()