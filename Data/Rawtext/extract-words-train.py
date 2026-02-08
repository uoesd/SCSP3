
import os
from os import listdir
import os.path
import re
import string
import collections
import math
from collections import Counter


inputdirectory = "/Users/ross/Dropbox/Teaching/Edinburgh/Statistical Case Studies/2025-2026/Assignmennt1/Data/Rawtext/Gemini/"
outputdirectory = "/Users/ross/Dropbox/Teaching/Edinburgh/Statistical Case Studies/2025-2026/Assignmennt1/Data/FunctionWords/Gemini/"
frequentwordsfile = "/Users/ross/Dropbox/Teaching/Edinburgh/Statistical Case Studies/2025-2026/Assignmennt1/wordfile200.txt"

frequentwords70 = []
frequentwords = []


maxwords = 5000000 #maximum size of training set

fin = open(frequentwordsfile,'r')
for line in fin:
	frequentwords.append(line.split(',')[0].strip())
	

for i in range(0,len(frequentwords)):
	frequentwords70.append(frequentwords[i])



def only_printable_ascii(string):
    ''' Returns the string without non-ASCII characters and non-printable ASCII'''
    stripped = (c for c in string if 31 < ord(c) < 127)
    return ''.join(stripped)
    


#takes a chunk of words and returns vectors containing the counts of word lenghts, function words, common words etc
def createvector(wordlist):
	wordcounter = Counter()
	numwords = 0
	for word in wordlist:	
		if len(word) > 0:	
			wordcounter[word] += 1
			numwords+=1
		else:
			print('Error: zero length words')		
		
	frequentwords70vec = []
	frequentwords70count = 0
	
	for word in frequentwords70:
		frequentwords70vec.append(wordcounter[word])
		frequentwords70count = frequentwords70count+wordcounter[word]
		
	frequentwords70vec.append(numwords-frequentwords70count)

	str = ""
	for word in frequentwords70:
		if wordcounter[word] == 0:
			str = str + ", " + word 
	print(str)

	return (frequentwords70vec)



print("Processing files...")
outputdir =  outputdirectory 
if not os.path.exists(outputdir):
	os.makedirs(outputdir)
		

files = listdir(inputdirectory)
files.sort() #puts unknown at the end


for text in files:
	if text == ".DS_Store":
		continue

	frequentwordsfile70 = open(outputdir+'/'+text +" --- frequentwords.txt", 'w')
		
	with open(inputdirectory  + '/' +  text, 'r') as myfile:
			
		allwords = []
		numwords = 0 
			
		for line in myfile:
			if numwords == maxwords:
				break
			line = only_printable_ascii(line)											#strips other non-english characters
			line =  re.sub('[,\.]',' ',line)											#replace . and , with spaces
			line =  line.translate(string.maketrans("",""), string.punctuation) 		#delete all remaining punctuation										line = line.replace('\n', ' ').replace('\r', '')							#replaces new line with space									
			line = ' '.join(line.split())												#replace multiple spaces with a single one
			line = line.lower()		
		
			words = line.split(" ")													#break the line down into individual words by splitting on spaces
			for word in words:		
				word = word.replace(" ", "")										
				word = word.translate(None, string.whitespace)
				if word=='' or len(word)<1: 										#if we are left with an empty word after stripping out bad characrers, then throw it away
					continue		
				allwords.append(word)

				numwords = numwords+1
				if numwords==maxwords:
					break

		if numwords != maxwords:
			print(text + " max words not reached")
	
		#process in chunks
		frequentwords70vec = createvector(allwords)		
		frequentwordsfile70.write(str(frequentwords70vec).strip('[]')  + "\n")


    
	