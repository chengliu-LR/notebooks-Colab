import torch

###3.3 Text
with open("../DL_PyTorch/dlwpt-code-master/data/p1ch4/jane-austin/1342-0.txt") as f:
    text = f.read() #with包含异常处理
lines = text.split('\n')
line = lines[1000]
letter_tensor = torch.zeros(len(line), 128) #next will set this vector of 1 at the index corresponding to the location of the character in the encoding
for i, letter in enumerate(line.lower().strip()):    #lowercase; remove space before and at the end of the string
    letter_index = ord(letter) if ord(letter) < 128 else 0  #ord(): return the corresponding ASCII or UniCode
    letter_tensor[i][letter_index] = 1


def extract_words(input_str):
    punctuation = '.,;:"!?“”_-'
    word_list = input_str.lower().replace('\n', ' ').split()    #split: split every null str including \t; \n; ' ' and so on
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

words_in_single_line = extract_words(line)
print("the original single line:\n", line, "\n\nthe words in single line:\n", words_in_single_line)

word_list = sorted(set(extract_words(text)))    #set(): create an unorderd, not repeated set
word2index_dict = {word : i for (i, word) in enumerate(word_list)}
print("\nthe index of word 'impossible' in the word_index_dictionary:", word2index_dict['impossible'])
print("the length of the word_index_dictionary:", len(word2index_dict), '\n')

word_tensor = torch.zeros(len(words_in_single_line), len(word2index_dict))
for i, word in enumerate(words_in_single_line):
    word_index = word2index_dict[word]
    word_tensor[i][word_index] = 1  #assign the one-hot encoded values of the word from the sentence (one-hot)
    print("{:2} {:4} {}".format(i, word_index, word))

print("\nthe size of the generated word_tensor of this single line:", word_tensor.shape)    #at this point, word_tensor represents one sentence of length len(line) in an encoding space of size 7261 -- the number of words in your dictionary
